"""
Training pipeline for JWST wavefront sensing model.

This module provides the Trainer class that handles the complete training loop,
validation, checkpointing, and logging.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import time
from typing import Dict, Optional, Tuple
import numpy as np

from neural_wavefront.training.metrics import MetricsTracker


class Trainer:
    """
    Trainer class for managing the training loop.
    
    Handles:
    - Training and validation loops
    - Model checkpointing (best and latest)
    - TensorBoard logging
    - Early stopping
    - Learning rate scheduling
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        config: Configuration dictionary
        device: Device to train on ('cuda' or 'cpu')
        experiment_dir: Directory for saving checkpoints and logs
    
    Example:
        >>> trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, config)
        >>> trainer.train(n_epochs=50)
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Dict,
        device: str = 'cuda',
        experiment_dir: Optional[str] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = device
        
        # Setup directories
        if experiment_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            experiment_dir = f"outputs/experiments/{config.get('experiment_name', 'exp')}_{timestamp}"
        
        self.experiment_dir = Path(experiment_dir)
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup TensorBoard
        if config.get('logging', {}).get('tensorboard', True):
            self.writer = SummaryWriter(log_dir=str(self.experiment_dir / "tensorboard"))
        else:
            self.writer = None
        
        # Setup learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.train_losses = []
        self.val_losses = []
        
        # Early stopping config
        early_stop_config = config.get('training', {}).get('early_stopping', {})
        self.early_stopping_patience = early_stop_config.get('patience', 10)
        self.early_stopping_min_delta = early_stop_config.get('min_delta', 1e-4)
        
        # Gradient clipping
        self.gradient_clip = config.get('training', {}).get('gradient_clip', None)
        
        print(f"✅ Trainer initialized")
        print(f"   Experiment dir: {self.experiment_dir}")
        print(f"   Device: {self.device}")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler based on config."""
        scheduler_type = self.config.get('training', {}).get('scheduler', 'ReduceLROnPlateau')
        
        if scheduler_type == 'ReduceLROnPlateau':
            params = self.config.get('training', {}).get('scheduler_params', {})
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=params.get('factor', 0.5),
                patience=params.get('patience', 5),
                min_lr=params.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'StepLR':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        else:
            return None
    
    def train_epoch(self) -> Tuple[float, Dict]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.train()
        metrics_tracker = MetricsTracker()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            psf = batch['psf'].to(self.device)
            coeffs_true = batch['coeffs'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            coeffs_pred = self.model(psf)
            loss = self.criterion(coeffs_pred, coeffs_true)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            self.optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            metrics_tracker.update(coeffs_pred, coeffs_true)
            
            # Log batch
            if self.writer is not None:
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('train/batch_loss', loss.item(), global_step)
        
        # Compute epoch metrics
        avg_loss = epoch_loss / len(self.train_loader)
        metrics = metrics_tracker.compute(prefix='train_')
        
        return avg_loss, metrics
    
    def validate(self) -> Tuple[float, Dict]:
        """
        Validate the model.
        
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.eval()
        metrics_tracker = MetricsTracker()
        epoch_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                psf = batch['psf'].to(self.device)
                coeffs_true = batch['coeffs'].to(self.device)
                
                # Forward pass
                coeffs_pred = self.model(psf)
                loss = self.criterion(coeffs_pred, coeffs_true)
                
                # Track metrics
                epoch_loss += loss.item()
                metrics_tracker.update(coeffs_pred, coeffs_true)
        
        # Compute metrics
        avg_loss = epoch_loss / len(self.val_loader)
        metrics = metrics_tracker.compute(prefix='val_')
        
        return avg_loss, metrics
    
    def save_checkpoint(self, is_best: bool = False, filename: str = 'checkpoint.pth'):
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
            filename: Checkpoint filename
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        # Save as best if needed
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"   💾 Saved best model (val_loss={self.best_val_loss:.4f})")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"✅ Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, n_epochs: int):
        """
        Main training loop.
        
        Args:
            n_epochs: Number of epochs to train
        """
        print("\n" + "="*70)
        print("Starting Training")
        print("="*70)
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, n_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train
            train_loss, train_metrics = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_metrics = self.validate()
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar('train/loss', train_loss, epoch)
                self.writer.add_scalar('val/loss', val_loss, epoch)
                self.writer.add_scalar('train/lr', current_lr, epoch)
                
                for name, value in train_metrics.items():
                    self.writer.add_scalar(name, value, epoch)
                for name, value in val_metrics.items():
                    self.writer.add_scalar(name, value, epoch)
            
            # Print progress
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch+1}/{n_epochs} ({epoch_time:.1f}s)")
            print(f"  Train - Loss: {train_loss:.4f} | MAE: {train_metrics['train_mae']:.4f} | R²: {train_metrics['train_r2']:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f} | MAE: {val_metrics['val_mae']:.4f} | R²: {val_metrics['val_r2']:.4f}")
            print(f"  LR: {current_lr:.2e}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss - self.early_stopping_min_delta
            
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint(is_best=True, filename=f'epoch_{epoch+1}.pth')
            else:
                self.epochs_without_improvement += 1
            
            # Save latest
            self.save_checkpoint(is_best=False, filename='latest.pth')
            
            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\n⚠️  Early stopping triggered (no improvement for {self.early_stopping_patience} epochs)")
                break
        
        # Training complete
        total_time = time.time() - start_time
        print("\n" + "="*70)
        print(f"✅ Training Complete!")
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"   Best val loss: {self.best_val_loss:.4f}")
        print(f"   Checkpoints saved to: {self.checkpoint_dir}")
        print("="*70 + "\n")
        
        if self.writer is not None:
            self.writer.close()


# Demo/test code
if __name__ == "__main__":
    print("="*60)
    print("Training Pipeline Demonstration")
    print("="*60)
    
    # This is a minimal test - full training requires generated data
    print("\n⚠️  Full training test requires generated dataset")
    print("   Run: uv run python scripts/generate_data.py --train-only")
    print("\nFor now, testing trainer initialization...")
    
    # Create dummy components
    from neural_wavefront.models.resnet import ZernikeRegressor
    
    model = ZernikeRegressor(n_modes=15, pretrained=False, dropout=0.2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Minimal config
    config = {
        'experiment_name': 'test_exp',
        'logging': {'tensorboard': False},
        'training': {
            'scheduler': 'ReduceLROnPlateau',
            'scheduler_params': {'factor': 0.5, 'patience': 5},
            'gradient_clip': 1.0,
            'early_stopping': {'patience': 10, 'min_delta': 1e-4}
        }
    }
    
    # Create dummy data loaders
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 20
        
        def __getitem__(self, idx):
            return {
                'psf': torch.randn(1, 256, 256),
                'coeffs': torch.randn(15),
                'idx': idx
            }
    
    train_loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=4)
    val_loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=4)
    
    # Test trainer initialization
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
        device='cpu'
    )
    
    print(f"\n✅ Trainer initialized successfully")
    print(f"   Can run: trainer.train(n_epochs=2) for quick test")
    
    # Quick test with 1 epoch
    print("\nRunning 1 test epoch...")
    trainer.train(n_epochs=1)
    
    print("\n" + "="*60)
    print("✅ Training pipeline implementation complete!")
    print("="*60)
