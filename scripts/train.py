"""
Training script for JWST phase retrieval model.

This script loads generated datasets, initializes the model and trainer,
and runs the complete training pipeline.

Usage:
    uv run python scripts/train.py
    uv run python scripts/train.py --config configs/my_config.yaml
    uv run python scripts/train.py --resume outputs/experiments/exp_xyz/checkpoints/latest.pth
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path

from neural_wavefront.utils.config import load_config
from neural_wavefront.data.dataset import create_dataloaders
from neural_wavefront.models.resnet import create_model
from neural_wavefront.training.trainer import Trainer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train JWST phase retrieval model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detect if not specified"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Override experiment name from config"
    )
    args = parser.parse_args()
    
    print("="*70)
    print(" JWST Phase Retrieval - Model Training")
    print("="*70)
    
    # Load configuration
    print(f"\n📋 Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    if args.experiment_name:
        config['experiment_name'] = args.experiment_name
    
    # Setup device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"🖥️  Using device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Check if data exists
    data_dir = Path("data/processed")
    train_path = data_dir / "train.npz"
    val_path = data_dir / "val.npz"
    
    if not train_path.exists() or not val_path.exists():
        print(f"\n❌ ERROR: Training data not found!")
        print(f"   Expected: {train_path} and {val_path}")
        print(f"\n   Please generate data first:")
        print(f"   uv run python scripts/generate_data.py")
        return
    
    # Create dataloaders
    print(f"\n📁 Loading datasets...")
    # Use num_workers=0 on Windows to avoid multiprocessing issues
    import platform
    num_workers = 0 if platform.system() == 'Windows' else config['data'].get('num_workers', 4)
    
    train_loader, val_loader = create_dataloaders(
        train_path=str(train_path),
        val_path=str(val_path),
        batch_size=config['data']['batch_size'],
        num_workers=num_workers,
        log_scale=config['visualization'].get('log_scale', True),
        normalize_coeffs=False,  # Keep coefficients in radians
        augment_train=config['data']['augmentation'].get('enable', True)
    )
    
    # Create model
    print(f"\n🏗️  Creating model...")
    model = create_model(
        model_name=config['model']['name'],
        n_modes=config['model']['output_dim'],
        pretrained=config['model']['pretrained'],
        dropout=config['model']['dropout']
    )
    
    # Create optimizer
    print(f"\n⚙️  Setting up optimizer and loss...")
    optimizer_name = config['training']['optimizer']
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Create loss function
    loss_type = config['loss']['type']
    if loss_type == 'MSE':
        criterion = nn.MSELoss()
    elif loss_type == 'MAE' or loss_type == 'L1':
        criterion = nn.L1Loss()
    elif loss_type == 'Huber':
        criterion = nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    print(f"   Optimizer: {optimizer_name} (lr={lr:.2e})")
    print(f"   Loss: {loss_type}")
    
    # Create trainer
    print(f"\n🎯 Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
        device=device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\n📂 Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    n_epochs = config['training']['epochs']
    print(f"\n🚀 Starting training for {n_epochs} epochs...")
    print("="*70)
    
    try:
        trainer.train(n_epochs=n_epochs)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("   Saving checkpoint...")
        trainer.save_checkpoint(is_best=False, filename='interrupted.pth')
        print("   ✅ Checkpoint saved")
    
    # Training summary
    print("\n" + "="*70)
    print(" Training Summary")
    print("="*70)
    print(f"✅ Training completed successfully")
    print(f"   Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"   Final epoch: {trainer.current_epoch}")
    print(f"   Experiment dir: {trainer.experiment_dir}")
    print(f"\n📊 Next steps:")
    print(f"   1. Visualize training: tensorboard --logdir {trainer.experiment_dir}/tensorboard")
    print(f"   2. Evaluate model: uv run python scripts/evaluate.py --checkpoint {trainer.checkpoint_dir}/best_model.pth")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
