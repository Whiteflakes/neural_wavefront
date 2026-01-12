"""
Evaluation metrics for Zernike coefficient prediction.

This module provides metrics to evaluate the quality of predicted
Zernike coefficients and reconstructed PSFs.
"""

import torch
import numpy as np
from typing import Dict, Optional
from scipy.stats import pearsonr


def mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Mean Absolute Error.
    
    Args:
        predictions: Predicted values, shape (batch_size, n_modes)
        targets: Ground truth values, shape (batch_size, n_modes)
    
    Returns:
        Mean absolute error across all samples and modes
    """
    return torch.mean(torch.abs(predictions - targets)).item()


def rmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Root Mean Square Error.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
    
    Returns:
        Root mean square error
    """
    mse = torch.mean((predictions - targets) ** 2)
    return torch.sqrt(mse).item()


def r2_score(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    R² (coefficient of determination) score.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
    
    Returns:
        R² score (1.0 is perfect, 0.0 is baseline)
    """
    predictions = predictions.detach().cpu()
    targets = targets.detach().cpu()
    
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    return r2.item()


def per_mode_mae(predictions: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
    """
    Mean Absolute Error for each Zernike mode separately.
    
    Args:
        predictions: Predicted values, shape (batch_size, n_modes)
        targets: Ground truth values, shape (batch_size, n_modes)
    
    Returns:
        Array of MAE values, shape (n_modes,)
    """
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    
    mae_per_mode = np.mean(np.abs(predictions - targets), axis=0)
    return mae_per_mode


def per_mode_pearson(predictions: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
    """
    Pearson correlation coefficient for each Zernike mode.
    
    Args:
        predictions: Predicted values, shape (batch_size, n_modes)
        targets: Ground truth values, shape (batch_size, n_modes)
    
    Returns:
        Array of correlation coefficients, shape (n_modes,)
    """
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    
    n_modes = predictions.shape[1]
    correlations = np.zeros(n_modes)
    
    for i in range(n_modes):
        if np.std(predictions[:, i]) > 1e-8 and np.std(targets[:, i]) > 1e-8:
            corr, _ = pearsonr(predictions[:, i], targets[:, i])
            correlations[i] = corr
        else:
            correlations[i] = 0.0
    
    return correlations


def compute_all_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Compute all metrics at once.
    
    Args:
        predictions: Predicted Zernike coefficients
        targets: Ground truth Zernike coefficients
        prefix: Prefix for metric names (e.g., 'train_', 'val_')
    
    Returns:
        Dictionary of metric name -> value
    
    Example:
        >>> preds = torch.randn(100, 15)
        >>> targets = torch.randn(100, 15)
        >>> metrics = compute_all_metrics(preds, targets, prefix='val_')
        >>> print(metrics['val_mae'])
    """
    metrics = {
        f'{prefix}mae': mae(predictions, targets),
        f'{prefix}rmse': rmse(predictions, targets),
        f'{prefix}r2': r2_score(predictions, targets),
    }
    
    # Add per-mode metrics
    mae_per_mode = per_mode_mae(predictions, targets)
    pearson_per_mode = per_mode_pearson(predictions, targets)
    
    metrics[f'{prefix}mean_pearson'] = float(np.mean(pearson_per_mode))
    
    return metrics


class MetricsTracker:
    """
    Track metrics over multiple batches/epochs.
    
    Useful for accumulating metrics during training/validation.
    
    Example:
        >>> tracker = MetricsTracker()
        >>> for batch in dataloader:
        ...     preds, targets = model(batch), batch['coeffs']
        ...     tracker.update(preds, targets)
        >>> metrics = tracker.compute()
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all accumulated values."""
        self.predictions = []
        self.targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Add a batch of predictions and targets.
        
        Args:
            predictions: Predicted values
            targets: Ground truth values
        """
        self.predictions.append(predictions.detach().cpu())
        self.targets.append(targets.detach().cpu())
    
    def compute(self, prefix: str = "") -> Dict[str, float]:
        """
        Compute metrics over all accumulated batches.
        
        Args:
            prefix: Prefix for metric names
        
        Returns:
            Dictionary of metrics
        """
        if len(self.predictions) == 0:
            return {}
        
        # Concatenate all batches
        all_preds = torch.cat(self.predictions, dim=0)
        all_targets = torch.cat(self.targets, dim=0)
        
        # Compute metrics
        metrics = compute_all_metrics(all_preds, all_targets, prefix=prefix)
        
        return metrics
    
    def get_per_mode_metrics(self) -> Dict[str, np.ndarray]:
        """
        Get per-mode metrics.
        
        Returns:
            Dictionary with 'mae_per_mode' and 'pearson_per_mode'
        """
        if len(self.predictions) == 0:
            return {}
        
        all_preds = torch.cat(self.predictions, dim=0)
        all_targets = torch.cat(self.targets, dim=0)
        
        return {
            'mae_per_mode': per_mode_mae(all_preds, all_targets),
            'pearson_per_mode': per_mode_pearson(all_preds, all_targets)
        }


# Demo/test code
if __name__ == "__main__":
    print("="*60)
    print("Metrics Module Demonstration")
    print("="*60)
    
    # Create synthetic data
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_samples = 100
    n_modes = 15
    
    # Simulate predictions with some noise
    targets = torch.randn(n_samples, n_modes) * 2.0
    predictions = targets + torch.randn(n_samples, n_modes) * 0.3
    
    # Test individual metrics
    print("\n1. Testing individual metrics...")
    print(f"   MAE: {mae(predictions, targets):.4f}")
    print(f"   RMSE: {rmse(predictions, targets):.4f}")
    print(f"   R²: {r2_score(predictions, targets):.4f}")
    
    # Test per-mode metrics
    print("\n2. Testing per-mode metrics...")
    mae_modes = per_mode_mae(predictions, targets)
    pearson_modes = per_mode_pearson(predictions, targets)
    print(f"   MAE per mode (first 5): {mae_modes[:5]}")
    print(f"   Pearson per mode (first 5): {pearson_modes[:5]}")
    print(f"   Mean Pearson: {np.mean(pearson_modes):.4f}")
    
    # Test compute_all_metrics
    print("\n3. Testing compute_all_metrics...")
    metrics = compute_all_metrics(predictions, targets, prefix='test_')
    for name, value in metrics.items():
        print(f"   {name}: {value:.4f}")
    
    # Test MetricsTracker
    print("\n4. Testing MetricsTracker...")
    tracker = MetricsTracker()
    
    # Simulate batches
    batch_size = 20
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_preds = predictions[i:end_idx]
        batch_targets = targets[i:end_idx]
        tracker.update(batch_preds, batch_targets)
    
    # Compute metrics
    tracked_metrics = tracker.compute(prefix='tracked_')
    print(f"   Tracked MAE: {tracked_metrics['tracked_mae']:.4f}")
    print(f"   Tracked RMSE: {tracked_metrics['tracked_rmse']:.4f}")
    print(f"   ✅ MetricsTracker working correctly")
    
    # Test per-mode retrieval
    print("\n5. Testing per-mode metric retrieval...")
    per_mode = tracker.get_per_mode_metrics()
    print(f"   MAE per mode shape: {per_mode['mae_per_mode'].shape}")
    print(f"   Pearson per mode shape: {per_mode['pearson_per_mode'].shape}")
    
    print("\n" + "="*60)
    print("✅ Metrics module implementation complete!")
    print("="*60)
