"""
Evaluation script for JWST phase retrieval model.

This script loads a trained model and evaluates it on the test set,
generating comprehensive metrics and visualizations.

Usage:
    uv run python scripts/evaluate.py --checkpoint outputs/experiments/exp_xyz/checkpoints/best_model.pth
    uv run python scripts/evaluate.py --checkpoint outputs/experiments/exp_xyz/checkpoints/best_model.pth --test-path data/processed/test.npz
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from neural_wavefront.utils.config import load_config
from neural_wavefront.data.dataset import ZernikePSFDataset
from neural_wavefront.models.resnet import create_model
from neural_wavefront.training.metrics import MetricsTracker, per_mode_mae, per_mode_pearson
from neural_wavefront.optics import zernike, pupil, propagation


def visualize_predictions(
    psfs: np.ndarray,
    coeffs_true: np.ndarray,
    coeffs_pred: np.ndarray,
    save_path: str,
    n_samples: int = 6,
    config: dict = None
):
    """
    Visualize PSFs with true vs predicted coefficients.
    
    Args:
        psfs: PSF images
        coeffs_true: Ground truth coefficients
        coeffs_pred: Predicted coefficients
        save_path: Where to save the figure
        n_samples: Number of samples to show
        config: Configuration dict for regenerating PSFs
    """
    fig, axes = plt.subplots(3, n_samples, figsize=(3*n_samples, 9))
    
    # Setup for PSF regeneration if config provided
    if config is not None:
        grid_size = config['simulation']['grid_size']
        n_modes = config['zernike']['n_modes']
        aperture = pupil.create_jwst_aperture(grid_size=grid_size)
        basis = zernike.generate_zernike_basis(n_modes=n_modes, grid_size=grid_size)
    
    for i in range(min(n_samples, len(psfs))):
        # Original PSF
        ax = axes[0, i]
        ax.imshow(np.log10(psfs[i] + 1e-10), cmap='viridis')
        ax.set_title(f'Sample {i}\nObserved PSF', fontsize=10)
        ax.axis('off')
        
        # True coefficients PSF (regenerated)
        ax = axes[1, i]
        if config is not None:
            phase_true = zernike.combine_zernike_modes(basis, coeffs_true[i])
            psf_true = propagation.compute_psf(aperture, phase_true)
            ax.imshow(np.log10(psf_true + 1e-10), cmap='viridis')
            ax.set_title(f'True Coeffs\nMAE: {np.abs(coeffs_true[i]).mean():.2f} rad', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'Config\nRequired', ha='center', va='center')
            ax.set_title('True PSF', fontsize=10)
        ax.axis('off')
        
        # Predicted coefficients PSF
        ax = axes[2, i]
        if config is not None:
            phase_pred = zernike.combine_zernike_modes(basis, coeffs_pred[i])
            psf_pred = propagation.compute_psf(aperture, phase_pred)
            ax.imshow(np.log10(psf_pred + 1e-10), cmap='viridis')
            error = np.abs(coeffs_pred[i] - coeffs_true[i]).mean()
            ax.set_title(f'Predicted Coeffs\nMAE: {error:.2f} rad', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'Config\nRequired', ha='center', va='center')
            ax.set_title('Predicted PSF', fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved predictions visualization: {save_path}")


def plot_coefficient_scatter(
    coeffs_true: np.ndarray,
    coeffs_pred: np.ndarray,
    save_path: str
):
    """
    Create scatter plots of true vs predicted coefficients.
    
    Args:
        coeffs_true: Ground truth coefficients, shape (n_samples, n_modes)
        coeffs_pred: Predicted coefficients
        save_path: Where to save the figure
    """
    n_modes = coeffs_true.shape[1]
    
    # Select modes to display (first 9)
    modes_to_plot = min(9, n_modes)
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flat
    
    for i in range(modes_to_plot):
        ax = axes[i]
        
        # Scatter plot
        ax.scatter(coeffs_true[:, i], coeffs_pred[:, i], alpha=0.5, s=10)
        
        # Perfect prediction line
        min_val = min(coeffs_true[:, i].min(), coeffs_pred[:, i].min())
        max_val = max(coeffs_true[:, i].max(), coeffs_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
        
        # Compute correlation
        from scipy.stats import pearsonr
        corr, _ = pearsonr(coeffs_true[:, i], coeffs_pred[:, i])
        mae = np.mean(np.abs(coeffs_true[:, i] - coeffs_pred[:, i]))
        
        ax.set_xlabel('True Coefficient (rad)', fontsize=10)
        ax.set_ylabel('Predicted Coefficient (rad)', fontsize=10)
        ax.set_title(f'Z_{i+1} (Noll)\nr={corr:.3f}, MAE={mae:.3f}', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    # Hide unused subplots
    for i in range(modes_to_plot, 9):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved coefficient scatter plots: {save_path}")


def plot_per_mode_metrics(
    mae_per_mode: np.ndarray,
    pearson_per_mode: np.ndarray,
    save_path: str
):
    """
    Plot per-mode MAE and Pearson correlation.
    
    Args:
        mae_per_mode: MAE for each mode
        pearson_per_mode: Pearson correlation for each mode
        save_path: Where to save the figure
    """
    n_modes = len(mae_per_mode)
    mode_indices = np.arange(1, n_modes + 1)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # MAE per mode
    ax1.bar(mode_indices, mae_per_mode, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Zernike Mode (Noll Index)', fontsize=12)
    ax1.set_ylabel('Mean Absolute Error (rad)', fontsize=12)
    ax1.set_title('Per-Mode Prediction Error', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticks(mode_indices)
    
    # Pearson correlation per mode
    ax2.bar(mode_indices, pearson_per_mode, color='coral', alpha=0.7)
    ax2.set_xlabel('Zernike Mode (Noll Index)', fontsize=12)
    ax2.set_ylabel('Pearson Correlation', fontsize=12)
    ax2.set_title('Per-Mode Correlation (True vs Predicted)', fontsize=14, fontweight='bold')
    ax2.axhline(y=0.9, color='g', linestyle='--', label='r=0.9')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(mode_indices)
    ax2.set_ylim([0, 1])
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved per-mode metrics: {save_path}")


def evaluate_model(
    model,
    test_loader,
    device: str,
    config: dict,
    output_dir: Path
):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on
        config: Configuration dict
        output_dir: Directory to save results
    """
    model.eval()
    metrics_tracker = MetricsTracker()
    
    all_psfs = []
    all_coeffs_true = []
    all_coeffs_pred = []
    
    print("\n🔍 Evaluating model on test set...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            psf = batch['psf'].to(device)
            coeffs_true = batch['coeffs'].to(device)
            
            # Predict
            coeffs_pred = model(psf)
            
            # Track metrics
            metrics_tracker.update(coeffs_pred, coeffs_true)
            
            # Store for visualization
            all_psfs.append(psf.cpu().numpy())
            all_coeffs_true.append(coeffs_true.cpu().numpy())
            all_coeffs_pred.append(coeffs_pred.cpu().numpy())
    
    # Compute metrics
    metrics = metrics_tracker.compute(prefix='test_')
    per_mode_metrics = metrics_tracker.get_per_mode_metrics()
    
    # Print results
    print("\n" + "="*70)
    print(" Test Set Results")
    print("="*70)
    print(f"Overall Metrics:")
    print(f"  MAE:  {metrics['test_mae']:.4f} rad")
    print(f"  RMSE: {metrics['test_rmse']:.4f} rad")
    print(f"  R²:   {metrics['test_r2']:.4f}")
    print(f"  Mean Pearson: {metrics['test_mean_pearson']:.4f}")
    print("="*70)
    
    # Concatenate all batches
    all_psfs = np.concatenate(all_psfs, axis=0)
    all_coeffs_true = np.concatenate(all_coeffs_true, axis=0)
    all_coeffs_pred = np.concatenate(all_coeffs_pred, axis=0)
    
    # Create visualizations
    print("\n📊 Generating visualizations...")
    
    # PSF predictions
    visualize_predictions(
        all_psfs[:, 0, :, :],  # Remove channel dimension
        all_coeffs_true,
        all_coeffs_pred,
        save_path=str(output_dir / "predictions.png"),
        n_samples=6,
        config=config
    )
    
    # Coefficient scatter
    plot_coefficient_scatter(
        all_coeffs_true,
        all_coeffs_pred,
        save_path=str(output_dir / "coefficient_scatter.png")
    )
    
    # Per-mode metrics
    plot_per_mode_metrics(
        per_mode_metrics['mae_per_mode'],
        per_mode_metrics['pearson_per_mode'],
        save_path=str(output_dir / "per_mode_metrics.png")
    )
    
    # Save numerical results
    results_path = output_dir / "test_results.npz"
    np.savez(
        results_path,
        metrics=metrics,
        mae_per_mode=per_mode_metrics['mae_per_mode'],
        pearson_per_mode=per_mode_metrics['pearson_per_mode'],
        coeffs_true=all_coeffs_true,
        coeffs_pred=all_coeffs_pred
    )
    print(f"✅ Saved numerical results: {results_path}")
    
    return metrics, per_mode_metrics


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate JWST phase retrieval model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="data/processed/test.npz",
        help="Path to test dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config (if not in checkpoint)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results"
    )
    args = parser.parse_args()
    
    print("="*70)
    print(" JWST Phase Retrieval - Model Evaluation")
    print("="*70)
    
    # Setup device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"\n🖥️  Using device: {device}")
    
    # Load checkpoint
    print(f"\n📂 Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Get config
    if args.config:
        config = load_config(args.config)
    else:
        config = checkpoint.get('config', {})
        print("   Using config from checkpoint")
    
    # Create model
    print(f"\n🏗️  Creating model...")
    model = create_model(
        model_name=config.get('model', {}).get('name', 'ResNet18'),
        n_modes=config.get('model', {}).get('output_dim', 15),
        pretrained=False,
        dropout=config.get('model', {}).get('dropout', 0.2)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"✅ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Load test dataset
    print(f"\n📁 Loading test dataset: {args.test_path}")
    test_dataset = ZernikePSFDataset(
        args.test_path,
        log_scale=config.get('visualization', {}).get('log_scale', True),
        normalize_coeffs=False,
        augment_rotation=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    # Setup output directory
    if args.output_dir is None:
        checkpoint_dir = Path(args.checkpoint).parent
        output_dir = checkpoint_dir.parent / "evaluation"
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📊 Results will be saved to: {output_dir}")
    
    # Evaluate
    metrics, per_mode_metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        config=config,
        output_dir=output_dir
    )
    
    print("\n" + "="*70)
    print("✅ Evaluation Complete!")
    print(f"   Results saved to: {output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
