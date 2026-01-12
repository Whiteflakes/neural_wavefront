"""
Data generation script for JWST PSF dataset.

This script generates synthetic PSF images with known wavefront aberrations
for training and validation of the neural network.

Usage:
    uv run python scripts/generate_data.py
    uv run python scripts/generate_data.py --config configs/my_config.yaml
    uv run python scripts/generate_data.py --train-only  # Generate only training set
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from neural_wavefront.utils.config import load_config
from neural_wavefront.data.generator import generate_dataset, save_dataset


def visualize_samples(data: dict, save_path: str, n_samples: int = 6):
    """
    Create visualization of sample PSFs from the dataset.
    
    Args:
        data: Dataset dictionary with 'psf' and 'zernike_coeffs' keys
        save_path: Path to save the figure
        n_samples: Number of samples to visualize
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Sample PSFs from Generated Dataset', fontsize=16, fontweight='bold')
    
    for idx, ax in enumerate(axes.flat):
        if idx < n_samples and idx < len(data['psf']):
            psf = data['psf'][idx]
            coeffs = data['zernike_coeffs'][idx]
            
            # Display PSF in log scale
            im = ax.imshow(np.log10(psf + 1e-10), cmap='viridis')
            
            # Show statistics
            max_coeff_idx = np.abs(coeffs).argmax()
            max_coeff_val = coeffs[max_coeff_idx]
            
            ax.set_title(
                f'Sample {idx}\n' +
                f'Max coeff: Z_{max_coeff_idx+1} = {max_coeff_val:.2f} rad\n' +
                f'RMS: {np.std(coeffs):.2f} rad',
                fontsize=10
            )
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved sample visualization to: {save_path}")


def print_dataset_statistics(data: dict, dataset_name: str):
    """Print statistics about the generated dataset."""
    psfs = data['psf']
    coeffs = data['zernike_coeffs']
    metadata = data['metadata']
    
    print(f"\n{'='*60}")
    print(f"{dataset_name} Dataset Statistics")
    print(f"{'='*60}")
    print(f"Number of samples: {metadata['n_samples']}")
    print(f"Grid size: {metadata['grid_size']}x{metadata['grid_size']}")
    print(f"Zernike modes: {metadata['n_modes']}")
    print(f"\nPSF Statistics:")
    print(f"  Shape: {psfs.shape}")
    print(f"  Min: {psfs.min():.3e}")
    print(f"  Max: {psfs.max():.3e}")
    print(f"  Mean: {psfs.mean():.3e}")
    print(f"  Std: {psfs.std():.3e}")
    print(f"\nCoefficient Statistics:")
    print(f"  Shape: {coeffs.shape}")
    print(f"  Range: [{coeffs.min():.3f}, {coeffs.max():.3f}] radians")
    print(f"  Mean: {coeffs.mean():.3f} rad")
    print(f"  Std: {coeffs.std():.3f} rad")
    print(f"  RMS per mode: {np.sqrt(np.mean(coeffs**2, axis=0)).mean():.3f} rad")
    print(f"{'='*60}\n")


def main():
    """Generate synthetic dataset for JWST phase retrieval."""
    parser = argparse.ArgumentParser(description="Generate JWST PSF dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Generate only training dataset",
    )
    parser.add_argument(
        "--val-only",
        action="store_true",
        help="Generate only validation dataset",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Generate only test dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for generated datasets",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=True,
        help="Create sample visualizations",
    )
    args = parser.parse_args()

    print("="*70)
    print(" JWST PSF Dataset Generation")
    print("="*70)
    
    # Load configuration
    print(f"\n📋 Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which datasets to generate
    generate_train = not (args.val_only or args.test_only)
    generate_val = not (args.train_only or args.test_only)
    generate_test = not (args.train_only or args.val_only)
    
    # Generate training dataset
    if generate_train:
        print(f"\n🔄 Generating TRAINING dataset...")
        train_data = generate_dataset(
            n_samples=config['data']['train_samples'],
            config=config,
            show_progress=True,
            random_seed=config.get('random_seed', 42)
        )
        
        train_path = output_dir / "train.npz"
        save_dataset(train_data, str(train_path))
        print_dataset_statistics(train_data, "TRAINING")
        
        if args.visualize:
            viz_path = "outputs/figures/train_samples.png"
            Path(viz_path).parent.mkdir(parents=True, exist_ok=True)
            visualize_samples(train_data, viz_path, n_samples=6)
    
    # Generate validation dataset
    if generate_val:
        print(f"\n🔄 Generating VALIDATION dataset...")
        val_data = generate_dataset(
            n_samples=config['data']['val_samples'],
            config=config,
            show_progress=True,
            random_seed=config.get('random_seed', 42) + 1000  # Different seed
        )
        
        val_path = output_dir / "val.npz"
        save_dataset(val_data, str(val_path))
        print_dataset_statistics(val_data, "VALIDATION")
        
        if args.visualize:
            viz_path = "outputs/figures/val_samples.png"
            visualize_samples(val_data, viz_path, n_samples=6)
    
    # Generate test dataset
    if generate_test:
        print(f"\n🔄 Generating TEST dataset...")
        test_data = generate_dataset(
            n_samples=config['data']['test_samples'],
            config=config,
            show_progress=True,
            random_seed=config.get('random_seed', 42) + 2000  # Different seed
        )
        
        test_path = output_dir / "test.npz"
        save_dataset(test_data, str(test_path))
        print_dataset_statistics(test_data, "TEST")
        
        if args.visualize:
            viz_path = "outputs/figures/test_samples.png"
            visualize_samples(test_data, viz_path, n_samples=6)
    
    # Summary
    print("\n" + "="*70)
    print(" ✅ DATA GENERATION COMPLETE")
    print("="*70)
    print(f"\n📁 Output directory: {output_dir.absolute()}")
    if generate_train:
        print(f"   ✅ train.npz ({config['data']['train_samples']} samples)")
    if generate_val:
        print(f"   ✅ val.npz ({config['data']['val_samples']} samples)")
    if generate_test:
        print(f"   ✅ test.npz ({config['data']['test_samples']} samples)")
    
    if args.visualize:
        print(f"\n📊 Visualizations saved to: outputs/figures/")
    
    print("\n🎯 Next steps:")
    print("   1. Implement PyTorch Dataset class (src/neural_wavefront/data/dataset.py)")
    print("   2. Implement model architecture (src/neural_wavefront/models/resnet.py)")
    print("   3. Run training: uv run python scripts/train.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
