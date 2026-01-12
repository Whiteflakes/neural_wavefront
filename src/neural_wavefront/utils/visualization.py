"""Utility functions for visualization."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def plot_psf(
    psf: np.ndarray,
    title: str = "Point Spread Function",
    log_scale: bool = True,
    colormap: str = "inferno",
    save_path: Optional[Path] = None,
    dpi: int = 150,
) -> Figure:
    """
    Plot a Point Spread Function.

    Args:
        psf: 2D array containing PSF intensity
        title: Plot title
        log_scale: Whether to display in log scale
        colormap: Matplotlib colormap name
        save_path: If provided, save figure to this path
        dpi: Resolution for saved figure

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    # Prepare data for display
    if log_scale:
        # Avoid log(0) by adding small offset
        data = np.log10(psf + 1e-10)
        label = "log₁₀(Intensity)"
    else:
        data = psf
        label = "Intensity"

    # Plot
    im = ax.imshow(data, cmap=colormap, origin="lower")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("x (pixels)", fontsize=11)
    ax.set_ylabel("y (pixels)", fontsize=11)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label, fontsize=11)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_zernike_comparison(
    true_coeffs: np.ndarray,
    pred_coeffs: np.ndarray,
    mode_names: Optional[list[str]] = None,
    save_path: Optional[Path] = None,
    dpi: int = 150,
) -> Figure:
    """
    Plot comparison of true vs predicted Zernike coefficients.

    Args:
        true_coeffs: Ground truth coefficients (1D array)
        pred_coeffs: Predicted coefficients (1D array)
        mode_names: Optional names for Zernike modes
        save_path: If provided, save figure to this path
        dpi: Resolution for saved figure

    Returns:
        Matplotlib figure object
    """
    n_modes = len(true_coeffs)
    if mode_names is None:
        mode_names = [f"Z{i+1}" for i in range(n_modes)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot comparison
    x = np.arange(n_modes)
    width = 0.35

    ax1.bar(x - width / 2, true_coeffs, width, label="Ground Truth", alpha=0.8)
    ax1.bar(x + width / 2, pred_coeffs, width, label="Predicted", alpha=0.8)
    ax1.set_xlabel("Zernike Mode", fontsize=11)
    ax1.set_ylabel("Coefficient Value", fontsize=11)
    ax1.set_title("Zernike Coefficients Comparison", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(mode_names, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Scatter plot (perfect prediction = diagonal line)
    ax2.scatter(true_coeffs, pred_coeffs, alpha=0.6, s=80)
    ax2.plot(
        [true_coeffs.min(), true_coeffs.max()],
        [true_coeffs.min(), true_coeffs.max()],
        "r--",
        lw=2,
        label="Perfect Prediction",
    )
    ax2.set_xlabel("Ground Truth", fontsize=11)
    ax2.set_ylabel("Predicted", fontsize=11)
    ax2.set_title("Prediction Accuracy", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_aspect("equal", adjustable="box")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    save_path: Optional[Path] = None,
    dpi: int = 150,
) -> Figure:
    """
    Plot training and validation loss curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: If provided, save figure to this path
        dpi: Resolution for saved figure

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = np.arange(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, label="Training Loss", linewidth=2, marker="o", markersize=4)
    ax.plot(
        epochs, val_losses, label="Validation Loss", linewidth=2, marker="s", markersize=4
    )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Progress", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig
