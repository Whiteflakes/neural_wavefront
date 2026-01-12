"""
Optical propagation and PSF computation.

This module implements the Fourier optics relationships for computing
Point Spread Functions (PSFs) from aperture and phase information.

The key equation: PSF = |FFT{A(x,y) * exp(i*phi(x,y))}|^2

Where:
- A(x,y) is the aperture mask (JWST geometry)
- phi(x,y) is the wavefront phase error
- FFT is the 2D Fourier transform
- PSF is the intensity pattern in the image plane
"""

import numpy as np
from typing import Optional, Tuple
import warnings


def compute_pupil_field(
    aperture: np.ndarray,
    phase: np.ndarray,
    wavelength: Optional[float] = None
) -> np.ndarray:
    """
    Compute the complex pupil field from aperture and phase.
    
    E_pupil(x, y) = A(x, y) * exp(i * phi(x, y))
    
    Parameters
    ----------
    aperture : np.ndarray
        Binary aperture mask (0 or 1), shape (N, N)
    phase : np.ndarray
        Wavefront phase in radians, shape (N, N)
        If wavelength is provided, phase is assumed to be in waves and
        will be converted to radians
    wavelength : float, optional
        Wavelength for converting phase from waves to radians.
        If None, phase is assumed to be already in radians
        
    Returns
    -------
    np.ndarray
        Complex electric field at pupil plane, shape (N, N)
        
    Examples
    --------
    >>> from neural_wavefront.optics import pupil
    >>> aperture = pupil.create_circular_aperture(128)
    >>> phase = np.zeros_like(aperture)  # Perfect wavefront
    >>> E_pupil = compute_pupil_field(aperture, phase)
    >>> np.allclose(np.abs(E_pupil), aperture)
    True
    """
    if aperture.shape != phase.shape:
        raise ValueError(
            f"Aperture and phase must have same shape. "
            f"Got {aperture.shape} and {phase.shape}"
        )
    
    # Convert phase from waves to radians if wavelength is provided
    if wavelength is not None:
        phase_radians = 2 * np.pi * phase
    else:
        phase_radians = phase
    
    # Compute complex field
    E_pupil = aperture * np.exp(1j * phase_radians)
    
    return E_pupil


def propagate_to_focal_plane(
    pupil_field: np.ndarray,
    focal_length: Optional[float] = None,
    pixel_scale: Optional[float] = None
) -> np.ndarray:
    """
    Propagate pupil field to focal (image) plane using Fraunhofer diffraction.
    
    E_image = FFT{E_pupil}
    
    The 2D Fourier transform performs the optical propagation from pupil to
    focal plane. Uses fftshift to center the PSF.
    
    Parameters
    ----------
    pupil_field : np.ndarray
        Complex field at pupil plane, shape (N, N)
    focal_length : float, optional
        Focal length of the optical system (not used in computation,
        but can be stored for physical scaling)
    pixel_scale : float, optional
        Pixel scale in image plane (arcsec/pixel) for documentation
        
    Returns
    -------
    np.ndarray
        Complex field at image plane, shape (N, N), centered
        
    Notes
    -----
    The FFT implements the Fraunhofer diffraction integral:
    U(u,v) ∝ ∫∫ E(x,y) exp(-i 2π (ux + vy)) dx dy
    
    We use fftshift before and after to handle centering properly.
    """
    # Shift zero-frequency to center before FFT
    pupil_shifted = np.fft.fftshift(pupil_field)
    
    # Compute 2D Fourier transform
    image_field = np.fft.fft2(pupil_shifted)
    
    # Shift zero-frequency back to center
    image_field = np.fft.fftshift(image_field)
    
    # Normalization to conserve energy
    image_field = image_field / np.sqrt(pupil_field.size)
    
    return image_field


def compute_psf(
    aperture: np.ndarray,
    phase: np.ndarray,
    wavelength: Optional[float] = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute the Point Spread Function from aperture and phase.
    
    This is the main function that combines all steps:
    1. Compute pupil field: E_pupil = A * exp(i*phi)
    2. Propagate to focal plane: E_image = FFT(E_pupil)
    3. Compute intensity: PSF = |E_image|^2
    
    Parameters
    ----------
    aperture : np.ndarray
        Aperture mask, shape (N, N), values in [0, 1]
    phase : np.ndarray
        Wavefront phase error, shape (N, N)
        If wavelength=None: in radians
        If wavelength is set: in waves
    wavelength : float, optional
        Wavelength in meters (e.g., 2e-6 for 2 microns)
        If provided, converts phase from waves to radians
    normalize : bool
        If True, normalize PSF so peak = 1.0
        
    Returns
    -------
    np.ndarray
        Point Spread Function (intensity), shape (N, N)
        
    Examples
    --------
    >>> from neural_wavefront.optics import pupil, zernike
    >>> # Perfect aperture (circular)
    >>> aperture = pupil.create_circular_aperture(256)
    >>> phase = np.zeros_like(aperture)
    >>> psf_perfect = compute_psf(aperture, phase)
    >>> # Aberrated aperture
    >>> basis = zernike.generate_zernike_basis(n_modes=15, grid_size=256)
    >>> coeffs = np.array([0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # Defocus
    >>> phase_aberr = np.sum(coeffs[:, None, None] * basis, axis=0)
    >>> psf_aberr = compute_psf(aperture, phase_aberr)
    """
    # Step 1: Compute pupil field
    E_pupil = compute_pupil_field(aperture, phase, wavelength)
    
    # Step 2: Propagate to focal plane
    E_image = propagate_to_focal_plane(E_pupil)
    
    # Step 3: Compute intensity
    psf = np.abs(E_image) ** 2
    
    # Normalize if requested
    if normalize:
        psf_max = np.max(psf)
        if psf_max > 0:
            psf = psf / psf_max
        else:
            warnings.warn("PSF maximum is zero, cannot normalize")
    
    return psf


def psf_metrics(psf: np.ndarray) -> dict:
    """
    Compute useful metrics for a PSF.
    
    Parameters
    ----------
    psf : np.ndarray
        Point Spread Function, shape (N, N)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'peak': Maximum value
        - 'total_energy': Sum of all pixel values
        - 'strehl_ratio': Peak / total_energy (approximation)
        - 'fwhm_x': Full width at half maximum in x direction (pixels)
        - 'fwhm_y': Full width at half maximum in y direction (pixels)
    """
    metrics = {}
    
    # Peak value
    metrics['peak'] = np.max(psf)
    
    # Total energy
    metrics['total_energy'] = np.sum(psf)
    
    # Strehl ratio (approximate: peak / total for normalized PSF)
    if metrics['total_energy'] > 0:
        metrics['strehl_ratio'] = metrics['peak'] / metrics['total_energy'] * psf.size
    else:
        metrics['strehl_ratio'] = 0.0
    
    # FWHM (simple estimation)
    center_y, center_x = np.array(psf.shape) // 2
    
    # X direction
    profile_x = psf[center_y, :]
    half_max = metrics['peak'] / 2
    above_half = profile_x >= half_max
    if np.any(above_half):
        indices = np.where(above_half)[0]
        metrics['fwhm_x'] = indices[-1] - indices[0]
    else:
        metrics['fwhm_x'] = 0
    
    # Y direction
    profile_y = psf[:, center_x]
    above_half = profile_y >= half_max
    if np.any(above_half):
        indices = np.where(above_half)[0]
        metrics['fwhm_y'] = indices[-1] - indices[0]
    else:
        metrics['fwhm_y'] = 0
    
    return metrics


def compare_psfs(
    psf1: np.ndarray,
    psf2: np.ndarray,
    labels: Tuple[str, str] = ("PSF 1", "PSF 2")
) -> dict:
    """
    Compare two PSFs and compute difference metrics.
    
    Parameters
    ----------
    psf1, psf2 : np.ndarray
        PSFs to compare, shape (N, N)
    labels : tuple of str
        Names for the two PSFs
        
    Returns
    -------
    dict
        Comparison metrics including RMS difference, correlation, etc.
    """
    if psf1.shape != psf2.shape:
        raise ValueError(f"PSFs must have same shape. Got {psf1.shape} and {psf2.shape}")
    
    comparison = {
        'labels': labels,
    }
    
    # Normalize both to same scale for comparison
    psf1_norm = psf1 / np.max(psf1) if np.max(psf1) > 0 else psf1
    psf2_norm = psf2 / np.max(psf2) if np.max(psf2) > 0 else psf2
    
    # RMS difference
    comparison['rms_diff'] = np.sqrt(np.mean((psf1_norm - psf2_norm) ** 2))
    
    # Correlation
    comparison['correlation'] = np.corrcoef(psf1_norm.ravel(), psf2_norm.ravel())[0, 1]
    
    # Peak ratio
    comparison['peak_ratio'] = np.max(psf1) / np.max(psf2) if np.max(psf2) > 0 else np.inf
    
    return comparison


if __name__ == "__main__":
    """Demonstration of PSF computation."""
    import matplotlib.pyplot as plt
    from neural_wavefront.optics import pupil, zernike
    
    print("=== PSF Propagation Demonstration ===\n")
    
    grid_size = 256
    
    # Create apertures
    print("Creating apertures...")
    jwst_aperture = pupil.create_jwst_aperture(grid_size=grid_size)
    circular_aperture = pupil.create_circular_aperture(grid_size=grid_size)
    
    # Generate Zernike basis
    print("Generating Zernike basis...")
    basis = zernike.generate_zernike_basis(n_modes=15, grid_size=grid_size)
    
    # Test 1: Perfect wavefront
    print("\nTest 1: Perfect wavefront (no aberrations)")
    phase_perfect = np.zeros((grid_size, grid_size))
    
    psf_jwst_perfect = compute_psf(jwst_aperture, phase_perfect)
    psf_circ_perfect = compute_psf(circular_aperture, phase_perfect)
    
    print(f"JWST PSF metrics: {psf_metrics(psf_jwst_perfect)}")
    print(f"Circular PSF metrics: {psf_metrics(psf_circ_perfect)}")
    
    # Test 2: Defocus aberration
    print("\nTest 2: Defocus aberration (Z_4 = 0.5 waves)")
    coeffs_defocus = np.zeros(15)
    coeffs_defocus[3] = 0.5 * 2 * np.pi  # Z_4 is index 3 (0-indexed), convert to radians
    
    # Combine Zernike modes
    phase_defocus = np.sum(coeffs_defocus[:, None, None] * basis, axis=0)
    
    psf_jwst_defocus = compute_psf(jwst_aperture, phase_defocus)
    psf_circ_defocus = compute_psf(circular_aperture, phase_defocus)
    
    # Test 3: Coma aberration
    print("\nTest 3: Coma aberration (Z_7 = 0.3 waves)")
    coeffs_coma = np.zeros(15)
    coeffs_coma[6] = 0.3 * 2 * np.pi  # Z_7 is index 6
    phase_coma = np.sum(coeffs_coma[:, None, None] * basis, axis=0)
    
    psf_jwst_coma = compute_psf(jwst_aperture, phase_coma)
    
    # Test 4: Mixed aberrations
    print("\nTest 4: Mixed aberrations")
    coeffs_mixed = np.array([0, 0, 0, 0.2, -0.15, 0.1, 0.15, 0, 0, 0, 0, 0, 0, 0, 0]) * 2 * np.pi
    phase_mixed = np.sum(coeffs_mixed[:, None, None] * basis, axis=0)
    
    psf_jwst_mixed = compute_psf(jwst_aperture, phase_mixed)
    
    # Visualize results
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # Row 1: Perfect
    im1 = axes[0, 0].imshow(phase_perfect, cmap='RdBu_r', origin='lower')
    axes[0, 0].set_title('Perfect Phase', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    
    im2 = axes[0, 1].imshow(np.log10(psf_circ_perfect + 1e-10), cmap='inferno', origin='lower')
    axes[0, 1].set_title('Circular PSF (Perfect)', fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, label='log10(Intensity)')
    
    im3 = axes[0, 2].imshow(np.log10(psf_jwst_perfect + 1e-10), cmap='inferno', origin='lower')
    axes[0, 2].set_title('JWST PSF (Perfect)', fontsize=11, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, label='log10(Intensity)')
    
    # Row 2: Defocus
    im4 = axes[1, 0].imshow(phase_defocus, cmap='RdBu_r', origin='lower')
    axes[1, 0].set_title('Defocus Phase (Z_4)', fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)
    
    im5 = axes[1, 1].imshow(np.log10(psf_circ_defocus + 1e-10), cmap='inferno', origin='lower')
    axes[1, 1].set_title('Circular PSF (Defocus)', fontsize=11, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046, label='log10(Intensity)')
    
    im6 = axes[1, 2].imshow(np.log10(psf_jwst_defocus + 1e-10), cmap='inferno', origin='lower')
    axes[1, 2].set_title('JWST PSF (Defocus)', fontsize=11, fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046, label='log10(Intensity)')
    
    # Row 3: Mixed
    im7 = axes[2, 0].imshow(phase_mixed, cmap='RdBu_r', origin='lower')
    axes[2, 0].set_title('Mixed Phase\n(Defocus + Astigmatism + Coma)', fontsize=10, fontweight='bold')
    axes[2, 0].axis('off')
    plt.colorbar(im7, ax=axes[2, 0], fraction=0.046)
    
    im8 = axes[2, 1].imshow(np.log10(psf_jwst_coma + 1e-10), cmap='inferno', origin='lower')
    axes[2, 1].set_title('JWST PSF (Coma)', fontsize=11, fontweight='bold')
    axes[2, 1].axis('off')
    plt.colorbar(im8, ax=axes[2, 1], fraction=0.046, label='log10(Intensity)')
    
    im9 = axes[2, 2].imshow(np.log10(psf_jwst_mixed + 1e-10), cmap='inferno', origin='lower')
    axes[2, 2].set_title('JWST PSF (Mixed)', fontsize=11, fontweight='bold')
    axes[2, 2].axis('off')
    plt.colorbar(im9, ax=axes[2, 2], fraction=0.046, label='log10(Intensity)')
    
    plt.tight_layout()
    plt.savefig('outputs/figures/psf_demonstration.png', dpi=150, bbox_inches='tight')
    print("\nSaved PSF demonstration to outputs/figures/psf_demonstration.png")
    
    print("\n✅ PSF propagation implementation complete!")
