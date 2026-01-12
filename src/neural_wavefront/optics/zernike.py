"""Zernike polynomials (Noll indexing).

This module provides utilities for generating Zernike polynomials on a
square grid and evaluating single modes via the Noll indexing (1-based).

References
- Noll, R. J. (1976). Zernike polynomials and atmospheric turbulence.
- Goodman, Introduction to Fourier Optics (for conventions)

Functions
- noll_to_nm(j): Convert 1-based Noll index j -> (n, m)
- nm_to_noll(n, m): Convert (n, m) -> Noll index j
- radial_polynomial(n, m, rho): Evaluate radial polynomial R_n^m(rho)
- zernike_nm(n, m, rho, theta, normalize=True): Evaluate a single Zernike (n,m)
- zernike_noll(j, rho, theta, normalize=True): Evaluate Zernike with Noll index j
- generate_zernike_basis(n_modes=None, max_order=None, grid_size=65): Generate a stack
  of Noll-indexed Zernike modes on a unit disk grid (rho <= 1)

Notes
- Noll indexing is 1-based: j=1 is piston (n=0, m=0)
- By default the polynomials are returned with Noll's normalization
  (sqrt(n+1) for m=0, sqrt(2*(n+1)) for m!=0)
"""

from __future__ import annotations

from typing import Optional, Tuple

import math

import numpy as np


__all__ = [
    "noll_to_nm",
    "nm_to_noll",
    "radial_polynomial",
    "zernike_nm",
    "zernike_noll",
    "generate_zernike_basis",
    "n_modes_from_order",
    "zernike_grid",
    "combine_zernike_modes",
]


def n_modes_from_order(max_order: int) -> int:
    """Return the number of Zernike modes up to and including `max_order`.

    Args:
        max_order: maximum radial order n (>= 0)

    Returns:
        Number of modes (int)
    """
    if max_order < 0:
        raise ValueError("max_order must be >= 0")
    return (max_order + 1) * (max_order + 2) // 2


def noll_to_nm(j: int) -> Tuple[int, int]:
    """Convert a 1-based Noll index j to radial degree n and azimuthal order m.

    The mapping follows the standard Noll ordering (1-based). Examples:
    j=1 -> (0, 0) (piston)
    j=2 -> (1, -1)
    j=3 -> (1, 1)
    j=4 -> (2, -2)
    j=5 -> (2, 0)
    j=6 -> (2, 2)

    Args:
        j: 1-based Noll index (int)

    Returns:
        (n, m) tuple of integers
    """
    j0 = int(j)
    if j0 < 1:
        raise ValueError("Noll index j must be >= 1")

    # Find radial degree n such that j <= (n+1)(n+2)/2
    n = 0
    while j0 > (n + 1) * (n + 2) // 2:
        n += 1

    # number of modes up to previous order (n-1)
    t_prev = n * (n + 1) // 2
    k = j0 - t_prev - 1  # 0-based index within current n

    m = -n + 2 * k
    return n, m


def nm_to_noll(n: int, m: int) -> int:
    """Convert (n, m) to 1-based Noll index j.

    Args:
        n: radial order
        m: azimuthal frequency (can be negative)

    Returns:
        j (1-based)
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if abs(m) > n:
        raise ValueError("|m| must be <= n")
    if (n - abs(m)) % 2 != 0:
        raise ValueError("n - |m| must be even")

    k = (m + n) // 2
    t_prev = n * (n + 1) // 2
    return t_prev + k + 1


def radial_polynomial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """Evaluate the Zernike radial polynomial R_n^m(rho).

    Args:
        n: radial degree (n >= 0)
        m: azimuthal order (m >= 0)
        rho: radial coordinate (array-like, 0 <= rho <= 1)

    Returns:
        R_n^m evaluated at rho (same shape as rho)
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if m < 0:
        raise ValueError("m must be >= 0 (use absolute value of azimuthal order)")
    if (n - m) % 2 != 0:
        # Polynomial is zero for these indices
        return np.zeros_like(rho, dtype=float)

    rho = np.asarray(rho, dtype=float)
    R = np.zeros_like(rho, dtype=float)
    k_max = (n - m) // 2
    for k in range(k_max + 1):
        num = (-1) ** k * math.factorial(n - k)
        denom = (
            math.factorial(k)
            * math.factorial((n + m) // 2 - k)
            * math.factorial((n - m) // 2 - k)
        )
        coeff = num / denom
        R = R + coeff * rho ** (n - 2 * k)
    return R


def zernike_nm(
    n: int, m: int, rho: np.ndarray, theta: np.ndarray, normalize: bool = True
) -> np.ndarray:
    """Evaluate the Zernike polynomial with radial degree n and azimuthal order m.

    Args:
        n: radial degree (>= 0)
        m: azimuthal order (can be negative)
        rho: radial coordinate array (same shape as theta)
        theta: angular coordinate array (same shape as rho)
        normalize: if True apply Noll normalization

    Returns:
        Array of the Zernike polynomial values (same shape as rho/theta)
    """
    if rho.shape != theta.shape:
        raise ValueError("rho and theta must have the same shape")

    m_abs = abs(int(m))
    R = radial_polynomial(int(n), m_abs, rho)

    if m == 0:
        Z = R
    elif m > 0:
        Z = R * np.cos(m_abs * theta)
    else:
        Z = R * np.sin(m_abs * theta)

    if normalize:
        if m == 0:
            norm = math.sqrt(n + 1)
        else:
            norm = math.sqrt(2 * (n + 1))
        Z = Z * norm

    # Zero outside the unit disk
    Z = np.where(rho <= 1.0, Z, 0.0)
    return Z


def zernike_noll(j: int, rho: np.ndarray, theta: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Evaluate the Zernike polynomial for Noll index j (1-based).

    Args:
        j: Noll index (1-based)
        rho: radial coordinate array
        theta: angular coordinate array
        normalize: apply Noll normalization

    Returns:
        Array with same shape as rho/theta
    """
    n, m = noll_to_nm(j)
    return zernike_nm(n, m, rho, theta, normalize=normalize)


def zernike_grid(grid_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create a (rho, theta) grid on [-1, 1] x [-1, 1].

    The unit circle radius is 1.

    Args:
        grid_size: number of pixels along x and y

    Returns:
        (rho, theta) arrays each of shape (grid_size, grid_size)
    """
    if grid_size < 1:
        raise ValueError("grid_size must be >= 1")
    coords = np.linspace(-1.0, 1.0, grid_size)
    x, y = np.meshgrid(coords, coords)
    rho = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return rho, theta


def generate_zernike_basis(
    n_modes: Optional[int] = None,
    grid_size: int = 65,
    max_order: Optional[int] = None,
    normalize: bool = True,
) -> np.ndarray:
    """Generate a stack of Noll-indexed Zernike modes on a unit-disk grid.

    Args:
        n_modes: Number of Noll modes to generate (1-based count). If None and
            max_order is provided it will be computed as (max_order+1)(max_order+2)/2.
        grid_size: Size of the square grid (grid_size x grid_size)
        max_order: maximum radial degree n; if provided overrides n_modes when
            n_modes is None.
        normalize: apply Noll normalization

    Returns:
        basis: numpy array of shape (n_modes, grid_size, grid_size)

    Notes:
        The returned basis is ordered by increasing Noll index j (1..n_modes).
    """
    if n_modes is None:
        if max_order is None:
            raise ValueError("Either n_modes or max_order must be provided")
        n_modes = n_modes_from_order(max_order)

    if n_modes < 1:
        raise ValueError("n_modes must be >= 1")

    rho, theta = zernike_grid(grid_size)
    basis = np.zeros((n_modes, grid_size, grid_size), dtype=float)

    for j in range(1, n_modes + 1):
        basis[j - 1] = zernike_noll(j, rho, theta, normalize=normalize)

    return basis


def combine_zernike_modes(basis: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """
    Combine Zernike basis functions with coefficients to create a phase map.
    
    This function performs a weighted sum of Zernike polynomials:
        phase(x, y) = Σ_j a_j * Z_j(x, y)
    
    Args:
        basis: Zernike basis functions, shape (n_modes, grid_size, grid_size)
        coefficients: Zernike coefficients in radians, shape (n_modes,)
    
    Returns:
        Phase map in radians, shape (grid_size, grid_size)
    
    Example:
        >>> basis = generate_zernike_basis(n_modes=15, grid_size=256)
        >>> coeffs = np.array([0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # Z_4 defocus
        >>> phase = combine_zernike_modes(basis, coeffs)
        >>> phase.shape
        (256, 256)
    """
    if basis.shape[0] != len(coefficients):
        raise ValueError(
            f"Number of basis functions ({basis.shape[0]}) must match "
            f"number of coefficients ({len(coefficients)})"
        )
    
    # Weighted sum: phase = Σ a_j * Z_j
    phase = np.sum(coefficients[:, np.newaxis, np.newaxis] * basis, axis=0)
    
    return phase


if __name__ == "__main__":
    # Quick smoke test
    import sys

    print("Zernike module smoke test: generating first 15 modes on 65x65 grid")
    b = generate_zernike_basis(n_modes=15, grid_size=65)
    print("basis shape:", b.shape)

    # Show (n, m) for first several indices
    for j in range(1, 16):
        print(j, "->", noll_to_nm(j))

    # Check that piston (j=1) is constant within the disk
    rho, theta = zernike_grid(7)
    p = zernike_noll(1, rho, theta)
    print("piston unique values (inside disk):", np.unique(p[rho <= 1]))
    
    # Test combine function
    print("\nTesting combine_zernike_modes...")
    basis_test = generate_zernike_basis(n_modes=10, grid_size=128)
    coeffs_test = np.random.randn(10) * 0.5
    phase_test = combine_zernike_modes(basis_test, coeffs_test)
    print(f"Combined phase shape: {phase_test.shape}")
    print(f"Phase range: [{phase_test.min():.3f}, {phase_test.max():.3f}] radians")

    sys.exit(0)
