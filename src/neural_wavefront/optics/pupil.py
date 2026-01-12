"""
JWST aperture mask generation.

This module creates the geometric aperture function A(x,y) for the
James Webb Space Telescope's 18-segment primary mirror.

The JWST primary mirror consists of 18 hexagonal segments arranged in a
specific pattern. This module generates a binary mask representing this
geometry on a square grid.
"""

import numpy as np
from typing import Tuple, Optional


def hexagon_vertices(center: Tuple[float, float], radius: float) -> np.ndarray:
    """
    Generate vertices of a regular hexagon.
    
    Parameters
    ----------
    center : tuple of float
        (x, y) coordinates of hexagon center
    radius : float
        Distance from center to vertex (circumradius)
        
    Returns
    -------
    np.ndarray
        Array of shape (6, 2) containing (x, y) coordinates of vertices
    """
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, 7)[:-1] + np.pi / 2  # Start from top
    vertices = np.column_stack([
        cx + radius * np.cos(angles),
        cy + radius * np.sin(angles)
    ])
    return vertices


def point_in_hexagon(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """
    Check if points are inside a hexagon using the winding number algorithm.
    
    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 2) containing (x, y) coordinates
    vertices : np.ndarray
        Array of shape (6, 2) containing hexagon vertex coordinates
        
    Returns
    -------
    np.ndarray
        Boolean array of shape (N,) indicating if each point is inside
    """
    def cross_product_z(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Z-component of cross product."""
        return v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    
    # Check if points are on the correct side of all edges
    inside = np.ones(len(points), dtype=bool)
    
    for i in range(6):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % 6]
        
        # Edge vector
        edge = v2 - v1
        
        # Vector from v1 to points
        to_point = points - v1
        
        # Cross product (negative means point is on right side of edge)
        cross = cross_product_z(
            np.tile(edge, (len(points), 1)),
            to_point
        )
        
        inside &= (cross >= 0)
    
    return inside


def create_hexagon_mask(
    grid_size: int,
    center: Tuple[float, float],
    radius: float
) -> np.ndarray:
    """
    Create a binary mask for a single hexagon on a grid.
    
    Parameters
    ----------
    grid_size : int
        Size of the square grid
    center : tuple of float
        (x, y) coordinates of hexagon center (in grid coordinates, -1 to 1)
    radius : float
        Hexagon circumradius (in grid coordinates, -1 to 1)
        
    Returns
    -------
    np.ndarray
        Binary mask of shape (grid_size, grid_size)
    """
    # Create coordinate grid
    coords = np.linspace(-1, 1, grid_size)
    x, y = np.meshgrid(coords, coords)
    
    # Flatten for processing
    points = np.column_stack([x.ravel(), y.ravel()])
    
    # Get hexagon vertices
    vertices = hexagon_vertices(center, radius)
    
    # Check which points are inside
    inside = point_in_hexagon(points, vertices)
    
    # Reshape to grid
    mask = inside.reshape((grid_size, grid_size))
    
    return mask.astype(float)


def jwst_segment_positions(aperture_diameter: float = 6.5) -> list[Tuple[float, float]]:
    """
    Get the positions of JWST's 18 hexagonal mirror segments.
    
    The JWST primary mirror has 18 hexagonal segments arranged in 3 rings:
    - Ring 0 (center): 1 segment at origin
    - Ring 1 (inner): 6 segments in a hexagonal pattern
    - Ring 2 (outer): 6 segments in the gaps, 6 in corners (12 total, but we use simplified layout)
    
    Parameters
    ----------
    aperture_diameter : float
        Full diameter of the telescope primary mirror in meters (default: 6.5m for JWST)
        
    Returns
    -------
    list of tuples
        List of (x, y) positions for each segment center (normalized to [-1, 1])
        
    Notes
    -----
    This is a simplified but representative geometry. The actual JWST has a more
    complex arrangement with specific gaps and orientations.
    """
    positions = []
    
    # Segment flat-to-flat distance (approximate, for 18 segments in ~6.5m diameter)
    # Each segment is roughly 1.32m flat-to-flat
    segment_flat = 1.32  # meters
    segment_radius = segment_flat / np.sqrt(3)  # circumradius
    
    # Normalize to aperture diameter
    # Scale factor to fit in [-1, 1] coordinate system
    scale = 2.0 / aperture_diameter
    
    # Ring 0: Center segment
    positions.append((0.0, 0.0))
    
    # Ring 1: 6 segments in hexagonal pattern around center
    # Distance from center to ring 1 segment centers
    ring1_radius = segment_flat * 1.0  # Adjacent hexagons
    
    for i in range(6):
        angle = i * np.pi / 3  # 60-degree increments
        x = ring1_radius * np.cos(angle) * scale
        y = ring1_radius * np.sin(angle) * scale
        positions.append((x, y))
    
    # Ring 2: 12 segments in two sub-rings
    # 6 segments in cardinal/ordinal directions
    ring2a_radius = segment_flat * 2.0
    for i in range(6):
        angle = i * np.pi / 3  # 60-degree increments
        x = ring2a_radius * np.cos(angle) * scale
        y = ring2a_radius * np.sin(angle) * scale
        positions.append((x, y))
    
    # 6 segments in between (offset by 30 degrees)
    ring2b_radius = segment_flat * 1.732  # sqrt(3) spacing
    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6  # Offset by 30 degrees
        x = ring2b_radius * np.cos(angle) * scale
        y = ring2b_radius * np.sin(angle) * scale
        positions.append((x - 0.05, y))  # Slight adjustment for better fit
    
    # Only return first 18 (we might have defined 19)
    return positions[:18]


def create_jwst_aperture(
    grid_size: int = 256,
    num_segments: int = 18,
    aperture_diameter: float = 6.5,
    segment_gap: float = 0.01,
) -> np.ndarray:
    """
    Create the JWST primary mirror aperture mask.
    
    Generates a binary mask representing the 18-hexagon geometry of the
    James Webb Space Telescope's primary mirror.
    
    Parameters
    ----------
    grid_size : int
        Size of the square grid (typically 256)
    num_segments : int
        Number of hexagonal segments (must be 18 for JWST)
    aperture_diameter : float
        Diameter of the full aperture in meters (default: 6.5m)
    segment_gap : float
        Gap between segments as fraction of aperture (default: 0.01)
        
    Returns
    -------
    np.ndarray
        Binary aperture mask of shape (grid_size, grid_size)
        1.0 inside mirror segments, 0.0 outside
        
    Examples
    --------
    >>> aperture = create_jwst_aperture(grid_size=256)
    >>> aperture.shape
    (256, 256)
    >>> # Check fill factor (should be ~80-85% for JWST)
    >>> fill_factor = np.sum(aperture) / aperture.size
    >>> print(f"Fill factor: {fill_factor:.2%}")
    """
    if num_segments != 18:
        raise ValueError(f"JWST has 18 segments, got {num_segments}")
    
    # Get segment positions
    positions = jwst_segment_positions(aperture_diameter)
    
    # Segment size (circumradius in normalized coordinates)
    segment_flat = 1.32 / aperture_diameter  # Normalized
    segment_radius = (segment_flat / np.sqrt(3)) * 2.0 * (1.0 - segment_gap)
    
    # Initialize aperture
    aperture = np.zeros((grid_size, grid_size), dtype=float)
    
    # Add each hexagonal segment
    for center in positions:
        hexagon = create_hexagon_mask(grid_size, center, segment_radius)
        aperture = np.maximum(aperture, hexagon)  # Union of all segments
    
    return aperture


def create_circular_aperture(
    grid_size: int = 256,
    diameter: float = 1.0
) -> np.ndarray:
    """
    Create a simple circular aperture mask.
    
    Useful for comparison with JWST and for testing.
    
    Parameters
    ----------
    grid_size : int
        Size of the square grid
    diameter : float
        Diameter of the aperture (in grid coordinates, typically 1.0 for unit circle)
        
    Returns
    -------
    np.ndarray
        Binary mask with 1.0 inside circle, 0.0 outside
    """
    coords = np.linspace(-1, 1, grid_size)
    x, y = np.meshgrid(coords, coords)
    rho = np.sqrt(x**2 + y**2)
    
    radius = diameter / 2.0
    aperture = (rho <= radius).astype(float)
    
    return aperture


if __name__ == "__main__":
    """Demonstration and validation of JWST aperture."""
    import matplotlib.pyplot as plt
    
    print("=== JWST Aperture Generator ===\n")
    
    # Generate JWST aperture
    print("Generating JWST aperture mask...")
    jwst = create_jwst_aperture(grid_size=512, num_segments=18)
    
    # Compute statistics
    fill_factor = np.sum(jwst) / jwst.size
    print(f"Grid size: {jwst.shape}")
    print(f"Fill factor: {fill_factor:.2%}")
    print(f"Number of illuminated pixels: {np.sum(jwst):.0f}")
    
    # Create comparison with circular aperture
    circular = create_circular_aperture(grid_size=512, diameter=1.0)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # JWST aperture
    axes[0].imshow(jwst, cmap='gray', origin='lower')
    axes[0].set_title('JWST Primary Mirror\n(18 Hexagonal Segments)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    axes[0].text(0.5, -0.05, f'Fill Factor: {fill_factor:.1%}', 
                 transform=axes[0].transAxes, ha='center', fontsize=11)
    
    # Circular aperture
    circ_fill = np.sum(circular) / circular.size
    axes[1].imshow(circular, cmap='gray', origin='lower')
    axes[1].set_title('Circular Aperture\n(Reference)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    axes[1].text(0.5, -0.05, f'Fill Factor: {circ_fill:.1%}',
                 transform=axes[1].transAxes, ha='center', fontsize=11)
    
    # Difference/overlay
    overlay = np.zeros((*jwst.shape, 3))
    overlay[..., 0] = jwst  # JWST in red channel
    overlay[..., 2] = circular  # Circle in blue channel
    overlay[..., 1] = jwst * circular  # Overlap in green
    
    axes[2].imshow(overlay, origin='lower')
    axes[2].set_title('Overlay\n(JWST=Red, Circle=Blue, Both=Purple)', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/figures/jwst_aperture.png', dpi=150, bbox_inches='tight')
    print("\nSaved aperture visualization to outputs/figures/jwst_aperture.png")
    
    # Show cross-section
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    
    mid_row = jwst.shape[0] // 2
    x_coords = np.linspace(-1, 1, jwst.shape[1])
    
    ax.plot(x_coords, jwst[mid_row, :], 'r-', linewidth=2, label='JWST')
    ax.plot(x_coords, circular[mid_row, :], 'b--', linewidth=2, label='Circular')
    ax.set_xlabel('Position (normalized)', fontsize=12)
    ax.set_ylabel('Transmission', fontsize=12)
    ax.set_title('Aperture Cross-Section (Horizontal Mid-Line)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim([-0.1, 1.1])
    
    plt.tight_layout()
    plt.savefig('outputs/figures/aperture_cross_section.png', dpi=150, bbox_inches='tight')
    print("Saved cross-section to outputs/figures/aperture_cross_section.png")
    
    print("\n✅ JWST aperture generation complete!")
