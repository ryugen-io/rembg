"""
Stage 4: Morphological Cleanup and Edge Refinement
"""

import colorsys

import cv2
import numpy as np
from scipy import ndimage

from ..config import PipelineConfig
from ..logger import PipelineLogger


def cleanup_mask(
    image: np.ndarray,
    mask: np.ndarray,
    bg_hsv: tuple[float, float, float],
    config: PipelineConfig,
    logger: PipelineLogger,
) -> np.ndarray:
    """
    Clean up mask using morphological operations + edge refinement

    Steps:
    1. Morphological opening to remove noise
    2. Multi-pass edge cleanup (remove background-colored edge pixels)

    Args:
        image: Original RGB image
        mask: Binary mask (255 = foreground, 0 = background)
        bg_hsv: Detected background color (H, S, V)
        config: Pipeline configuration
        logger: Logger instance

    Returns:
        Cleaned binary mask
    """
    logger.log_info("Stage 4: Cleaning up mask...")

    # Step 1: Morphological opening (erosion + dilation)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.morphology_kernel_size, config.morphology_kernel_size),
    )
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    morphology_pixels_removed = int(np.sum(mask) - np.sum(cleaned))
    logger.log_info(
        f"  Morphological opening removed {morphology_pixels_removed:,} noise pixels"
    )

    # Step 2: Edge cleanup
    if config.cleanup_passes > 0:
        cleaned = multi_pass_edge_cleanup(image, cleaned, bg_hsv, config, logger)

    logger.log_s4(
        method="opening + edge_cleanup",
        morphological_kernel=f"MORPH_ELLIPSE ({config.morphology_kernel_size}x{config.morphology_kernel_size})",
        morphology_pixels_removed=morphology_pixels_removed,
        cleanup_passes=config.cleanup_passes,
        mode="pixel_art" if config.pixel_art_mode else "normal",
    )

    return cleaned


def multi_pass_edge_cleanup(
    image: np.ndarray,
    mask: np.ndarray,
    bg_hsv: tuple[float, float, float],
    config: PipelineConfig,
    logger: PipelineLogger,
) -> np.ndarray:
    """
    Multi-pass edge cleanup to remove background-tinted edge pixels

    Algorithm:
    1. Detect edges (where mask changes from 0 to 255)
    2. Convert edge pixels to HSV
    3. Check if they match background color (within tolerance)
    4. Remove matching pixels
    5. Repeat for N passes

    Args:
        image: RGB image
        mask: Binary mask
        bg_hsv: Background color (H, S, V)
        config: Pipeline configuration
        logger: Logger instance

    Returns:
        Cleaned mask
    """
    h_bg, s_bg, v_bg = bg_hsv
    h_tol, s_tol = config.edge_tolerances

    # Create RGBA result
    result = np.dstack([image, mask])

    total_cleaned = 0

    for pass_num in range(config.cleanup_passes):
        # Find edges using gradient
        alpha = result[:, :, 3].astype(np.float32)
        dy, dx = np.gradient(alpha)
        gradient = np.sqrt(dx**2 + dy**2)
        edge_mask = gradient > 5

        # Find pixels adjacent to transparent regions
        transparent = result[:, :, 3] == 0
        kernel = np.ones((3, 3), dtype=np.uint8)
        adjacent = ndimage.binary_dilation(transparent, kernel) & (result[:, :, 3] > 0)

        # Combine both edge detection methods
        edge_pixels = edge_mask | adjacent

        if not np.any(edge_pixels):
            if pass_num == 0:
                logger.log_info(
                    f"  Pass {pass_num + 1}/{config.cleanup_passes}: No edges to clean"
                )
            break

        # Get edge pixel colors (RGB)
        edge_rgb = result[edge_pixels, :3].astype(np.float32) / 255.0

        # Convert to HSV
        edge_hsv = np.array([colorsys.rgb_to_hsv(*rgb) for rgb in edge_rgb])
        h_e = edge_hsv[:, 0] * 360.0
        s_e = edge_hsv[:, 1] * 100.0
        # v_e not used for edge cleanup (only H and S matter for background matching)

        # Check if edge pixels match background color
        h_min = max(0.0, h_bg - h_tol)
        h_max = min(360.0, h_bg + h_tol)
        s_min = max(0.0, s_bg - s_tol)

        # Handle hue wrap-around
        if h_min < h_max:
            h_match = (h_e >= h_min) & (h_e <= h_max)
        else:
            h_match = (h_e >= h_min) | (h_e <= h_max)

        bg_tint = h_match & (s_e >= s_min)

        # Remove tinted edge pixels
        edge_coords = np.where(edge_pixels)
        tinted_indices = np.where(bg_tint)[0]

        if len(tinted_indices) == 0:
            logger.log_info(
                f"  Pass {pass_num + 1}/{config.cleanup_passes}: No tinted pixels (cleaned {total_cleaned:,} total)"
            )
            break

        for idx in tinted_indices:
            y, x = edge_coords[0][idx], edge_coords[1][idx]
            result[y, x, 3] = 0

        total_cleaned += len(tinted_indices)
        logger.log_info(
            f"  Pass {pass_num + 1}/{config.cleanup_passes}: Cleaned {len(tinted_indices):,} pixels ({total_cleaned:,} total)"
        )

    if config.cleanup_passes > 1 and total_cleaned > 0:
        logger.log_info(
            f"  Multi-pass cleanup complete: {total_cleaned:,} edge pixels removed"
        )

    return result[:, :, 3]
