"""
Stage 2: Sprite Detection using Flood Fill + Connected Components
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from ..config import PipelineConfig
from ..logger import PipelineLogger


def create_hsv_mask(
    image: np.ndarray,
    bg_hsv: Tuple[float, float, float],
    tolerances: Tuple[float, float, float],
) -> np.ndarray:
    """
    Create binary mask for background color

    Args:
        image: RGB image
        bg_hsv: Background color (H, S, V) in degrees/percentages
        tolerances: (hue_tol, sat_tol, val_tol) in degrees/percentages

    Returns:
        Binary mask (255 = background, 0 = foreground)
    """
    # Convert to HSV (OpenCV uses H: 0-180, S: 0-255, V: 0-255)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    h_bg, s_bg, v_bg = bg_hsv
    h_tol, s_tol, v_tol = tolerances

    # Convert from degrees/percentages to OpenCV range
    # H: 0-360 → 0-180
    h_cv = h_bg / 2.0
    h_tol_cv = h_tol / 2.0

    # S, V: 0-100 → 0-255
    s_cv = s_bg * 2.55
    s_tol_cv = s_tol * 2.55
    v_cv = v_bg * 2.55
    v_tol_cv = v_tol * 2.55

    # Create range
    lower = np.array(
        [max(0, h_cv - h_tol_cv), max(0, s_cv - s_tol_cv), max(0, v_cv - v_tol_cv)],
        dtype=np.uint8,
    )

    upper = np.array(
        [
            min(180, h_cv + h_tol_cv),
            min(255, s_cv + s_tol_cv),
            min(255, v_cv + v_tol_cv),
        ],
        dtype=np.uint8,
    )

    # Create mask
    mask = cv2.inRange(hsv, lower, upper)

    return mask


def flood_fill_from_edges(mask: np.ndarray) -> np.ndarray:
    """
    Flood fill background from all image edges

    Args:
        mask: Binary mask (255 = potential background)

    Returns:
        Filled mask (255 = confirmed background from edges)
    """
    h, w = mask.shape
    filled = mask.copy()

    # Create flood mask (must be h+2, w+2 per OpenCV requirements)
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    # Fill from all four corners
    for y, x in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
        if filled[y, x] == 255:
            cv2.floodFill(filled, flood_mask, (x, y), 128)

    # Fill from all edge pixels
    # Top and bottom edges
    for x in range(w):
        if filled[0, x] == 255:
            cv2.floodFill(filled, flood_mask, (x, 0), 128)
        if filled[h - 1, x] == 255:
            cv2.floodFill(filled, flood_mask, (x, h - 1), 128)

    # Left and right edges
    for y in range(h):
        if filled[y, 0] == 255:
            cv2.floodFill(filled, flood_mask, (0, y), 128)
        if filled[y, w - 1] == 255:
            cv2.floodFill(filled, flood_mask, (w - 1, y), 128)

    # Convert back: 128 = background (filled), 0/255 = foreground
    filled = np.where(filled == 128, 255, 0).astype(np.uint8)

    return filled


def validate_sprite_component(
    component_stats: np.ndarray,
    component_idx: int,
    image_shape: Tuple[int, int],
    config: PipelineConfig,
) -> Tuple[bool, str]:
    """
    Validate if a connected component is likely a sprite

    Checks:
    1. Size: Must be at least X% of image
    2. Position: Should be relatively centered

    Returns:
        (is_valid, reason)
    """
    x, y, w, h, area = component_stats[component_idx]
    img_h, img_w = image_shape[:2]

    # Check 1: Minimum size
    min_area = (img_h * img_w) * config.min_sprite_size_ratio
    if area < min_area:
        return (
            False,
            f"Too small: {area} < {min_area:.0f} ({config.min_sprite_size_ratio:.1%})",
        )

    # Check 2: Reasonable position (not stuck in corner)
    center_x = x + w / 2
    center_y = y + h / 2
    img_center_x = img_w / 2
    img_center_y = img_h / 2

    # Distance from center as ratio of image dimension
    dx_ratio = abs(center_x - img_center_x) / img_w
    dy_ratio = abs(center_y - img_center_y) / img_h

    if (
        dx_ratio > config.sprite_center_tolerance
        or dy_ratio > config.sprite_center_tolerance
    ):
        return False, f"Too far from center: dx={dx_ratio:.1%}, dy={dy_ratio:.1%}"

    return True, "Valid sprite candidate"


def detect_sprite(
    image: np.ndarray,
    bg_hsv: Tuple[float, float, float],
    config: PipelineConfig,
    logger: PipelineLogger,
) -> Optional[dict]:
    """
    Detect sprite using flood fill from edges + connected components

    Algorithm:
    1. Create HSV mask for background color
    2. Flood fill from all edges to identify connected background
    3. Find connected components in non-background regions
    4. Select largest component as sprite candidate
    5. Validate sprite (size, position)

    Args:
        image: RGB image
        bg_hsv: Background color HSV
        config: Pipeline configuration
        logger: Logger instance

    Returns:
        Sprite info dict or None if no valid sprite found
        {
            'bbox': (x, y, w, h),
            'area': int,
            'centroid': (cx, cy),
            'mask': binary mask for this component
        }
    """
    logger.log_info("Stage 2: Detecting sprite...")

    # Create background mask
    # Use moderate tolerances for initial mask
    initial_tolerances = (10.0, 15.0, 15.0)  # (H, S, V)
    bg_mask = create_hsv_mask(image, bg_hsv, initial_tolerances)

    # Flood fill from edges
    filled_bg = flood_fill_from_edges(bg_mask)

    # Invert to get foreground
    fg_mask = cv2.bitwise_not(filled_bg)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        fg_mask, connectivity=config.flood_fill_connectivity
    )

    logger.log_info(f"  Found {num_labels - 1} foreground components")

    if num_labels <= 1:
        # No components found (label 0 is background)
        logger.log_s2(
            method="flood_fill + connected_components",
            num_components=0,
            sprite_found=False,
            reason="No foreground components detected",
        )
        return None

    # Find largest component (excluding background label 0)
    areas = stats[1:, cv2.CC_STAT_AREA]  # Skip label 0
    largest_idx = np.argmax(areas) + 1  # +1 because we skipped label 0

    # Validate sprite
    is_valid, reason = validate_sprite_component(
        stats, largest_idx, image.shape, config
    )

    if not config.skip_validation and not is_valid:
        logger.log_s2(
            method="flood_fill + connected_components",
            num_components=num_labels - 1,
            largest_component_area=int(stats[largest_idx, cv2.CC_STAT_AREA]),
            sprite_found=False,
            validation_failed=True,
            reason=reason,
        )
        logger.log_warning(f"  Sprite validation failed: {reason}")
        return None

    # Extract sprite info
    x, y, w, h, area = stats[largest_idx]
    cx, cy = centroids[largest_idx]
    sprite_mask = (labels == largest_idx).astype(np.uint8) * 255

    sprite_info = {
        "bbox": (int(x), int(y), int(w), int(h)),
        "area": int(area),
        "centroid": (float(cx), float(cy)),
        "mask": sprite_mask,
    }

    logger.log_s2(
        method="flood_fill + connected_components",
        num_components=num_labels - 1,
        sprite_found=True,
        bbox=sprite_info["bbox"],
        area=sprite_info["area"],
        centroid=sprite_info["centroid"],
        validation={"passed": is_valid, "reason": reason},
    )

    logger.log_info(f"  Sprite detected: {w}x{h} @ ({x}, {y})")

    return sprite_info
