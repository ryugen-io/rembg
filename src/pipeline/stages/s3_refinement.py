"""
Stage 3: Mask Refinement using GrabCut (Optional)
"""

from typing import Optional

import cv2
import numpy as np

from ..config import PipelineConfig
from ..logger import PipelineLogger


def calculate_mask_quality(mask: np.ndarray, image: np.ndarray) -> float:
    """
    Calculate quality score for a mask

    Simple heuristic: ratio of foreground to total pixels
    Better masks have reasonable foreground ratios (not too small, not too large)

    Returns:
        Quality score 0.0-1.0 (higher is better)
    """
    fg_pixels = np.sum(mask > 0)
    total_pixels = mask.size

    fg_ratio = fg_pixels / total_pixels

    # Ideal ratio is around 20-60% (sprite with reasonable background)
    # Score decreases as we move away from ideal range
    if 0.2 <= fg_ratio <= 0.6:
        quality = 1.0
    elif fg_ratio < 0.2:
        # Too little foreground
        quality = fg_ratio / 0.2
    else:
        # Too much foreground
        quality = (1.0 - fg_ratio) / 0.4

    return max(0.0, min(1.0, quality))


def refine_mask(
    image: np.ndarray,
    sprite_info: Optional[dict],
    config: PipelineConfig,
    logger: PipelineLogger,
) -> Optional[np.ndarray]:
    """
    Refine mask using GrabCut algorithm

    GrabCut uses Graph Cut to segment foreground/background more precisely
    than simple color-based masking.

    Args:
        image: RGB image
        sprite_info: Sprite detection results from Stage 2
        config: Pipeline configuration
        logger: Logger instance

    Returns:
        Refined binary mask (255 = foreground) or None if skipped
    """
    if not config.use_grabcut:
        logger.log_s3(method="skipped", reason="grabcut disabled")
        return None

    if sprite_info is None:
        logger.log_s3(method="skipped", reason="no sprite detected in stage 2")
        return None

    logger.log_info("Stage 3: Refining mask with GrabCut...")

    # Initialize mask for GrabCut
    # 0 = definite background
    # 1 = definite foreground
    # 2 = probable background
    # 3 = probable foreground
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Initialize background/foreground models
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    # Get bounding box
    x, y, w, h = sprite_info["bbox"]
    rect = (x, y, w, h)

    try:
        # Run GrabCut
        cv2.grabCut(
            image,
            mask,
            rect,
            bgd_model,
            fgd_model,
            config.grabcut_iterations,
            cv2.GC_INIT_WITH_RECT,
        )

        # Convert mask: Keep definite and probable foreground
        refined_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)

        # Calculate quality
        quality = calculate_mask_quality(refined_mask, image)

        logger.log_s3(
            method="grabcut",
            iterations=config.grabcut_iterations,
            rect=rect,
            quality_score=quality,
            fg_pixels=int(np.sum(refined_mask > 0)),
            success=True,
        )

        logger.log_info(f"  GrabCut complete (quality: {quality:.2f})")

        return refined_mask

    except Exception as e:
        logger.log_error(f"  GrabCut failed: {e}")
        logger.log_s3(method="grabcut", success=False, error=str(e))
        return None
