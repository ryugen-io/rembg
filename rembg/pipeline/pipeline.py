"""
BackgroundRemovalPipeline: Main orchestration class
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from .config import PipelineConfig
from .logger import PipelineLogger
from .stages import cleanup_mask, detect_background_color, detect_sprite, refine_mask


class BackgroundRemovalPipeline:
    """
    Production-ready multi-stage background removal pipeline

    Stages:
    1. Background Color Detection (K-Means)
    2. Sprite Detection (Flood Fill + Connected Components)
    3. Mask Refinement (GrabCut - optional)
    4. Morphological Cleanup

    With comprehensive logging and error handling
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        logger: Optional[PipelineLogger] = None,
    ):
        self.config = config or PipelineConfig()
        self.logger = logger or PipelineLogger()

    def process(self, input_path: Path) -> Path:
        """
        Process a single image through the full pipeline

        Args:
            input_path: Path to input image

        Returns:
            Path to output image

        Raises:
            FileNotFoundError: If input doesn't exist
            RuntimeError: If processing fails
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found: {input_path}")

        self.logger.start_image(input_path)
        self.logger.log_info(f"Processing: {input_path}")

        try:
            # Load image
            with Image.open(input_path) as img:
                img = img.convert("RGB")
                image = np.array(img, dtype=np.uint8)

            self.logger.log_info(f"  Image size: {image.shape[1]}x{image.shape[0]}")

            # Stage 1: Detect background color
            try:
                bg_hsv, confidence = detect_background_color(
                    image, self.config, self.logger
                )
            except Exception as e:
                self.logger.log_error(f"Stage 1 failed: {e}", exc_info=True)
                raise RuntimeError(f"Background detection failed: {e}")

            # Stage 2: Detect sprite
            sprite_info = None
            try:
                sprite_info = detect_sprite(image, bg_hsv, self.config, self.logger)
            except Exception as e:
                self.logger.log_warning(
                    f"Stage 2 failed: {e}, continuing without sprite detection"
                )

            # Stage 3: Refine mask (optional)
            refined_mask = None
            if self.config.use_grabcut and sprite_info is not None:
                try:
                    refined_mask = refine_mask(
                        image, sprite_info, self.config, self.logger
                    )
                except Exception as e:
                    self.logger.log_warning(f"Stage 3 failed: {e}, skipping refinement")

            # Create initial mask if we don't have a refined one
            if refined_mask is None:
                # Use sprite mask from Stage 2 if available
                if sprite_info is not None and "mask" in sprite_info:
                    initial_mask = sprite_info["mask"]
                    self.logger.log_info("  Using sprite mask from Stage 2")
                else:
                    initial_mask = self._create_basic_mask(image, bg_hsv)
                    self.logger.log_info("  Using basic HSV mask (fallback)")
            else:
                initial_mask = refined_mask

            # Stage 4: Cleanup
            try:
                final_mask = cleanup_mask(
                    image, initial_mask, bg_hsv, self.config, self.logger
                )
            except Exception as e:
                self.logger.log_error(f"Stage 4 failed: {e}", exc_info=True)
                raise RuntimeError(f"Cleanup failed: {e}")

            # Create RGBA output
            result = np.dstack([image, final_mask])
            result_img = Image.fromarray(result, "RGBA")

            # Autocrop if requested
            if self.config.autocrop:
                bbox = result_img.getbbox()
                if bbox:
                    orig_size = result_img.size
                    result_img = result_img.crop(bbox)
                    cropped_size = result_img.size
                    self.logger.log_info(
                        f"  Cropped {orig_size[0]}x{orig_size[1]} → {cropped_size[0]}x{cropped_size[1]}"
                    )

            # Determine output path
            output_path = self._compute_output_path(input_path)

            # Save
            result_img.save(output_path, "PNG", optimize=True)
            self.logger.log_info(f"  Saved → {output_path}")

            # Save log
            self.logger.save_image_log()

            return output_path

        except Exception as e:
            self.logger.log_error(f"Pipeline failed: {e}", exc_info=True)
            self.logger.save_image_log()
            raise

    def _create_basic_mask(
        self, image: np.ndarray, bg_hsv: tuple[float, float, float]
    ) -> np.ndarray:
        """
        Create basic HSV-based mask when GrabCut is not used

        Args:
            image: RGB image
            bg_hsv: Background color

        Returns:
            Binary mask (255 = foreground)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        h_bg, s_bg, v_bg = bg_hsv

        # Use mode-dependent tolerances
        if self.config.pixel_art_mode:
            h_tol, s_tol = 5.0, 5.0
            v_tol = 5.0
        else:
            h_tol, s_tol = 10.0, 15.0
            v_tol = 15.0

        # Create HSV range
        lower = np.array(
            [
                max(0, h_bg - h_tol) / 2,
                max(0, (s_bg - s_tol) * 2.55),
                max(0, (v_bg - v_tol) * 2.55),
            ]
        )
        upper = np.array(
            [
                min(180, h_bg + h_tol) / 2,
                min(255, (s_bg + s_tol) * 2.55),
                min(255, (v_bg + v_tol) * 2.55),
            ]
        )

        # Create background mask
        bg_mask = cv2.inRange(hsv, lower, upper)

        # Invert to get foreground
        fg_mask = cv2.bitwise_not(bg_mask)

        return fg_mask

    def _compute_output_path(self, input_path: Path) -> Path:
        """
        Compute output path

        Args:
            input_path: Input image path

        Returns:
            Output path
        """
        if self.config.output_path is not None:
            return self.config.output_path

        # Default: same directory, add _transparent suffix
        return input_path.with_stem(f"{input_path.stem}_transparent")
