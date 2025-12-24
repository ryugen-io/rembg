#!/usr/bin/env python3
"""
Production-quality chromakey (chroma key) background removal - HARDENED VERSION
Algorithm based on gc-films.com chromakey implementation
Uses YCbCr color space and euclidean distance for proper feathering

Features:
- Type hints (Python 3.10+)
- Comprehensive error handling
- Input validation
- Security hardening
- Latest Pillow/NumPy best practices
"""

from __future__ import annotations

import sys
import argparse
import logging
from pathlib import Path
from typing import Final
from collections import Counter

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

# Constants
MAX_FILE_SIZE: Final[int] = 100 * 1024 * 1024  # 100MB
MIN_IMAGE_SIZE: Final[tuple[int, int]] = (1, 1)
MAX_IMAGE_SIZE: Final[tuple[int, int]] = (16384, 16384)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class ChromakeyError(Exception):
    """Base exception for chromakey processing errors"""
    pass


class ValidationError(ChromakeyError):
    """Raised when input validation fails"""
    pass


def rgb_to_ycbcr(
    r: NDArray[np.floating],
    g: NDArray[np.floating],
    b: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Convert RGB to YCbCr color space (JPEG formula from gc-films.com)

    Args:
        r, g, b: RGB channels as float arrays

    Returns:
        Tuple of (Y, Cb, Cr) channels
    """
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return y, cb, cr


def validate_image_path(path: Path) -> None:
    """
    Validate image file path for security and accessibility

    Args:
        path: Path to validate

    Raises:
        ValidationError: If validation fails
    """
    if not path.exists():
        raise ValidationError(f"File does not exist: {path}")

    if not path.is_file():
        raise ValidationError(f"Not a file: {path}")

    # Check file size
    file_size = path.stat().st_size
    if file_size > MAX_FILE_SIZE:
        raise ValidationError(
            f"File too large: {file_size / 1024 / 1024:.1f}MB "
            f"(max {MAX_FILE_SIZE / 1024 / 1024}MB)"
        )

    if file_size == 0:
        raise ValidationError(f"File is empty: {path}")

    # Prevent path traversal
    try:
        path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        # File is outside CWD, check if it's an absolute path we trust
        if not path.is_absolute():
            raise ValidationError(f"Suspicious file path: {path}")


def detect_background_color(image_path: Path) -> tuple[int, int, int]:
    """
    Detect dominant background color from corner samples

    Args:
        image_path: Path to image file

    Returns:
        RGB tuple of dominant background color

    Raises:
        ChromakeyError: If detection fails
    """
    try:
        with Image.open(image_path) as img:
            # Verify image integrity
            img.verify()

        # Reopen for actual processing (verify() invalidates the image)
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            data = np.asarray(img, dtype=np.uint8)

        h, w = data.shape[0], data.shape[1]

        if h < MIN_IMAGE_SIZE[1] or w < MIN_IMAGE_SIZE[0]:
            raise ValidationError(f"Image too small: {w}x{h}")

        if h > MAX_IMAGE_SIZE[1] or w > MAX_IMAGE_SIZE[0]:
            raise ValidationError(f"Image too large: {w}x{h}")

        sample_size = max(10, min(h, w) // 10)

        # Sample all 4 corners
        corners: list[tuple[int, int, int]] = []
        corners.extend(
            tuple(px) for px in
            data[0:sample_size, 0:sample_size, :].reshape(-1, 3)
        )
        corners.extend(
            tuple(px) for px in
            data[0:sample_size, w-sample_size:w, :].reshape(-1, 3)
        )
        corners.extend(
            tuple(px) for px in
            data[h-sample_size:h, 0:sample_size, :].reshape(-1, 3)
        )
        corners.extend(
            tuple(px) for px in
            data[h-sample_size:h, w-sample_size:w, :].reshape(-1, 3)
        )

        if not corners:
            raise ChromakeyError("No corner pixels found")

        # Find dominant color
        color_counts = Counter(corners)
        bg_color, count = color_counts.most_common(1)[0]

        logger.debug(
            f"Detected background: RGB{bg_color} "
            f"(#{bg_color[0]:02X}{bg_color[1]:02X}{bg_color[2]:02X}) "
            f"- {count} samples"
        )

        return bg_color

    except UnidentifiedImageError as e:
        raise ChromakeyError(f"Cannot identify image format: {e}")
    except OSError as e:
        raise ChromakeyError(f"Error reading image: {e}")
    except Exception as e:
        raise ChromakeyError(f"Unexpected error during detection: {e}")


def remove_background(
    input_path: Path,
    output_path: Path | None = None,
    autocrop: bool = False
) -> Path:
    """
    Remove chromakey background using YCbCr color distance

    Args:
        input_path: Path to input image
        output_path: Path to save output (None = auto-generate)
        autocrop: Whether to crop to sprite bounding box

    Returns:
        Path to output file

    Raises:
        ChromakeyError: If processing fails
    """
    # Validate input
    validate_image_path(input_path)

    # Detect key color
    key_color = detect_background_color(input_path)
    logger.info(
        f"Background: RGB{key_color} = "
        f"#{key_color[0]:02X}{key_color[1]:02X}{key_color[2]:02X}"
    )

    # Convert key color to YCbCr
    key_r, key_g, key_b = (float(c) for c in key_color)
    key_y, key_cb, key_cr = rgb_to_ycbcr(
        np.array([key_r]), np.array([key_g]), np.array([key_b])
    )

    try:
        # Load image
        with Image.open(input_path) as img:
            img = img.convert('RGBA')
            data = np.asarray(img, dtype=np.float32)

        r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]

        # Convert to YCbCr
        y, cb, cr = rgb_to_ycbcr(r, g, b)

        # Calculate euclidean distance in CbCr plane
        cb_dist = cb - float(key_cb[0])
        cr_dist = cr - float(key_cr[0])

        with np.errstate(invalid='raise'):
            color_distance = np.sqrt(cb_dist**2 + cr_dist**2)

        # Auto-calculate thresholds
        distances_sorted = np.sort(color_distance.flatten())
        close_distances = distances_sorted[distances_sorted < 50.0]

        if len(close_distances) == 0:
            logger.warning("No background pixels detected, using default thresholds")
            threshold_near = 40.0
        else:
            threshold_near = float(np.percentile(close_distances, 90))

        threshold_far = threshold_near + 30.0

        logger.debug(f"Thresholds: near={threshold_near:.1f}, far={threshold_far:.1f}")

        # Calculate new alpha with gradient
        new_alpha = np.zeros_like(a, dtype=np.float32)

        # Region 1: Close to key color → fully transparent
        close_mask = color_distance < threshold_near
        new_alpha[close_mask] = 0.0

        # Region 3: Far from key color → keep original alpha
        far_mask = color_distance > threshold_far
        new_alpha[far_mask] = a[far_mask]

        # Region 2: Middle → gradient transparency
        middle_mask = ~close_mask & ~far_mask
        if np.any(middle_mask):
            gradient = (color_distance[middle_mask] - threshold_near) / (threshold_far - threshold_near)
            new_alpha[middle_mask] = a[middle_mask] * gradient

        # Apply new alpha
        data[:,:,3] = new_alpha

        # Convert to uint8
        result_data = np.clip(data, 0, 255).astype(np.uint8)
        result = Image.fromarray(result_data, 'RGBA')

        # Autocrop if requested
        if autocrop:
            bbox = result.getbbox()
            if bbox:
                orig_size = result.size
                result = result.crop(bbox)
                cropped_size = result.size
                logger.info(
                    f"  Cropped {orig_size[0]}x{orig_size[1]} → "
                    f"{cropped_size[0]}x{cropped_size[1]}"
                )

        # Determine output path
        if output_path is None:
            output_path = input_path.with_stem(f"{input_path.stem}_transparent")

        # Save
        result.save(output_path, 'PNG', optimize=True)
        logger.info(f"  Saved → {output_path}")

        return output_path

    except (UnidentifiedImageError, OSError) as e:
        raise ChromakeyError(f"Error processing image: {e}")
    except np.linalg.LinAlgError as e:
        raise ChromakeyError(f"Numerical error: {e}")
    except Exception as e:
        raise ChromakeyError(f"Unexpected error: {e}")


def main() -> int:
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Remove chromakey backgrounds from sprite images',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'files',
        nargs='+',
        type=Path,
        help='Input image files'
    )
    parser.add_argument(
        '--autocrop',
        action='store_true',
        help='Crop to sprite bounding box'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Output path (for single file only)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.output and len(args.files) > 1:
        logger.error("--output can only be used with a single input file")
        return 1

    logger.info(f"Processing {len(args.files)} image(s)...")

    success_count = 0
    for file_path in args.files:
        try:
            logger.info(f"Processing: {file_path}")
            output = args.output if len(args.files) == 1 else None
            remove_background(file_path, output, args.autocrop)
            success_count += 1
        except ChromakeyError as e:
            logger.error(f"  Failed: {e}")
        except KeyboardInterrupt:
            logger.warning("\nInterrupted by user")
            return 130
        except Exception as e:
            logger.error(f"  Unexpected error: {e}", exc_info=args.verbose)

    logger.info(f"\nDone! Processed {success_count}/{len(args.files)} images successfully.")

    return 0 if success_count == len(args.files) else 1


if __name__ == "__main__":
    sys.exit(main())
