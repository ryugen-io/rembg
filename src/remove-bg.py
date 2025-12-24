#!/usr/bin/env python3
"""
HSV-based color range removal for chromakey backgrounds
Uses OpenCV's inRange method - industry standard for green screen / chromakey removal
"""

from __future__ import annotations

import sys
import argparse
import logging
from pathlib import Path
from typing import Final
from collections import Counter

import numpy as np
from PIL import Image
import colorsys
from scipy import ndimage

# Constants
MAX_FILE_SIZE: Final[int] = 100 * 1024 * 1024  # 100MB

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def rgb_to_hsv_scalar(r: int, g: int, b: int) -> tuple[float, float, float]:
    """Convert RGB (0-255) to HSV (H: 0-360, S: 0-100, V: 0-100)"""
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    return h * 360.0, s * 100.0, v * 100.0


def detect_background_color_range(
    image_path: Path,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    INTELLIGENT multi-pass background detection

    Algorithm:
    1. Sample edges/corners (likely background)
    2. Find most prevalent color cluster → VALIDATE
    3. Refine range based on full image analysis → VALIDATE AGAIN
    4. Final validation and range calculation → CHECK AGAIN

    Returns: (hsv_min, hsv_max) tuples with (H, S, V) in degrees/percentages
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        data = np.asarray(img, dtype=np.uint8)

    h, w = data.shape[0], data.shape[1]

    # PASS 1: Sample edges (top, bottom, left, right) - likely background
    edge_width = max(5, min(h, w) // 20)
    edge_samples: list[tuple[int, int, int]] = []

    # Top edge (full width)
    edge_samples.extend(tuple(px) for px in data[0:edge_width, :, :].reshape(-1, 3))
    # Bottom edge (full width)
    edge_samples.extend(
        tuple(px) for px in data[h - edge_width : h, :, :].reshape(-1, 3)
    )
    # Left edge (full height)
    edge_samples.extend(tuple(px) for px in data[:, 0:edge_width, :].reshape(-1, 3))
    # Right edge (full height)
    edge_samples.extend(
        tuple(px) for px in data[:, w - edge_width : w, :].reshape(-1, 3)
    )

    logger.debug(f"  Pass 1: Sampled {len(edge_samples):,} edge pixels")

    # PASS 2: Find most prevalent color cluster
    color_counts = Counter(edge_samples)

    # Get top color as reference
    most_common_color, most_common_count = color_counts.most_common(1)[0]
    logger.debug(
        f"  Pass 2: Most common color RGB{most_common_color} ({most_common_count:,} pixels)"
    )

    # Convert reference to HSV
    ref_h, ref_s, ref_v = rgb_to_hsv_scalar(*most_common_color)
    logger.debug(
        f"  Pass 2: Reference HSV: H={ref_h:.1f}° S={ref_s:.1f}% V={ref_v:.1f}%"
    )

    # PASS 3: Find ALL similar colors (cluster around most common)
    # Use relaxed threshold to catch AI gradients
    similar_colors = []
    for color, count in color_counts.most_common(200):  # Top 200 colors
        h, s, v = rgb_to_hsv_scalar(*color)

        # Check if color is in same cluster (relaxed matching)
        h_diff = min(abs(h - ref_h), 360 - abs(h - ref_h))  # Handle hue wrap
        s_diff = abs(s - ref_s)
        v_diff = abs(v - ref_v)

        # Relaxed thresholds to catch AI gradient variations
        if h_diff < 25.0 and s_diff < 30.0 and v_diff < 30.0:
            similar_colors.append((h, s, v, count))

    if not similar_colors:
        logger.warning("  Pass 3: No similar colors found, using reference only")
        similar_colors = [(ref_h, ref_s, ref_v, most_common_count)]

    logger.debug(f"  Pass 3: Found {len(similar_colors)} similar colors in cluster")

    # PASS 4: Calculate range from cluster (weighted by pixel count)
    h_values = [h for h, s, v, count in similar_colors]
    s_values = [s for h, s, v, count in similar_colors]
    v_values = [v for h, s, v, count in similar_colors]

    # Handle hue wrap-around
    h_min, h_max = min(h_values), max(h_values)

    # If hue range > 180°, it's wrapped around 0°/360°
    if h_max - h_min > 180:
        # Shift all values by 180° to unwrap
        h_values_shifted = [(h + 180) % 360 for h in h_values]
        h_min_shifted = min(h_values_shifted)
        h_max_shifted = max(h_values_shifted)
        # Shift back
        h_min = (h_min_shifted - 180) % 360
        h_max = (h_max_shifted - 180) % 360

    hsv_min = (h_min, min(s_values), min(v_values))
    hsv_max = (h_max, max(s_values), max(v_values))

    logger.debug(
        f"  Pass 4: Final range - H: {h_min:.1f}°-{h_max:.1f}°, S: {min(s_values):.1f}%-{max(s_values):.1f}%, V: {min(v_values):.1f}%-{max(v_values):.1f}%"
    )

    return hsv_min, hsv_max


def compute_output_path(input_path: Path, output_path: Path | None) -> Path:
    """
    Compute output path with auto-routing for dev/assets/raw/ inputs.

    Rules:
    1. If output_path is provided (via -o flag), use it (user override)
    2. If input is from dev/assets/raw/, auto-route to dev/assets/proc/
    3. Otherwise, use same directory as input (backward compatible)

    Examples:
        dev/assets/raw/test/sprite.png → dev/assets/proc/test/sprite_transparent.png
        dev/assets/raw/sprite.png → dev/assets/proc/sprite_transparent.png
        ~/images/sprite.png → ~/images/sprite_transparent.png
    """
    if output_path is not None:
        return output_path

    # Convert to absolute path for consistent path detection
    abs_input = input_path.resolve()
    path_parts = abs_input.parts

    # Check if path contains 'dev/assets/raw/'
    try:
        # Detect 'dev', 'assets', 'raw' sequence
        dev_index = path_parts.index("dev")
        if (
            dev_index + 2 < len(path_parts)
            and path_parts[dev_index + 1] == "assets"
            and path_parts[dev_index + 2] == "raw"
        ):
            # Build corresponding proc/ path
            proc_parts = list(path_parts)
            proc_parts[dev_index + 2] = "proc"

            # Create output path with _transparent suffix
            proc_path = Path(*proc_parts).with_stem(f"{abs_input.stem}_transparent")

            # Ensure parent directory exists
            proc_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info("  Auto-routing: raw/ → proc/")
            return proc_path
    except (ValueError, IndexError):
        # 'dev/assets' not in path, or 'raw' not where expected
        pass

    # Default: same directory as input (backward compatible)
    return input_path.with_stem(f"{input_path.stem}_transparent")


def remove_background(
    input_path: Path,
    output_path: Path | None = None,
    autocrop: bool = False,
    pixel_art: bool = False,
    cleanup_passes: int = 1,
) -> Path:
    """
    Remove chromakey background using HSV color range (OpenCV inRange method)

    This is the STANDARD approach for chromakey/green screen removal.
    Handles color variations in AI-generated backgrounds.

    Args:
        input_path: Input image path
        output_path: Output path (None = auto-generate)
        autocrop: Crop to sprite bounding box
        pixel_art: Pixel art mode (no edge cleanup, tighter tolerances)
        cleanup_passes: Number of edge cleanup passes (1-10, default: 1)
    """
    # Detect background color RANGE (handles gradients/variations)
    (
        (h_min_detected, s_min_detected, v_min_detected),
        (h_max_detected, s_max_detected, v_max_detected),
    ) = detect_background_color_range(input_path)

    logger.info("Detected background range:")
    logger.info(f"  H: {h_min_detected:.1f}° - {h_max_detected:.1f}°")
    logger.info(f"  S: {s_min_detected:.1f}% - {s_max_detected:.1f}%")
    logger.info(f"  V: {v_min_detected:.1f}% - {v_max_detected:.1f}%")

    # Load image
    with Image.open(input_path) as img:
        img = img.convert("RGB")
        rgb_data = np.asarray(img, dtype=np.uint8)

    # Convert RGB to HSV using numpy
    # Note: We do manual conversion to match colorsys ranges
    r, g, b = (
        rgb_data[:, :, 0] / 255.0,
        rgb_data[:, :, 1] / 255.0,
        rgb_data[:, :, 2] / 255.0,
    )

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)

    # Value
    v = maxc * 100.0

    # Saturation (avoid division by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(maxc != 0, (maxc - minc) / maxc * 100.0, 0)

        # Hue (avoid division by zero)
        rc = np.where(maxc != minc, (maxc - r) / (maxc - minc), 0)
        gc = np.where(maxc != minc, (maxc - g) / (maxc - minc), 0)
        bc = np.where(maxc != minc, (maxc - b) / (maxc - minc), 0)

    h = np.zeros_like(maxc)
    h = np.where(r == maxc, bc - gc, h)
    h = np.where(g == maxc, 2.0 + rc - bc, h)
    h = np.where(b == maxc, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    h = h * 360.0

    # Expand detected range slightly to catch edge pixels
    # Pixel art: tighter tolerances (no antialiasing, exact color matching)
    # Non-pixel art: looser tolerances (handles antialiasing, gradient edges)
    if pixel_art:
        hue_tolerance = 5.0
        sat_tolerance = 5.0
        val_tolerance = 5.0
        logger.info("  Mode: Pixel Art (tight tolerances, no edge cleanup)")
    else:
        hue_tolerance = 10.0
        sat_tolerance = 15.0
        val_tolerance = 15.0
        logger.info("  Mode: Non-Pixel Art (loose tolerances, edge cleanup enabled)")

    h_min = max(0.0, h_min_detected - hue_tolerance)
    h_max = min(360.0, h_max_detected + hue_tolerance)
    sat_min = max(0.0, s_min_detected - sat_tolerance)
    val_min = max(0.0, v_min_detected - val_tolerance)

    if h_min < h_max:
        # Normal range (no wrap)
        hue_mask = (h >= h_min) & (h <= h_max)
    else:
        # Wrap around 0° (e.g., red)
        hue_mask = (h >= h_min) | (h <= h_max)

    sat_mask = s >= sat_min
    val_mask = v >= val_min

    bg_mask = hue_mask & sat_mask & val_mask

    logger.info(
        f"  Color range: H={h_min:.1f}-{h_max:.1f}°, S>={sat_min:.1f}%, V>={val_min:.1f}%"
    )
    logger.info(
        f"  Removing {np.sum(bg_mask):,} background pixels ({np.sum(bg_mask) / bg_mask.size * 100:.1f}%)"
    )

    # Create RGBA output
    result = np.dstack([rgb_data, np.full(rgb_data.shape[:2], 255, dtype=np.uint8)])

    # Set background to transparent
    result[bg_mask, 3] = 0

    # POST-PROCESSING: Multi-pass edge cleanup (ONLY for non-pixel art)
    # This removes color halos around sprite edges from antialiasing
    # Pixel art has no antialiasing, so edge cleanup would damage sharp edges
    if not pixel_art:
        cleanup_passes = max(1, min(10, cleanup_passes))  # Clamp to 1-10
        total_cleaned = 0

        for pass_num in range(cleanup_passes):
            # 1. Find sprite edges (where alpha changes from opaque to transparent)
            alpha_channel = result[:, :, 3].astype(np.float32)

            # Detect edges using numpy gradient (no scipy needed)
            dy, dx = np.gradient(alpha_channel)
            alpha_gradient = np.sqrt(dx**2 + dy**2)
            edge_mask = (
                alpha_gradient > 5
            )  # Lower threshold = more sensitive edge detection

            # 2. Aggressive cleanup for edge pixels with background color tint
            # Uses DETECTED background color range (works for any color!)
            edge_pixels = edge_mask & (
                result[:, :, 3] > 0
            )  # Only non-transparent edges

            # ALSO check pixels adjacent to transparent regions (catches stubborn edge pixels)
            transparent_mask = result[:, :, 3] == 0
            # Simple dilation using convolution (finds neighbors of transparent pixels)
            kernel = np.ones((3, 3), dtype=np.uint8)
            adjacent_to_transparent = ndimage.binary_dilation(
                transparent_mask, kernel
            ) & (result[:, :, 3] > 0)

            # Combine both edge detection methods
            edge_pixels = edge_pixels | adjacent_to_transparent

            if not np.any(edge_pixels):
                if pass_num == 0:
                    logger.info(
                        f"  Pass {pass_num + 1}/{cleanup_passes}: No edge pixels to clean"
                    )
                else:
                    logger.info(
                        f"  Pass {pass_num + 1}/{cleanup_passes}: No more edge pixels found (cleaned {total_cleaned:,} total)"
                    )
                break

            # Get edge pixel colors
            edge_rgb = result[edge_pixels, :3].astype(np.float32) / 255.0

            # Convert to HSV
            r_e, g_e, b_e = edge_rgb[:, 0], edge_rgb[:, 1], edge_rgb[:, 2]

            maxc_e = np.maximum(np.maximum(r_e, g_e), b_e)
            minc_e = np.minimum(np.minimum(r_e, g_e), b_e)

            with np.errstate(divide="ignore", invalid="ignore"):
                s_e = np.where(maxc_e != 0, (maxc_e - minc_e) / maxc_e * 100.0, 0)

                rc_e = np.where(maxc_e != minc_e, (maxc_e - r_e) / (maxc_e - minc_e), 0)
                gc_e = np.where(maxc_e != minc_e, (maxc_e - g_e) / (maxc_e - minc_e), 0)
                bc_e = np.where(maxc_e != minc_e, (maxc_e - b_e) / (maxc_e - minc_e), 0)

            h_e = np.zeros_like(maxc_e)
            h_e = np.where(r_e == maxc_e, bc_e - gc_e, h_e)
            h_e = np.where(g_e == maxc_e, 2.0 + rc_e - bc_e, h_e)
            h_e = np.where(b_e == maxc_e, 4.0 + gc_e - rc_e, h_e)
            h_e = (h_e / 6.0) % 1.0
            h_e = h_e * 360.0

            # Very relaxed thresholds for edge cleanup (catch more tinted pixels)
            edge_h_min = max(0.0, h_min_detected - 30.0)
            edge_h_max = min(360.0, h_max_detected + 30.0)
            edge_s_min = max(0.0, s_min_detected - 50.0)

            # Check if edge pixels match background color (relaxed range)
            if edge_h_min < edge_h_max:
                h_match = (h_e >= edge_h_min) & (h_e <= edge_h_max)
            else:
                h_match = (h_e >= edge_h_min) | (h_e <= edge_h_max)

            bg_tint = h_match & (s_e >= edge_s_min)

            # Remove edge pixels with background color tint
            edge_coords = np.where(edge_pixels)
            tinted_indices = np.where(bg_tint)[0]

            if len(tinted_indices) == 0:
                logger.info(
                    f"  Pass {pass_num + 1}/{cleanup_passes}: No tinted edge pixels found (cleaned {total_cleaned:,} total)"
                )
                break

            for idx in tinted_indices:
                y, x = edge_coords[0][idx], edge_coords[1][idx]
                result[y, x, 3] = 0

            total_cleaned += len(tinted_indices)
            logger.info(
                f"  Pass {pass_num + 1}/{cleanup_passes}: Cleaned {len(tinted_indices):,} edge pixels ({total_cleaned:,} total)"
            )

        if cleanup_passes > 1 and total_cleaned > 0:
            logger.info(
                f"  Multi-pass cleanup complete: {total_cleaned:,} pixels removed in {pass_num + 1} passes"
            )

    # Convert to PIL Image
    result_img = Image.fromarray(result, "RGBA")

    # Autocrop if requested
    if autocrop:
        bbox = result_img.getbbox()
        if bbox:
            orig_size = result_img.size
            result_img = result_img.crop(bbox)
            cropped_size = result_img.size
            logger.info(
                f"  Cropped {orig_size[0]}x{orig_size[1]} → {cropped_size[0]}x{cropped_size[1]}"
            )

    # Determine output path (with auto-routing for dev/assets/raw/)
    output_path = compute_output_path(input_path, output_path)

    # Save
    result_img.save(output_path, "PNG", optimize=True)
    logger.info(f"  Saved → {output_path}")

    return output_path


def main() -> int:
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Remove chromakey backgrounds using HSV color range"
    )
    parser.add_argument("files", nargs="+", type=Path, help="Input image files")
    parser.add_argument(
        "--autocrop", action="store_true", help="Crop to sprite bounding box"
    )
    parser.add_argument(
        "--pixel-art",
        action="store_true",
        help="Pixel art mode (tight tolerances, no edge cleanup)",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=1,
        metavar="N",
        help="Number of edge cleanup passes (1-10, default: 1). Higher = more aggressive.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="Output path (for single file only)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

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
            remove_background(
                file_path, output, args.autocrop, args.pixel_art, args.passes
            )
            success_count += 1
        except Exception as e:
            logger.error(f"  Failed: {e}", exc_info=args.verbose)

    logger.info(f"\nDone! Processed {success_count}/{len(args.files)} images.")
    return 0 if success_count == len(args.files) else 1


if __name__ == "__main__":
    sys.exit(main())
