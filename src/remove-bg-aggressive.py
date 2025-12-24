#!/usr/bin/env python3
"""
Aggressive chromakey edge cleanup pass
Use this after initial chromakey removal to catch stubborn edge pixels
"""

from PIL import Image
import numpy as np
import argparse
from pathlib import Path
from collections import Counter


def rgb_to_ycbcr(r, g, b):
    """Convert RGB to YCbCr (JPEG formula)"""
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return y, cb, cr


def compute_output_path(input_path, output_path):
    """
    Compute output path with auto-routing for dev/assets/raw/ inputs.

    For aggressive cleanup:
    - If input is already in proc/, overwrite it (default behavior)
    - If input is from raw/, route to proc/ with _transparent suffix
    - If output_path provided, use it (user override)
    """

    if output_path is not None:
        return output_path

    # Convert to Path object if string
    input_path = Path(input_path)
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

            proc_path = Path(*proc_parts)
            if not proc_path.stem.endswith("_transparent"):
                proc_path = proc_path.with_stem(f"{proc_path.stem}_transparent")

            # Ensure parent directory exists
            proc_path.parent.mkdir(parents=True, exist_ok=True)

            print("    Auto-routing: raw/ → proc/")
            return proc_path
    except (ValueError, IndexError):
        pass

    # Default: overwrite input (original behavior)
    return input_path


def detect_background_color(image_path):
    """Detect dominant background color from corner samples"""
    img = Image.open(image_path).convert("RGB")
    data = np.array(img, dtype=np.uint8)

    h, w = data.shape[0], data.shape[1]
    sample_size = max(10, min(h, w) // 10)

    # Sample all 4 corners
    corners = []
    corners.extend(
        tuple(px) for px in data[0:sample_size, 0:sample_size, :].reshape(-1, 3)
    )
    corners.extend(
        tuple(px) for px in data[0:sample_size, w - sample_size : w, :].reshape(-1, 3)
    )
    corners.extend(
        tuple(px) for px in data[h - sample_size : h, 0:sample_size, :].reshape(-1, 3)
    )
    corners.extend(
        tuple(px)
        for px in data[h - sample_size : h, w - sample_size : w, :].reshape(-1, 3)
    )

    # Get most common color
    color_counts = Counter(corners)
    bg_color, count = color_counts.most_common(1)[0]

    print(f"    Detected background: RGB{bg_color}")
    return bg_color


def aggressive_cleanup(input_path, output_path=None, key_color=None):
    """
    Aggressive edge cleanup for stubborn chromakey pixels

    Args:
        input_path: Path to input image (already processed with chromakey)
        output_path: Path to save cleaned image (None = overwrite input)
        key_color: RGB tuple of chroma key color (None = auto-detect)
    """
    print("  Applying aggressive cleanup...")

    # Auto-detect background color if not provided
    if key_color is None:
        key_color = detect_background_color(input_path)

    # Load image
    img = Image.open(input_path).convert("RGBA")
    data = np.array(img, dtype=np.float32)

    r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]

    # Convert key color to YCbCr
    key_r, key_g, key_b = float(key_color[0]), float(key_color[1]), float(key_color[2])
    key_y, key_cb, key_cr = rgb_to_ycbcr(key_r, key_g, key_b)

    # Convert image to YCbCr
    y, cb, cr = rgb_to_ycbcr(r, g, b)

    # Calculate euclidean distance in CbCr plane
    cb_dist = cb - key_cb
    cr_dist = cr - key_cr
    color_distance = np.sqrt(cb_dist**2 + cr_dist**2)

    # More aggressive thresholds
    threshold_near = 55  # Expanded
    threshold_far = 85  # Extended gradient

    has_alpha = a > 0
    new_alpha = a.copy()

    # Fully transparent zone
    close_mask = (color_distance < threshold_near) & has_alpha
    new_alpha[close_mask] = 0

    # Gradient zone
    middle_mask = (
        (color_distance >= threshold_near)
        & (color_distance <= threshold_far)
        & has_alpha
    )
    gradient = (color_distance[middle_mask] - threshold_near) / (
        threshold_far - threshold_near
    )
    new_alpha[middle_mask] = a[middle_mask] * gradient

    # Extra cleanup: background-color-like pixels with low alpha
    key_r, key_g, key_b = key_color
    tolerance = 50
    bg_ish = (
        (np.abs(r - key_r) < tolerance)
        & (np.abs(g - key_g) < tolerance)
        & (np.abs(b - key_b) < tolerance)
        & (a < 200)
        & (a > 0)
    )
    new_alpha[bg_ish] = 0

    print(f"    -> Removed: {np.sum(close_mask)} pixels")
    print(f"    -> Gradient: {np.sum(middle_mask)} pixels")
    print(f"    -> Extra cleanup: {np.sum(bg_ish)} background-ish pixels")

    # Apply new alpha
    data[:, :, 3] = new_alpha

    # Save result (with auto-routing for dev-assets/raw/)
    output_path = compute_output_path(input_path, output_path)

    result = Image.fromarray(data.astype(np.uint8), "RGBA")
    result.save(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Aggressive cleanup for stubborn chromakey edge pixels"
    )
    parser.add_argument("files", nargs="+", help="Input image files")
    parser.add_argument("-o", "--output", help="Output path (for single file)")

    args = parser.parse_args()

    print(f"Found {len(args.files)} image(s)")

    for i, file_path in enumerate(args.files, 1):
        print(f"  Processing: {file_path}")
        output = args.output if len(args.files) == 1 else None
        result_path = aggressive_cleanup(file_path, output)
        print(f"    -> {result_path}")

    print(f"\nDone! Processed {len(args.files)}/{len(args.files)} images.")


if __name__ == "__main__":
    main()
