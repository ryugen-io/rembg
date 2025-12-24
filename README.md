# rembg

Background removal tool for game sprites using HSV chromakey detection.

## Installation

```bash
./install.sh
```

This installs rembg to `~/.local/share/rembg/` with its own virtual environment.

## Usage

### Direct command (after installation)

```bash
rembg --autocrop --pixel-art --passes 3 image.png
```

### Fish function (recommended)

```fish
rembg pixel image.png 3        # Pixel art mode, 3 cleanup passes
rembg remove image.png         # Standard mode
rembg aggressive image.png 5   # Aggressive cleanup
```

## Scripts

- `remove-bg.py` - Main background removal (HSV-based)
- `remove-bg-aggressive.py` - Post-processing cleanup
- `remove-bg-hardened.py` - Alternative cleanup approach

## Options

- `--autocrop` - Crop to sprite bounding box
- `--pixel-art` - Pixel art mode (tight tolerances, no edge cleanup)
- `--passes N` - Number of cleanup passes (1-10, default: 1)
