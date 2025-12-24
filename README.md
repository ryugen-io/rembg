# rembg

Multi-stage background removal pipeline using K-Means clustering, flood fill, connected components analysis, and optional GrabCut refinement.

## Installation

```bash
pipx install .
```

Or use the wrapper script:

```bash
./install.sh
```

## Usage

Basic usage:

```bash
rembg input.png
```

With options:

```bash
rembg --autocrop --pixel-art --passes 3 input.png
rembg --use-grabcut --debug input.png
rembg --output result.png input.png
```

Multiple files:

```bash
rembg image1.png image2.png image3.png
```

### Fish Shell Wrapper

The `functions/rembg.fish` provides convenience shortcuts:

```fish
rembg pixel image.png 3        # Pixel art mode, 3 cleanup passes
rembg remove image.png         # Standard mode
rembg aggressive image.png 5   # GrabCut enabled, 5 passes
```

## CLI Options

### Output
- `-o, --output PATH` - Output path (single file only)
- `--autocrop` - Crop to sprite bounding box

### Mode
- `--pixel-art` - Tight tolerances, no blur (for pixel art)
- `--use-grabcut` - Enable GrabCut refinement (slower, more accurate)

### Cleanup
- `--passes N` - Edge cleanup passes (1-10, default: 1)

### Validation
- `--no-validation` - Skip validation checks

### Logging
- `-v, --verbose` - Verbose output
- `--debug` - Debug logging with detailed stage info
- `--log-file PATH` - Custom log file (default: ~/.local/share/rembg/debug.log)

## Pipeline Stages

### Stage 1: Background Detection
K-Means clustering with multiple k-values (3, 5, 8) to identify background color.

### Stage 2: Sprite Detection
Flood fill from image corners with connected components analysis. Selects largest component near center.

### Stage 3: Refinement (Optional)
GrabCut algorithm for edge refinement. Enable with `--use-grabcut`.

### Stage 4: Edge Cleanup
Morphological operations and HSV-based edge cleanup. Configurable passes and tolerances based on mode.

## Configuration

Default tolerances (pixel art mode):
- Hue: 8.0
- Saturation: 15.0

Default tolerances (normal mode):
- Hue: 30.0
- Saturation: 50.0

Customize via `PipelineConfig` in code.

## Requirements

- Python >= 3.8
- opencv-python
- numpy >= 1.24.0
- Pillow >= 10.0.0
- scipy >= 1.11.0
