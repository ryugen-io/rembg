#!/usr/bin/env python3
"""
Production-Ready Background Removal Pipeline CLI

Multi-stage pipeline with K-Means, Flood Fill, Connected Components,
optional GrabCut, and comprehensive logging.
"""

import argparse
import sys
from pathlib import Path

from rembg.pipeline import BackgroundRemovalPipeline, PipelineConfig, PipelineLogger


def main() -> int:
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Production-ready background removal using multi-stage CV pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  %(prog)s input.png

  # Production mode with GrabCut and debug logging
  %(prog)s --use-grabcut --debug --passes 5 input.png

  # Pixel art mode
  %(prog)s --pixel-art --passes 3 input.png

  # Custom log file
  %(prog)s --debug --log-file ~/debug.log input.png
        """,
    )

    # Positional arguments
    parser.add_argument("files", nargs="+", type=Path, help="Input image files")

    # Output options
    parser.add_argument(
        "-o", "--output", type=Path, help="Output path (for single file only)"
    )
    parser.add_argument(
        "--autocrop", action="store_true", help="Crop to sprite bounding box"
    )

    # Mode options
    parser.add_argument(
        "--pixel-art",
        action="store_true",
        help="Pixel art mode (tight tolerances, no blur)",
    )
    parser.add_argument(
        "--use-grabcut",
        action="store_true",
        help="Enable GrabCut refinement (Stage 3) - more accurate but slower",
    )

    # Cleanup options
    parser.add_argument(
        "--passes",
        type=int,
        default=1,
        metavar="N",
        help="Number of edge cleanup passes (1-10, default: 1). Higher = more aggressive.",
    )

    # Validation options
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip validation checks (faster but less robust)",
    )

    # Logging options
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode with detailed logging"
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Custom log file path (default: ~/.local/share/rembg/debug.log)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.output and len(args.files) > 1:
        parser.error("--output can only be used with a single input file")

    # Create logger
    logger = PipelineLogger(
        log_file=args.log_file, debug_mode=args.debug, verbose=args.verbose
    )

    logger.log_info(f"Processing {len(args.files)} image(s)...")

    success_count = 0

    for file_path in args.files:
        try:
            # Create config for this image
            config = PipelineConfig(
                pixel_art_mode=args.pixel_art,
                use_grabcut=args.use_grabcut,
                cleanup_passes=max(1, min(10, args.passes)),
                autocrop=args.autocrop,
                output_path=args.output if len(args.files) == 1 else None,
                skip_validation=args.no_validation,
            )

            # Create pipeline
            pipeline = BackgroundRemovalPipeline(config=config, logger=logger)

            # Process
            output_path = pipeline.process(file_path)
            logger.log_info(f"✓ Success: {file_path} → {output_path}")

            success_count += 1

        except FileNotFoundError as e:
            logger.log_error(f"✗ File not found: {e}")
        except Exception as e:
            logger.log_error(f"✗ Failed: {e}", exc_info=args.verbose or args.debug)

    logger.log_info(f"\nDone! Processed {success_count}/{len(args.files)} images.")

    return 0 if success_count == len(args.files) else 1


if __name__ == "__main__":
    sys.exit(main())
