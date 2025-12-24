"""
PipelineLogger: Structured JSON logging for background removal pipeline
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class PipelineLogger:
    """Production-ready logger with JSON output and debug modes"""

    def __init__(
        self,
        log_file: Optional[Path] = None,
        debug_mode: bool = False,
        verbose: bool = False,
    ):
        self.log_file = log_file or Path.home() / ".local/share/rembg/debug.log"
        self.debug_mode = debug_mode
        self.verbose = verbose
        self.current_image: Optional[Dict[str, Any]] = None
        self.logs: list[Dict[str, Any]] = []

        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Setup Python logger
        self.logger = logging.getLogger("rembg.pipeline")
        self.logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)

    def start_image(self, image_path: Path):
        """Start logging for a new image"""
        self.current_image = {
            "image": str(image_path),
            "timestamp": datetime.now().isoformat(),
            "stages": [],
        }

    def log_s1(self, **kwargs):
        """Log Stage 1: Background Detection"""
        self._log_stage("s1_background_detection", kwargs)

    def log_s2(self, **kwargs):
        """Log Stage 2: Sprite Detection"""
        self._log_stage("s2_sprite_detection", kwargs)

    def log_s3(self, **kwargs):
        """Log Stage 3: Mask Refinement"""
        self._log_stage("s3_mask_refinement", kwargs)

    def log_s4(self, **kwargs):
        """Log Stage 4: Cleanup"""
        self._log_stage("s4_cleanup", kwargs)

    def _log_stage(self, stage_name: str, data: Dict[str, Any]):
        """Internal method to log a stage"""
        if self.current_image is None:
            raise RuntimeError("Must call start_image() before logging stages")

        stage_log = {
            "stage": stage_name,
            "timestamp": datetime.now().isoformat(),
            **data,
        }
        self.current_image["stages"].append(stage_log)

        if self.debug_mode:
            print(f"[{stage_name}] {json.dumps(data, indent=2)}")

    def log_info(self, message: str):
        """Log info message"""
        self.logger.info(message)
        if self.verbose:
            print(f"INFO: {message}")

    def log_warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
        print(f"WARNING: {message}")

    def log_error(self, message: str, exc_info: bool = False):
        """Log error message"""
        self.logger.error(message, exc_info=exc_info)
        print(f"ERROR: {message}")

    def save_image_log(self):
        """Save current image log to file"""
        if self.current_image is None:
            return

        self.logs.append(self.current_image)

        with open(self.log_file, "a") as f:
            json.dump(self.current_image, f)
            f.write("\n")

        self.current_image = None
