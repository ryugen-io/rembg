"""
Pipeline Stages
"""

from .s1_background import detect_background_color
from .s2_sprite import detect_sprite
from .s3_refinement import refine_mask
from .s4_cleanup import cleanup_mask

__all__ = ["detect_background_color", "detect_sprite", "refine_mask", "cleanup_mask"]
