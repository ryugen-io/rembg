"""
Production-Ready Multi-Stage Background Removal Pipeline
"""

from .pipeline import BackgroundRemovalPipeline
from .logger import PipelineLogger
from .config import PipelineConfig

__all__ = ["BackgroundRemovalPipeline", "PipelineLogger", "PipelineConfig"]
