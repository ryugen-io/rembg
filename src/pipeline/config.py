"""
PipelineConfig: Configuration for background removal pipeline
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PipelineConfig:
    """Configuration for background removal pipeline"""

    # Stage 1: Background Detection
    kmeans_k_values: tuple[int, ...] = (3, 5, 8)
    kmeans_max_iter: int = 100
    kmeans_epsilon: float = 0.1
    kmeans_attempts: int = 10

    # Stage 2: Sprite Detection
    flood_fill_connectivity: int = 8
    min_sprite_size_ratio: float = 0.05  # Min 5% of image
    sprite_center_tolerance: float = 0.3  # Within 30% of center

    # Stage 3: Refinement
    use_grabcut: bool = False
    grabcut_iterations: int = 5

    # Stage 4: Cleanup
    pixel_art_mode: bool = False
    cleanup_passes: int = 1
    morphology_kernel_size: int = 3

    # Edge cleanup tolerances
    pixel_art_hue_tolerance: float = 8.0
    pixel_art_sat_tolerance: float = 15.0
    normal_hue_tolerance: float = 30.0
    normal_sat_tolerance: float = 50.0

    # Output
    autocrop: bool = False
    output_path: Optional[Path] = None

    # Validation
    skip_validation: bool = False

    @property
    def edge_tolerances(self) -> tuple[float, float]:
        """Get edge cleanup tolerances based on mode"""
        if self.pixel_art_mode:
            return (self.pixel_art_hue_tolerance, self.pixel_art_sat_tolerance)
        return (self.normal_hue_tolerance, self.normal_sat_tolerance)
