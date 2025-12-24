"""
Stage 1: Background Color Detection using K-Means Clustering
"""

import colorsys
from typing import Tuple

import cv2
import numpy as np

from ..config import PipelineConfig
from ..logger import PipelineLogger


def rgb_to_hsv_scalar(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Convert RGB (0-255) to HSV (H: 0-360, S: 0-100, V: 0-100)"""
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    return h * 360.0, s * 100.0, v * 100.0


def sample_edges(image: np.ndarray, edge_width_ratio: float = 0.05) -> np.ndarray:
    """
    Sample edge pixels from all four sides of the image

    Args:
        image: RGB image (H, W, 3)
        edge_width_ratio: Ratio of image dimension to use as edge width

    Returns:
        Array of RGB samples (N, 3)
    """
    h, w = image.shape[:2]
    edge_width = max(5, int(min(h, w) * edge_width_ratio))

    edge_samples = []

    # Top edge
    edge_samples.extend(image[0:edge_width, :, :].reshape(-1, 3))
    # Bottom edge
    edge_samples.extend(image[h - edge_width : h, :, :].reshape(-1, 3))
    # Left edge
    edge_samples.extend(image[:, 0:edge_width, :].reshape(-1, 3))
    # Right edge
    edge_samples.extend(image[:, w - edge_width : w, :].reshape(-1, 3))

    return np.array(edge_samples, dtype=np.float32)


def calculate_cluster_stability(
    labels: np.ndarray, centers: np.ndarray
) -> Tuple[float, int]:
    """
    Calculate stability score and find dominant cluster

    Stability = ratio of largest cluster to total samples
    Higher is better (more uniform background)

    Returns:
        (confidence_score, dominant_cluster_index)
    """
    unique, counts = np.unique(labels, return_counts=True)
    dominant_idx = unique[np.argmax(counts)]
    dominant_count = np.max(counts)
    total_count = len(labels)

    confidence = dominant_count / total_count
    return confidence, dominant_idx


def detect_background_color(
    image: np.ndarray, config: PipelineConfig, logger: PipelineLogger
) -> Tuple[Tuple[float, float, float], float]:
    """
    Detect background color using K-Means clustering on edge samples

    Algorithm:
    1. Sample edges of image (likely background)
    2. Run K-Means with multiple K values
    3. Select K with highest cluster stability
    4. Return dominant cluster center as background color

    Args:
        image: RGB image (H, W, 3)
        config: Pipeline configuration
        logger: Logger instance

    Returns:
        ((h_min, s_min, v_min), confidence) - HSV range and confidence score
    """
    logger.log_info("Stage 1: Detecting background color...")

    # Sample edges
    edge_samples = sample_edges(image)
    logger.log_info(f"  Sampled {len(edge_samples):,} edge pixels")

    # Try multiple K values and find most stable clustering
    best_cluster_center = None
    best_confidence = 0.0
    best_k = None
    k_results = {}

    for k in config.kmeans_k_values:
        # K-Means clustering
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            config.kmeans_max_iter,
            config.kmeans_epsilon,
        )

        _, labels, centers = cv2.kmeans(
            edge_samples,
            k,
            None,
            criteria,
            config.kmeans_attempts,
            cv2.KMEANS_PP_CENTERS,
        )

        # Calculate stability
        confidence, dominant_idx = calculate_cluster_stability(labels, centers)

        k_results[k] = {
            "confidence": confidence,
            "dominant_color_rgb": centers[dominant_idx].tolist(),
        }

        if confidence > best_confidence:
            best_confidence = confidence
            best_cluster_center = centers[dominant_idx]
            best_k = k

    # Convert best cluster to HSV
    r, g, b = best_cluster_center
    h, s, v = rgb_to_hsv_scalar(int(r), int(g), int(b))

    # Create HSV range (detected color as baseline)
    hsv_range = (h, s, v)

    # Log results
    logger.log_s1(
        method="kmeans",
        k_tested=list(config.kmeans_k_values),
        best_k=best_k,
        k_results=k_results,
        dominant_color_rgb=best_cluster_center.tolist(),
        dominant_color_hsv={"h": h, "s": s, "v": v},
        confidence=best_confidence,
    )

    logger.log_info(
        f"  Detected: H={h:.1f}° S={s:.1f}% V={v:.1f}% (K={best_k}, confidence={best_confidence:.2%})"
    )

    return hsv_range, best_confidence
