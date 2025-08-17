"""
Plotting utilities for visualizing detection and analysis results.
Moved from src/utils/visualization.py to keep visualizers cohesive.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List
import os

from ..config.settings import VISUALIZATION_CONFIG


def visualize_detection_results(image: np.ndarray, results: List[Dict]) -> None:
    """Visualize YOLO detections and color analysis results."""
    if not results:
        print("No results to visualize.")
        return

    n_results = len(results)
    fig, axes = plt.subplots(2, n_results, figsize=(5 * n_results, 10))
    if n_results == 1:
        axes = axes.reshape(2, 1)

    for i, result in enumerate(results):
        bbox = result['bbox']
        confidence = result['confidence']
        score = result.get('score', 0.0)
        max_sat = result.get('max_saturation', 0.0)
        labels = result.get('labels')
        region_shape = result.get('region_shape', (100, 100, 3))

        x1, y1, x2, y2 = bbox

        axes[0, i].imshow(image)
        axes[0, i].set_title(f"{result['class_name']} (Score: {score:.3f})")
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2)
        axes[0, i].add_patch(rect)
        label_text = f"conf:{confidence:.2f}\nscore:{score:.2f}\ntrim_sat:{max_sat:.0f}"
        axes[0, i].text(x1, y1 - 10, label_text, fontsize=8, color='red', weight='bold',
                        bbox=dict(facecolor='white', alpha=0.7))
        axes[0, i].axis('off')

        if labels is not None:
            height, width = region_shape[:2]
            clustered_image = np.zeros((height, width, 3), dtype=np.uint8)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            import random
            random.seed(42)
            cluster_colors = {j: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                              for j in range(n_clusters)}
            noise_color = (0, 0, 0)

            for idx, label in enumerate(labels):
                row = idx // width
                col = idx % width
                if row < height and col < width:
                    if label == -1:
                        clustered_image[row, col] = noise_color
                    else:
                        clustered_image[row, col] = cluster_colors.get(label, (255, 255, 255))

            axes[1, i].imshow(clustered_image)
            axes[1, i].set_title(f'DBSCAN Clustering\n({n_clusters} clusters)')
            axes[1, i].axis('off')
        else:
            axes[1, i].text(0.5, 0.5, 'No clustering\nresult',
                            ha='center', va='center', transform=axes[1, i].transAxes)
            axes[1, i].set_title('No Clustering')
            axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()


