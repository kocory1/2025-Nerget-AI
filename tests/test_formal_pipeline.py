#!/usr/bin/env python3
"""
Formal pipeline test
 - Visualize detected bounding boxes with per-box formal scores
 - Print final image-level formal score
"""

import sys
import os
import glob
import random

# Add project root to Python path (tests run from subdir)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipelines.formal_pipeline import FormalPipeline
from src.visualizers.formal_visualizer import FormalVisualizer


def pick_random_image() -> str:
    """Pick one random image from dataset"""
    candidates = []
    for d in ("dataset/minimal", "dataset/maximal"):
        candidates.extend(glob.glob(os.path.join(d, "*.jpg")))
        candidates.extend(glob.glob(os.path.join(d, "*.jpeg")))
        candidates.extend(glob.glob(os.path.join(d, "*.png")))
    if not candidates:
        raise FileNotFoundError("No images found in dataset directory.")
    random.seed()  # system seed
    return random.choice(candidates)


def test_formal_pipeline_visual():
    """Run Formal pipeline and visualize (1 random sample)"""
    print("Formal pipeline test (YOLOS + label-based formal score)")
    print("=" * 70)

    # 1) Initialize pipeline
    pipeline = FormalPipeline()

    # 2) Pick a random test image
    image_path = pick_random_image()
    print(f"Selected test image: {image_path}")

    # 3) Run analysis
    result = pipeline.detect_and_analyze(image_path, conf_threshold=0.8, verbose=True)

    if result.get("success", False):
        detections = result.get("detections", [])
        overall = result.get("formal_overall_score", 0.0)

        print("\nFinal formal image score:", f"{overall:.3f}")
        print("Detections:", result.get("total_detections"), "/ contributing:", result.get("contributing_detections"))

        # 4) visualize
        if detections:
            FormalVisualizer().visualize(image_path, detections, overall_score=overall)
        else:
            print("No detections to visualize.")
    else:
        print(f"Test failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    test_formal_pipeline_visual()


