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
from collections import Counter

# Add project root to Python path (tests run from subdir)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipelines.formal_pipeline import FormalPipeline
from src.visualizers.formal_visualizer import FormalVisualizer
from src.detectors.object_detector import ObjectDetector


PREFERRED_IMAGE = "/Users/bagminsu/Documents/옷마카세_ai/2025-Nerget-AI/dataset/examples/ex5.jpeg"


def pick_random_image() -> str:
    """Pick one random image from dataset"""
    candidates = []
    for d in ("dataset/examples",):
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

    # 2) Pick preferred or random test image
    image_path = PREFERRED_IMAGE if os.path.exists(PREFERRED_IMAGE) else pick_random_image()
    print(f"Selected test image: {image_path}")

    # 3) Optional: run a detection to list labels
    detector = ObjectDetector()
    detections_preview = detector.detect_objects(image_path, conf_threshold=0.8, verbose=False)
    class_names = [d.get("class_name", f"Class_{d.get('class_id')}") for d in detections_preview]
    counts = Counter(class_names)
    print("\nDetected labels (conf>=0.8):", len(detections_preview))
    if counts:
        print(" - labels:")
        for name, cnt in counts.most_common():
            print(f"    {name}: {cnt}")

    # 4) Run analysis
    result = pipeline.detect_and_analyze(image_path, conf_threshold=0.8, verbose=True)

    if result.get("success", False):
        scores = result.get("image_level_scores", {})
        overall = float(scores.get("formal") or 0.0)
        meta = result.get("meta", {})
        total = meta.get("total_detections")
        contributing = meta.get("contributing_detections")

        print("\nFinal formal image score:", f"{overall:.3f}")
        print("Detections:", total, "/ contributing:", contributing)

        # 5) visualize (detections not attached by default)
        FormalVisualizer().visualize(image_path, [], overall_score=overall)
    else:
        print(f"Test failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    test_formal_pipeline_visual()


