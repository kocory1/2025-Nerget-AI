#!/usr/bin/env python3
"""
Unified pipeline test
- CLI: --image <path> (optional), --out <path> (optional)
- Run maximal + colorful + formal pipelines together
- Print unified image-level vector [colorful, maximal, formal]
- Save an image with the vector overlaid
"""

import sys
import os
import glob
import random
import argparse
import cv2
import matplotlib.pyplot as plt

# Add project root to Python path (tests run from subdir)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipelines.unified_pipeline import UnifiedPipeline


def pick_random_image() -> str:
	"""Pick one random image from dataset (minimal or maximal)."""
	candidates = []
	for d in ("dataset/minimal", "dataset/maximal"):
		candidates.extend(glob.glob(os.path.join(d, "*.jpg")))
		candidates.extend(glob.glob(os.path.join(d, "*.jpeg")))
		candidates.extend(glob.glob(os.path.join(d, "*.png")))
	if not candidates:
		raise FileNotFoundError("No images found in dataset directory.")
	random.seed()
	return random.choice(candidates)


def scores_to_vector(scores: dict) -> list:
	"""Convert image_level_scores dict to fixed-order vector [colorful, maximal, formal]."""
	return [
		float(scores.get("colorful") or 0.0),
		float(scores.get("maximal") or 0.0),
		float(scores.get("formal") or 0.0),
	]


def show_image_and_vector(image_path: str, vector, save_path: str | None = None) -> None:
	img_bgr = cv2.imread(image_path)
	if img_bgr is None:
		raise FileNotFoundError(f"Failed to read image: {image_path}")
	img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

	fig = plt.figure(figsize=(12, 10))
	gs = fig.add_gridspec(2, 1, height_ratios=[4, 1])
	ax_img = fig.add_subplot(gs[0])
	ax_bar = fig.add_subplot(gs[1])

	ax_img.imshow(img_rgb)
	ax_img.axis("off")
	ax_img.set_title("Unified pipeline result")

	labels = ["colorful", "maximal", "formal"]
	colors = ["#ff7f0e", "#1f77b4", "#2ca02c"]
	ax_bar.bar(labels, vector, color=colors)
	ax_bar.set_ylim(-1.0, 1.0)
	ax_bar.grid(True, axis="y", alpha=0.3)
	ax_bar.set_ylabel("score")

	plt.tight_layout()
	if save_path:
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		plt.savefig(save_path, dpi=200, bbox_inches="tight")
	plt.show()


def parse_args() -> argparse.Namespace:
	ap = argparse.ArgumentParser(description="Unified pipeline visual test (image + vector plot)")
	ap.add_argument("--image", type=str, default=None, help="Path to input image (optional)")
	ap.add_argument("--out", type=str, default=None, help="Path to save the figure (optional)")
	return ap.parse_args()


def main():
	print("Unified pipeline test (maximal + colorful + formal)")
	print("=" * 70)
	args = parse_args()

	# 1) Initialize pipeline
	pipeline = UnifiedPipeline()

	# 2) Select image
	image_path = args.image or pick_random_image()
	print(f"Selected test image: {image_path}")

	# 3) Run unified analysis
	result = pipeline.detect_and_analyze(image_path, conf_threshold=0.8, verbose=True)
	assert isinstance(result, dict), "Unified pipeline should return a dict."

	# 4) Extract vector
	scores = result.get("image_level_scores", {})
	vector = scores_to_vector(scores)
	print("Vector [colorful, maximal, formal]:", vector)

	# 5) Visualize on screen (and optionally save)
	show_image_and_vector(image_path, vector, args.out)

	# 6) Basic sanity checks
	assert len(vector) == 3, "Vector length must be 3."
	for v in vector:
		assert isinstance(v, float), "Vector elements must be floats."

	print("🎉 Unified pipeline vector test completed.")


if __name__ == "__main__":
	main()


