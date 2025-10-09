#!/usr/bin/env python3
"""
Visualize (x, y, S) 3D features of a detected region.

Pipeline:
- Detect objects with YOLOS (conf threshold)
- Pick one bbox (by index)
- Center-crop region (same intent as colorful pipeline)
- Build 3D features [x_norm, y_norm, S_norm]
- Plot 3D scatter colored by S
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.detectors.object_detector import ObjectDetector
from src.core.color_processing import ColorProcessor


def extract_region_rgb(image_rgb: np.ndarray, bbox: Tuple[float, float, float, float], center_crop_ratio: float = 0.6) -> np.ndarray:
    processor = ColorProcessor()
    region = processor.extract_region_colors(image_rgb, bbox)
    try:
        crop_ratio = float(center_crop_ratio)
        if crop_ratio < 1.0:
            h, w = region.shape[:2]
            crop_ratio = max(0.1, max(0.0, crop_ratio))
            ch = int(round(h * crop_ratio))
            cw = int(round(w * crop_ratio))
            y1 = (h - ch) // 2
            x1 = (w - cw) // 2
            region = region[y1:y1+ch, x1:x1+cw]
    except Exception:
        pass
    return region


def build_xyS(region_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = region_rgb.shape[:2]
    hsv = cv2.cvtColor(region_rgb, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1].astype(np.float32) / 255.0  # [0,1]
    y_norm = np.linspace(0.0, 1.0, h, dtype=np.float32)
    x_norm = np.linspace(0.0, 1.0, w, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_norm, y_norm)
    return x_grid.reshape(-1), y_grid.reshape(-1), s.reshape(-1)


def subsample(x: np.ndarray, y: np.ndarray, s: np.ndarray, max_points: int = 20000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = x.shape[0]
    if n <= max_points:
        return x, y, s
    idx = np.random.choice(n, size=max_points, replace=False)
    return x[idx], y[idx], s[idx]


def main():
    ap = argparse.ArgumentParser(description="Visualize (x,y,S) features of a detected region")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--conf", type=float, default=0.8, help="Confidence threshold for detection")
    ap.add_argument("--idx", type=int, default=0, help="Detection index to visualize (default: 0)")
    ap.add_argument("--whole", action="store_true", help="Use whole image instead of a detected bbox")
    ap.add_argument("--center", type=float, default=0.6, help="Center crop ratio (0~1, default 0.6)")
    ap.add_argument("--out", type=str, default=None, help="Path to save figure (optional)")
    args = ap.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if args.whole:
        h, w = rgb.shape[:2]
        bbox = (0, 0, w, h)
        class_name = "<full_image>"
    else:
        detector = ObjectDetector()
        if not detector.is_ready():
            raise RuntimeError("Detector is not ready")
        dets = detector.detect_objects(args.image, conf_threshold=args.conf, verbose=True)
        if not dets:
            print("No detections above threshold; using whole image region.")
            h, w = rgb.shape[:2]
            bbox = (0, 0, w, h)
            class_name = "<full_image>"
        else:
            idx = max(0, min(args.idx, len(dets) - 1))
            d = dets[idx]
            bbox = tuple(d.get("bbox", (0, 0, rgb.shape[1], rgb.shape[0])))
            class_name = d.get("class_name", f"Class_{d.get('class_id')}")

    # If whole image requested and center==1.0, skip cropping
    center_ratio = 1.0 if (args.whole and args.center >= 1.0) else args.center
    region = extract_region_rgb(rgb, bbox, center_crop_ratio=center_ratio)
    x, y, s = build_xyS(region)
    x, y, s = subsample(x, y, s, max_points=30000)

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], projection='3d')

    ax0.imshow(region)
    ax0.axis('off')
    ax0.set_title(f"Region (class: {class_name})")

    p = ax1.scatter(x, y, s, c=s, cmap='viridis', s=1, alpha=0.6)
    ax1.set_xlabel('x_norm')
    ax1.set_ylabel('y_norm')
    ax1.set_zlabel('S_norm')
    ax1.set_title('(x, y, S) 3D Scatter')
    fig.colorbar(p, ax=ax1, fraction=0.03, pad=0.05, label='S_norm')

    plt.tight_layout()
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        plt.savefig(args.out, dpi=200, bbox_inches='tight')
        print(f"Saved figure to: {args.out}")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()


