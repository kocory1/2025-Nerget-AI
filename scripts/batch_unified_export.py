#!/usr/bin/env python3
"""
Batch run UnifiedPipeline over image directories and export JSON mapping:
  { "<filename>": [colorful, maximal, formal], ... }

Usage:
  python scripts/batch_unified_export.py \
    --dirs dataset/minimal dataset/maximal \
    --out runs/unified_vectors.json \
    --conf 0.8
"""

import os
import sys
import glob
import json
import argparse
from typing import List, Dict

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.pipelines.unified_pipeline import UnifiedPipeline


def find_images(dirs: List[str], patterns: List[str]) -> List[str]:
    images: List[str] = []
    for d in dirs:
        for pat in patterns:
            images.extend(glob.glob(os.path.join(d, pat)))
    # de-duplicate and sort for determinism
    images = sorted(list(dict.fromkeys(images)))
    return images


def scores_to_vector(scores: Dict[str, float]) -> List[float]:
    return [
        float(scores.get("colorful") or 0.0),
        float(scores.get("maximal") or 0.0),
        float(scores.get("formal") or 0.0),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Export unified vectors for images to JSON")
    ap.add_argument("--dirs", nargs="*", default=["dataset/minimal", "dataset/maximal"], help="Image directories")
    ap.add_argument("--out", type=str, default="runs/unified_vectors.json", help="Output JSON path")
    ap.add_argument("--conf", type=float, default=0.8, help="Confidence threshold")
    ap.add_argument("--max", type=int, default=0, help="Max images to process (0=all)")
    args = ap.parse_args()

    patterns = ["*.jpg", "*.jpeg", "*.png"]
    images = find_images(args.dirs, patterns)
    if args.max > 0:
        images = images[: args.max]

    print(f"Found {len(images)} images")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    pipeline = UnifiedPipeline()

    mapping: Dict[str, List[float]] = {}
    for idx, img in enumerate(images, start=1):
        try:
            result = pipeline.detect_and_analyze(img, conf_threshold=args.conf, verbose=False)
            vec = scores_to_vector(result.get("image_level_scores", {}))
            key = os.path.basename(img)
            mapping[key] = vec
        except Exception as e:
            print(f"[WARN] Failed on {img}: {e}")

        if idx % 10 == 0:
            print(f"Processed {idx}/{len(images)}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(mapping)} entries to {args.out}")


if __name__ == "__main__":
    main()


