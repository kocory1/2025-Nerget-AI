#!/usr/bin/env python3
"""
모듈화된 YOLO + DBSCAN 색상 클러스터링 테스트
"""

import sys
import os
import glob
import random

# 프로젝트 루트를 Python 경로에 추가 (tests에서 상위 디렉토리로)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipelines.colorful_pipeline import ColorfulPipeline
from src.detectors.object_detector import ObjectDetector

# 우선 적용할 고정 테스트 이미지 (존재하지 않으면 랜덤 폴백)
PREFERRED_IMAGE = "/Users/bagminsu/Documents/옷마카세_ai/2025-Nerget-AI/dataset/examples/ex3.jpeg"


def pick_random_image() -> str:
    """Pick one random image from dataset (minimal or maximal)."""
    candidates = []
    for d in ("dataset/minimal", "dataset/examples"):
        candidates.extend(glob.glob(os.path.join(d, "*.jpg")))
        candidates.extend(glob.glob(os.path.join(d, "*.jpeg")))
        candidates.extend(glob.glob(os.path.join(d, "*.png")))
    if not candidates:
        raise FileNotFoundError("No images found in dataset directory.")
    random.seed()
    return random.choice(candidates)


def test_yolo_color_modular():
    """모듈화된 YOLO 색상 분석 테스트"""
    
    print("🌈 Colorful 파이프라인 테스트 (YOLO + DBSCAN 색상 분석)")
    print("=" * 70)
    
    # 1. 통합 파이프라인 초기화
    print("1. YOLOS 분석기 초기화 중... (Fashionpedia 모델)")
    pipeline = ColorfulPipeline()
    detector = ObjectDetector()
    
    # 2. 테스트 이미지 분석
    image_path = PREFERRED_IMAGE if os.path.exists(PREFERRED_IMAGE) else pick_random_image()
    print(f"Selected test image: {image_path}")
    
    # 3. 감지 1회 수행 후 Colorful 분석 실행
    detections = detector.detect_objects(image_path, conf_threshold=0.8, verbose=True)
    result = pipeline.detect_and_analyze(
        image_path,
        conf_threshold=0.8,
        verbose=True,
        return_detections=True,
        precomputed_detections=detections,
    )
    
    if result.get("success", False):
        # 4. 시각화
        pipeline.visualize_results(result)
        
        print("🎉 Colorful 파이프라인 테스트 완료!")
    else:
        print(f"❌ 테스트 실패: {result.get('error', '알 수 없는 오류')}")


if __name__ == "__main__":
    test_yolo_color_modular()