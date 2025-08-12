#!/usr/bin/env python3
"""
모듈화된 YOLO + DBSCAN 색상 클러스터링 테스트
"""

import sys
import os

# 프로젝트 루트를 Python 경로에 추가 (tests에서 상위 디렉토리로)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipelines.colorful_pipeline import ColorfulPipeline


def test_yolo_color_modular():
    """모듈화된 YOLO 색상 분석 테스트"""
    
    print("🌈 Colorful 파이프라인 테스트 (YOLO + DBSCAN 색상 분석)")
    print("=" * 70)
    
    # 1. 통합 파이프라인 초기화
    print("1. YOLOS 분석기 초기화 중... (Fashionpedia 모델)")
    pipeline = ColorfulPipeline()
    
    # 2. 테스트 이미지 분석
    image_path = "dataset/minimal/000b3a87508b0fa185fbd53ecbe2e4c6.jpg"
    
    # 3. 감지 및 분석 실행
    result = pipeline.detect_and_analyze(image_path, conf_threshold=0.8, verbose=True)
    
    if result.get("success", False):
        # 4. 시각화
        pipeline.visualize_results(result)
        
        print("🎉 Colorful 파이프라인 테스트 완료!")
    else:
        print(f"❌ 테스트 실패: {result.get('error', '알 수 없는 오류')}")


if __name__ == "__main__":
    test_yolo_color_modular()