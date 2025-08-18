"""
Colorful 분석 파이프라인
YOLO 감지와 색상 분석을 통합하여 colorful 점수를 산출하는 모듈
"""

import os
from typing import Dict, Any

from .base_pipeline import BasePipeline
from ..detectors.object_detector import ObjectDetector
from ..analyzers.colorful_analyzer import ColorfulAnalyzer
from ..visualizers.image_visualizer import ImageVisualizer
from ..utils.file_utils import validate_image_path
from ..utils.result_schema import build_result_schema, build_image_level_scores, failure_schema


class ColorfulPipeline(BasePipeline):
    """Colorful 분석 파이프라인 (색상 화려함 분석)"""
    
    def __init__(self):
        """파이프라인 초기화"""
        self.detector = ObjectDetector()
        self.analyzer = ColorfulAnalyzer()
        self.visualizer = ImageVisualizer()
    
    def detect_and_analyze(self, image_path: str, conf_threshold: float = 0.8, 
                          verbose: bool = True, return_detections: bool = False) -> Dict[str, Any]:
        """
        이미지에서 객체를 감지하고 색상을 분석
        
        Args:
            image_path: 이미지 파일 경로
            conf_threshold: 신뢰도 임계값
            verbose: 상세 출력 여부
            
        Returns:
            분석 결과 딕셔너리
        """
        # 1. 입력 검증 (간단 파일 존재 확인)
        validation = validate_image_path(image_path)
        if validation:
            return failure_schema("colorful", image_path, validation.get("error", "Invalid input"))
        
        # 2. YOLO 분석기 상태 확인
        if not self.detector.is_ready():
            return failure_schema("colorful", image_path, "YOLO 분석기가 초기화되지 않았습니다.")
        
        if verbose:
            print(f"📸 테스트 이미지: {os.path.basename(image_path)}")
        
        try:
            # 3. 객체 감지
            if verbose:
                print("\\n2. YOLO 객체 감지 중...")
            
            detections = self.detector.detect_objects(
                image_path, 
                conf_threshold=conf_threshold, 
                verbose=verbose
            )
            
            if not detections:
                return failure_schema("colorful", image_path, "감지된 객체가 없습니다.")
            
            # 4. 색상 분석
            if verbose:
                print("\\n3. 각 영역별 색상 분석...")
            
            analyzed_results = self.analyzer.analyze_detections(
                image_path, 
                detections, 
                verbose=verbose,
            )
            # 5. 이미지 레벨 점수: 박스 대표 채도 점수 중 최대값 사용
            scores_list = [r.get("score", 0.0) for r in analyzed_results] if analyzed_results else []
            img_score = float(max(scores_list)) if scores_list else 0.0
            scores = build_image_level_scores(colorful=img_score)
            meta = {
                "total_detections": len(detections),
                "analyzed_regions": len(analyzed_results),
                "reduction": "max_over_boxes",
            }
            return build_result_schema(
                pipeline_type="colorful",
                image_path=image_path,
                image_level_scores=scores,
                success=True,
                meta=meta,
                include_detections=return_detections,
                detections=analyzed_results if return_detections else None,
            )
            
        except Exception as e:
            error_msg = f"분석 중 오류 발생: {e}"
            if verbose:
                print(f"❌ {error_msg}")
                import traceback
                traceback.print_exc()
            return failure_schema("colorful", image_path, error_msg)
    
    def visualize_results(self, analysis_result: Dict[str, Any]) -> None:
        """분석 결과를 시각화 (테스트 용)"""
        # 표준 스키마에는 image_rgb가 없을 수 있으므로 재로딩하여 시각화
        if not analysis_result.get("success", False):
            print("⚠️ 시각화할 결과가 없습니다.")
            return
        try:
            image_path = analysis_result.get("image_path")
            image_rgb = self.analyzer.load_image_rgb(image_path)
            tmp = {
                "image_rgb": image_rgb,
                "detections": analysis_result.get("detections", []),
                "success": True,
            }
            self.visualizer.visualize_analysis_results(tmp)
        except Exception as e:
            print(f"❌ 시각화 중 오류 발생: {e}")
    
    def is_ready(self) -> bool:
        """파이프라인이 사용 가능한지 확인"""
        return self.detector.is_ready()
