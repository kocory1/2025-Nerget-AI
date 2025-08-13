"""
Colorful 분석 파이프라인
YOLO 감지와 색상 분석을 통합하여 colorful 점수를 산출하는 모듈
"""

import os
from typing import Dict, Any

from .base_pipeline import BasePipeline
from ..detectors.object_detector import ObjectDetector
from ..analyzers.colorful_analyzer import ColorfulAnalyzer
from ..processors.result_processor import ResultProcessor
from ..visualizers.image_visualizer import ImageVisualizer


class ColorfulPipeline(BasePipeline):
    """Colorful 분석 파이프라인 (색상 화려함 분석)"""
    
    def __init__(self):
        """파이프라인 초기화"""
        self.detector = ObjectDetector()
        self.analyzer = ColorfulAnalyzer()
        self.processor = ResultProcessor()
        self.visualizer = ImageVisualizer()
    
    def detect_and_analyze(self, image_path: str, conf_threshold: float = 0.8, 
                          verbose: bool = True) -> Dict[str, Any]:
        """
        이미지에서 객체를 감지하고 색상을 분석
        
        Args:
            image_path: 이미지 파일 경로
            conf_threshold: 신뢰도 임계값
            verbose: 상세 출력 여부
            
        Returns:
            분석 결과 딕셔너리
        """
        # 1. 입력 검증
        validation_result = self.processor.validate_inputs(image_path)
        if validation_result:  # 오류가 있는 경우
            return validation_result
        
        # 2. YOLO 분석기 상태 확인
        if not self.detector.is_ready():
            return self.processor.create_error_result(
                "YOLO 분석기가 초기화되지 않았습니다.", 
                image_path
            )
        
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
                return self.processor.create_error_result(
                    "감지된 객체가 없습니다.", 
                    image_path
                )
            
            # 4. 색상 분석
            if verbose:
                print("\\n3. 각 영역별 색상 분석...")
            
            analyzed_results = self.analyzer.analyze_detections(
                image_path, 
                detections, 
                verbose=verbose
            )
            
            # 5. 이미지 로드 (결과 처리용)
            image_rgb = self.analyzer.load_image_rgb(image_path)
            
            # 6. 결과 처리 및 집계
            final_result = self.processor.process_analysis_results(
                image_path, 
                image_rgb, 
                detections, 
                analyzed_results, 
                verbose=verbose
            )
            
            return final_result
            
        except Exception as e:
            error_msg = f"분석 중 오류 발생: {e}"
            if verbose:
                print(f"❌ {error_msg}")
                import traceback
                traceback.print_exc()
            return self.processor.create_error_result(error_msg, image_path)
    
    def visualize_results(self, analysis_result: Dict[str, Any]) -> None:
        """분석 결과를 시각화"""
        self.visualizer.visualize_analysis_results(analysis_result)
    
    def is_ready(self) -> bool:
        """파이프라인이 사용 가능한지 확인"""
        return self.detector.is_ready()
