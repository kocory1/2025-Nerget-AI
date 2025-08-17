"""
Maximal 분석 파이프라인
YOLO 감지와 MaximalAnalyzer를 통합하여 미니멀/맥시멀을 판단하는 모듈
"""

from typing import Dict, Any
from .base_pipeline import BasePipeline
from ..detectors.object_detector import ObjectDetector
from ..analyzers.maximal_analyzer import MaximalAnalyzer
from ..processors.result_processor import ResultProcessor


class MaximalPipeline(BasePipeline):
    """Maximal 분석 파이프라인 (맥시멀/미니멀 분석)"""

    def __init__(self, threshold: int = 5):
        self.detector = ObjectDetector()
        self.analyzer = MaximalAnalyzer(threshold=threshold)
        self.processor = ResultProcessor()

    def detect_and_analyze(self, image_path: str, conf_threshold: float = 0.8, verbose: bool = True, **kwargs) -> Dict[str, Any]:
        """
        이미지에서 객체를 감지하고 maximal 분석

        Args:
            image_path: 이미지 파일 경로
            conf_threshold: YOLOS 신뢰도 임계값
            verbose: 로그 출력 여부

        Returns:
            maximal 분석 결과 딕셔너리
        """
        # 입력 검증
        validation = self.processor.validate_inputs(image_path)
        if validation:
            return validation

        # YOLO 준비 확인
        if not self.detector.is_ready():
            return self.processor.create_error_result("YOLO 분석기가 초기화되지 않았습니다.", image_path)

        if verbose:
            from os.path import basename
            print(f"테스트 이미지: {basename(image_path)}")
            print("\n2. YOLO 객체 감지 중...")

        # 감지
        detections = self.detector.detect_objects(image_path, conf_threshold=conf_threshold, verbose=verbose)
        if not detections:
            return self.processor.create_error_result("감지된 객체가 없습니다.", image_path)

        # 분석
        if verbose:
            print("\n3. Maximal 분석 중...")
        analysis = self.analyzer.analyze_detections(detections, verbose=verbose)

        # 결과 구성
        result = {
            "success": True,
            "pipeline_type": "maximal",
            "image_path": image_path,
            "detections": analysis.get("per_region", []),
            "predicted_style": analysis.get("predicted_style"),
            "core_item_count": analysis.get("core_item_count", 0),
            "core_ratio": analysis.get("core_ratio", 0.0),
            "threshold": analysis.get("threshold"),
            "maximal_score": analysis.get("maximal_score", 0.0),
            "total_detections": len(detections),
        }
        return result

    def visualize_results(self, analysis_result: Dict[str, Any]) -> None:
        # 간단 텍스트 출력 (상세 시각화는 image_visualizer 확장 시 연동)
        if not analysis_result.get("success"):
            print("시각화할 Maximal 결과가 없습니다.")
            return
        style = analysis_result.get("predicted_style")
        count = analysis_result.get("core_item_count")
        print(f"Maximal 결과: style={style}, core_items={count}")
