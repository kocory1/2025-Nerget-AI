"""
Maximal 분석 파이프라인
YOLO 감지와 MaximalAnalyzer를 통합하여 미니멀/맥시멀을 판단하는 모듈
"""

from typing import Dict, Any
from typing import List, Optional
from .base_pipeline import BasePipeline
from ..detectors.object_detector import ObjectDetector
from ..analyzers.maximal_analyzer import MaximalAnalyzer
from ..utils.result_schema import build_result_schema, build_image_level_scores, failure_schema
from ..utils.file_utils import validate_image_path


class MaximalPipeline(BasePipeline):
    """Maximal 분석 파이프라인 (맥시멀/미니멀 분석)"""

    def __init__(self, threshold: int = 5):
        self.detector = ObjectDetector()
        self.analyzer = MaximalAnalyzer(threshold=threshold)

    def detect_and_analyze(
        self,
        image_path: str,
        conf_threshold: float = 0.8,
        verbose: bool = True,
        precomputed_detections: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
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
        validation = validate_image_path(image_path)
        if validation:
            return failure_schema("maximal", image_path, validation.get("error", "Invalid input"))

        # YOLO 준비 확인
        if precomputed_detections is None and not self.detector.is_ready():
            return failure_schema("maximal", image_path, "YOLO 분석기가 초기화되지 않았습니다.")

        if verbose:
            from os.path import basename
            print(f"테스트 이미지: {basename(image_path)}")
            print("\n2. 감지 단계")

        # 감지
        if precomputed_detections is not None:
            detections = precomputed_detections
            if verbose:
                print("   - 사전 감지 결과 재사용")
        else:
            if verbose:
                print("   - YOLO 객체 감지 수행")
            detections = self.detector.detect_objects(image_path, conf_threshold=conf_threshold, verbose=verbose)
        if not detections:
            return failure_schema("maximal", image_path, "감지된 객체가 없습니다.")

        # 분석
        if verbose:
            print("\n3. Maximal 분석 중...")
        analysis = self.analyzer.analyze_detections(detections, verbose=verbose)

        # 표준 스키마 구성
        scores = build_image_level_scores(maximal=analysis.get("maximal_score", 0.0))
        meta = {
            "predicted_style": analysis.get("predicted_style"),
            "core_item_count": analysis.get("core_item_count", 0),
            "core_ratio": analysis.get("core_ratio", 0.0),
            "threshold": analysis.get("threshold"),
            "total_detections": len(detections),
        }
        return build_result_schema(
            pipeline_type="maximal",
            image_path=image_path,
            image_level_scores=scores,
            success=True,
            meta=meta,
        )

    def visualize_results(self, analysis_result: Dict[str, Any]) -> None:
        # 간단 텍스트 출력 (상세 시각화는 image_visualizer 확장 시 연동)
        if not analysis_result.get("success"):
            print("시각화할 Maximal 결과가 없습니다.")
            return
        style = analysis_result.get("predicted_style")
        count = analysis_result.get("core_item_count")
        print(f"Maximal 결과: style={style}, core_items={count}")
