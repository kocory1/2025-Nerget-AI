"""
Formal 분석 파이프라인
YOLO 감지와 라벨 기반 점수화로 포멀/캐주얼 스코어 산출 (-1/0/1)
"""

from typing import Dict, Any, List, Optional
from .base_pipeline import BasePipeline
from ..detectors.object_detector import ObjectDetector
from ..analyzers.formal_analyzer import FormalAnalyzer
from ..core.formal_processing import analyze_formality_detections
from ..visualizers.formal_visualizer import FormalVisualizer
from ..utils.result_schema import build_result_schema, build_image_level_scores, failure_schema


class FormalPipeline(BasePipeline):
    """Formal 분석 파이프라인 (포멀/캐주얼 분석)"""
    
    def __init__(self):
        """파이프라인 초기화"""
        self.detector = ObjectDetector()
        self.analyzer = FormalAnalyzer()
    
    def detect_and_analyze(
        self,
        image_path: str,
        conf_threshold: float = 0.8,
        verbose: bool = True,
        precomputed_detections: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        이미지에서 객체를 감지하고 formal 분석
        - 신뢰도 0.8 이상 감지만 사용하여 단순 평균(-1/0/1) 산출
        """
        if precomputed_detections is None and not self.detector.is_ready():
            return failure_schema("formal", image_path, "YOLO 분석기가 초기화되지 않았습니다.")

        try:
            if verbose:
                print("\n🔥 Formal 파이프라인: 감지 시작...")

            if precomputed_detections is not None:
                detections = precomputed_detections
                if verbose:
                    print("   - 사전 감지 결과 재사용")
            else:
                detections = self.detector.detect_objects(image_path, conf_threshold=conf_threshold, verbose=verbose)

            # 신뢰도 0.8 이상 필터
            filtered = [d for d in detections if d.get("confidence", 0.0) >= conf_threshold]

            if not filtered:
                scores = build_image_level_scores(formal=0.0)
                meta = {
                    "total_detections": len(detections),
                    "contributing_detections": 0,
                    "insufficient_evidence": True,
                }
                return build_result_schema(
                    pipeline_type="formal",
                    image_path=image_path,
                    image_level_scores=scores,
                    success=True,
                    meta=meta,
                )

            # Core formal processing (consistency with Colorful core design)
            agg = analyze_formality_detections(filtered, conf_threshold=conf_threshold, verbose=verbose)
            analyzed = agg["analyzed"]
            overall = agg["overall"]

            if verbose:
                print(f"\n📋 Formal 최종 스코어(단순 평균): {overall:.3f}")

            scores = build_image_level_scores(formal=overall)
            meta = {
                "total_detections": len(detections),
                "contributing_detections": agg.get("contributing", 0),
                "insufficient_evidence": agg.get("insufficient_evidence", False),
            }
            return build_result_schema(
                pipeline_type="formal",
                image_path=image_path,
                image_level_scores=scores,
                success=True,
                meta=meta,
            )

        except Exception as e:
            return failure_schema("formal", image_path, f"Formal 분석 중 오류 발생: {e}")
    
    def visualize_results(self, analysis_result: Dict[str, Any]) -> None:
        """분석 결과 시각화"""
        if not analysis_result.get("success", False):
            print("No successful formal result to visualize.")
            return
        FormalVisualizer().visualize_from_result(analysis_result)
