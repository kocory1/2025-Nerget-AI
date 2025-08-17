"""
Formal 분석 파이프라인
YOLO 감지와 라벨 기반 점수화로 포멀/캐주얼 스코어 산출 (-1/0/1)
"""

from typing import Dict, Any, List
from .base_pipeline import BasePipeline
from ..detectors.object_detector import ObjectDetector
from ..analyzers.formal_analyzer import FormalAnalyzer
from ..core.formal_processing import analyze_formality_detections
from ..visualizers.formal_visualizer import FormalVisualizer


class FormalPipeline(BasePipeline):
    """Formal 분석 파이프라인 (포멀/캐주얼 분석)"""
    
    def __init__(self):
        """파이프라인 초기화"""
        self.detector = ObjectDetector()
        self.analyzer = FormalAnalyzer()
    
    def detect_and_analyze(self, image_path: str, conf_threshold: float = 0.8, verbose: bool = True) -> Dict[str, Any]:
        """
        이미지에서 객체를 감지하고 formal 분석
        - 신뢰도 0.8 이상 감지만 사용하여 단순 평균(-1/0/1) 산출
        """
        if not self.detector.is_ready():
            return {
                "error": "YOLO 분석기가 초기화되지 않았습니다.",
                "success": False,
                "pipeline_type": "formal",
                "image_path": image_path,
                "detections": []
            }

        try:
            if verbose:
                print("\n🔥 Formal 파이프라인: 감지 시작...")

            detections = self.detector.detect_objects(image_path, conf_threshold=conf_threshold, verbose=verbose)

            # 신뢰도 0.8 이상 필터
            filtered = [d for d in detections if d.get("confidence", 0.0) >= conf_threshold]

            if not filtered:
                return {
                    "success": True,
                    "pipeline_type": "formal",
                    "image_path": image_path,
                    "detections": [],
                    "formal_overall_score": 0.0,
                    "total_detections": len(detections),
                    "contributing_detections": 0,
                    "insufficient_evidence": True
                }

            # Core formal processing (consistency with Colorful core design)
            agg = analyze_formality_detections(filtered, conf_threshold=conf_threshold, verbose=verbose)
            analyzed = agg["analyzed"]
            overall = agg["overall"]

            if verbose:
                print(f"\n📋 Formal 최종 스코어(단순 평균): {overall:.3f}")

            return {
                "success": True,
                "pipeline_type": "formal",
                "image_path": image_path,
                "detections": analyzed,
                "formal_overall_score": overall,
                "total_detections": len(detections),
                "contributing_detections": agg["contributing"],
                "insufficient_evidence": agg["insufficient_evidence"]
            }

        except Exception as e:
            return {
                "error": f"Formal 분석 중 오류 발생: {e}",
                "success": False,
                "pipeline_type": "formal",
                "image_path": image_path,
                "detections": []
            }
    
    def visualize_results(self, analysis_result: Dict[str, Any]) -> None:
        """분석 결과 시각화"""
        if not analysis_result.get("success", False):
            print("No successful formal result to visualize.")
            return
        FormalVisualizer().visualize_from_result(analysis_result)
