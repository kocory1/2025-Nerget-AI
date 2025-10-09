"""
통합 분석 파이프라인
maximal, colorful, formal 3가지 분석을 통합하여 실행하는 모듈
"""

from typing import Dict, Any
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from .base_pipeline import BasePipeline
from .colorful_pipeline import ColorfulPipeline
from .maximal_pipeline import MaximalPipeline
from .formal_pipeline import FormalPipeline
from ..utils.result_schema import build_result_schema, build_image_level_scores
from ..detectors.object_detector import ObjectDetector


class UnifiedPipeline(BasePipeline):
    """통합 분석 파이프라인 (3가지 분석 통합)"""
    
    def __init__(self):
        """파이프라인 초기화"""
        self.detector = ObjectDetector()
        self.colorful_pipeline = ColorfulPipeline()
        self.maximal_pipeline = MaximalPipeline()
        self.formal_pipeline = FormalPipeline()
    
    def detect_and_analyze(self, image_path: str, conf_threshold: float = 0.4, 
                          verbose: bool = True) -> Dict[str, Any]:
        """
        이미지에서 3가지 분석을 모두 수행
        """
        if verbose:
            print("🎯 통합 분석 시작 (single-shot YOLO + 3 analyzers)")
        t_start = perf_counter()

        # 1) YOLO 한 번만 수행 (Colorful의 detector 활용)
        detector = self.detector
        if not detector.is_ready():
            return build_result_schema(
                pipeline_type="unified",
                image_path=image_path,
                image_level_scores=build_image_level_scores(None, None, None),
                success=False,
                error="YOLO 분석기가 준비되지 않았습니다.",
                meta={}
            )
        if verbose:
            print("🔍 YOLO 객체 감지 (single-shot)...")
        t_det0 = perf_counter()
        detections = detector.detect_objects(image_path, conf_threshold=conf_threshold, verbose=verbose)
        t_det1 = perf_counter()
        if verbose:
            print(f"⏱️ detection: {(t_det1 - t_det0)*1000:.1f} ms, boxes={len(detections)}")

        # 병렬로 세 분석 실행 (ThreadPoolExecutor: numpy/sklearn 내부 C코드가 GIL을 대부분 해제)
        if verbose:
            print("⚙️ 분석 병렬 실행 (colorful/maximal/formal)...")

        def timed_call(fn, *args, **kwargs):
            t0 = perf_counter()
            result = fn(*args, **kwargs)
            t1 = perf_counter()
            return result, (t1 - t0)

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                "colorful": executor.submit(
                    timed_call,
                    self.colorful_pipeline.detect_and_analyze,
                    image_path,
                    conf_threshold,
                    False,
                    False,
                    detections,
                ),
                "maximal": executor.submit(
                    timed_call,
                    self.maximal_pipeline.detect_and_analyze,
                    image_path,
                    conf_threshold,
                    False,
                    detections,
                ),
                "formal": executor.submit(
                    timed_call,
                    self.formal_pipeline.detect_and_analyze,
                    image_path,
                    conf_threshold,
                    False,
                    detections,
                ),
            }

            # 수집
            colorful_result, t_col = futures["colorful"].result()
            maximal_result, t_max = futures["maximal"].result()
            formal_result, t_for = futures["formal"].result()

        if verbose:
            print(f"⏱️ colorful: {t_col*1000:.1f} ms, maximal: {t_max*1000:.1f} ms, formal: {t_for*1000:.1f} ms")

        results = {
            "colorful": colorful_result,
            "maximal": maximal_result,
            "formal": formal_result,
        }

        # 3) 통합 결과 생성
        unified_result = self._create_unified_result(results, image_path)
        if verbose:
            total_ms = (perf_counter() - t_start) * 1000
            print(f"✅ 통합 분석 완료 (total {total_ms:.1f} ms)")
        return unified_result

    def _create_unified_result(self, results: Dict[str, Any], image_path: str) -> Dict[str, Any]:
        """개별 분석 결과를 통합하여 최종 결과 생성 (부분 성공 허용)"""
        color_s = results.get("colorful", {}).get("image_level_scores", {})
        max_s = results.get("maximal", {}).get("image_level_scores", {})
        form_s = results.get("formal", {}).get("image_level_scores", {})

        colorful_score = color_s.get("colorful")
        maximal_score = max_s.get("maximal")
        formal_score = form_s.get("formal")

        scores = build_image_level_scores(
            colorful=colorful_score,
            maximal=maximal_score,
            formal=formal_score,
        )

        individuals = {
            "colorful": bool(results.get("colorful", {}).get("success", False)),
            "maximal": bool(results.get("maximal", {}).get("success", False)),
            "formal": bool(results.get("formal", {}).get("success", False)),
        }

        meta = {
            "individual_results": individuals,
        }

        return build_result_schema(
            pipeline_type="unified",
            image_path=image_path,
            image_level_scores=scores,
            success=True,
            meta=meta,
        )

    def visualize_results(self, analysis_result: Dict[str, Any]) -> None:
        """통합 분석 결과 시각화"""
        if not analysis_result.get("success", False):
            print("⚠️ 시각화할 통합 결과가 없습니다.")
            return
        print("🖼️ 통합 결과 시각화...")
        if analysis_result.get("individual_results", {}).get("colorful", False):
            colorful_data = {
                "image_rgb": analysis_result.get("image_rgb"),
                "detections": analysis_result.get("detections", []),
                "success": True
            }
            self.colorful_pipeline.visualize_results(colorful_data)

    def is_ready(self) -> bool:
        """파이프라인이 사용 가능한지 확인"""
        return self.detector.is_ready()
