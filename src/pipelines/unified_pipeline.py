"""
통합 분석 파이프라인
maximal, colorful, formal 3가지 분석을 통합하여 실행하는 모듈
"""

from typing import Dict, Any
from .base_pipeline import BasePipeline
from .colorful_pipeline import ColorfulPipeline
from .maximal_pipeline import MaximalPipeline
from .formal_pipeline import FormalPipeline


class UnifiedPipeline(BasePipeline):
    """통합 분석 파이프라인 (3가지 분석 통합)"""
    
    def __init__(self):
        """파이프라인 초기화"""
        self.colorful_pipeline = ColorfulPipeline()
        self.maximal_pipeline = MaximalPipeline()
        self.formal_pipeline = FormalPipeline()
    
    def detect_and_analyze(self, image_path: str, conf_threshold: float = 0.8, 
                          verbose: bool = True) -> Dict[str, Any]:
        """
        이미지에서 3가지 분석을 모두 수행
        
        Args:
            image_path: 이미지 파일 경로
            conf_threshold: 신뢰도 임계값
            verbose: 상세 출력 여부
            
        Returns:
            통합 분석 결과 딕셔너리
        """
        if verbose:
            print("🎯 통합 분석 시작 (maximal, colorful, formal)")
        
        results = {}
        
        # 1. Colorful 분석 (현재 구현됨)
        if verbose:
            print("🌈 Colorful 분석 중...")
        colorful_result = self.colorful_pipeline.detect_and_analyze(
            image_path, conf_threshold=conf_threshold, verbose=verbose
        )
        results["colorful"] = colorful_result
        
        # 2. Maximal 분석 (TODO)
        if verbose:
            print("🔥 Maximal 분석 중...")
        maximal_result = self.maximal_pipeline.detect_and_analyze(
            image_path, conf_threshold=conf_threshold, verbose=verbose
        )
        results["maximal"] = maximal_result
        
        # 3. Formal 분석 (구현됨: 단순 평균)
        if verbose:
            print("👔 Formal 분석 중...")
        formal_result = self.formal_pipeline.detect_and_analyze(
            image_path, conf_threshold=conf_threshold, verbose=verbose
        )
        results["formal"] = formal_result
        
        # 4. 통합 결과 생성
        unified_result = self._create_unified_result(results, image_path)
        
        if verbose:
            print("✅ 통합 분석 완료")
        
        return unified_result
    
    def _create_unified_result(self, results: Dict[str, Any], image_path: str) -> Dict[str, Any]:
        """개별 분석 결과를 통합하여 최종 결과 생성"""
        # 현재는 colorful만 성공하므로 그 결과를 기반으로 구성
        colorful_result = results["colorful"]
        
        if colorful_result.get("success", False):
            # 성공한 경우 통합 결과 구성
            detections = colorful_result.get("detections", [])
            
            # 각 감지 객체에 3가지 점수 추가 (현재는 colorful만 실제 값)
            unified_detections = []
            for detection in detections:
                unified_detection = detection.copy()
                unified_detection["scores"] = {
                    "colorful_score": detection.get("saturation_score", 0.0),
                    "maximal_score": 0.0,  # TODO: maximal 구현 후 실제 값
                    "formal_score": None
                }
                unified_detections.append(unified_detection)
            
            # Formal per-detection 매핑 (클래스/박스 매칭 단순화: region_id 기준)
            formal_map = {}
            if results["formal"].get("success"):
                for fr in results["formal"].get("detections", []):
                    formal_map[fr.get("region_id")] = fr.get("formal_score", 0.0)

            # region_id를 키로 매핑해 scores.formal_score 채우기
            for ud in unified_detections:
                rid = ud.get("region_id")
                if rid in formal_map:
                    ud["scores"]["formal_score"] = formal_map[rid]
                else:
                    ud["scores"]["formal_score"] = 0.0

            return {
                "image_path": image_path,
                "success": True,
                "pipeline_type": "unified",
                "detections": unified_detections,
                "total_detections": len(unified_detections),
                "individual_results": {
                    "colorful": colorful_result.get("success", False),
                    "maximal": results["maximal"].get("success", False),
                    "formal": results["formal"].get("success", False)
                },
                "overall_scores": {
                    "colorful_score": colorful_result.get("average_score", 0.0),
                    "maximal_score": 0.0,  # TODO: 구현 후 실제 값
                    "formal_score": results["formal"].get("formal_overall_score", 0.0)
                }
            }
        else:
            # 실패한 경우
            return {
                "error": "통합 분석 실패 (colorful 분석 실패)",
                "success": False,
                "pipeline_type": "unified",
                "image_path": image_path,
                "individual_results": {
                    "colorful": False,
                    "maximal": False,
                    "formal": False
                }
            }
    
    def visualize_results(self, analysis_result: Dict[str, Any]) -> None:
        """통합 분석 결과 시각화"""
        if not analysis_result.get("success", False):
            print("⚠️ 시각화할 통합 결과가 없습니다.")
            return
        
        print("🖼️ 통합 결과 시각화...")
        
        # 현재는 colorful 시각화만 실행
        if analysis_result.get("individual_results", {}).get("colorful", False):
            # colorful 결과만 추출해서 시각화
            colorful_data = {
                "image_rgb": analysis_result.get("image_rgb"),
                "detections": analysis_result.get("detections", []),
                "success": True
            }
            self.colorful_pipeline.visualize_results(colorful_data)
        
        # TODO: maximal, formal 시각화 구현 후 추가
    
    def is_ready(self) -> bool:
        """파이프라인이 사용 가능한지 확인"""
        # 현재는 colorful만 체크
        return self.colorful_pipeline.is_ready()
