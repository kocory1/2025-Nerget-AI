"""
통합 색상 분석기
의류 감지부터 색상 분석까지 전체 파이프라인 제공
"""

import numpy as np
from typing import List, Dict, Optional, Tuple

from .yolos_detector import YOLOSDetector
from ..core.color_processing import extract_saturation_from_bbox
from ..core.clustering import analyze_clothing_colors
from ..config.settings import get_analysis_summary


class ClothingColorAnalyzer:
    """통합 의류 색상 분석기"""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Args:
            model_name: 사용할 YOLOS 모델명 (None이면 기본값)
        """
        if model_name:
            self.detector = YOLOSDetector(model_name)
        else:
            self.detector = YOLOSDetector()
        
        self.analysis_summary = get_analysis_summary()
    
    def analyze_image(self, image_path: str, conf_threshold: Optional[float] = None) -> Dict:
        """
        이미지 전체 분석 수행
        
        Args:
            image_path: 분석할 이미지 경로
            conf_threshold: 객체 감지 신뢰도 임계값
            
        Returns:
            전체 분석 결과
        """
        # 1. 객체 감지
        detections = self.detector.detect_clothing(image_path, conf_threshold)
        
        if not detections:
            return {
                "error": "No clothing items detected",
                "detections": [],
                "analysis_settings": self.analysis_summary
            }
        
        # 2. 이미지 로드
        image_rgb = self.detector.load_image_as_rgb(image_path)
        
        # 3. 각 객체별 색상 분석
        region_results = []
        for i, detection in enumerate(detections):
            try:
                region_result = self._analyze_single_region(
                    image_rgb, detection, region_id=i+1
                )
                region_results.append(region_result)
            except Exception as e:
                print(f"⚠️ 영역 {i+1} 분석 실패: {e}")
                continue
        
        # 4. 전체 결과 요약
        overall_summary = self._calculate_overall_summary(region_results)
        
        return {
            "image_path": image_path,
            "total_detections": len(detections),
            "analyzed_regions": len(region_results),
            "region_results": region_results,
            "overall_summary": overall_summary,
            "analysis_settings": self.analysis_summary
        }
    
    def _analyze_single_region(self, image_rgb: np.ndarray, detection: Dict, region_id: int) -> Dict:
        """단일 영역 색상 분석"""
        # 기본 정보
        class_id = detection["class_id"]
        class_name = detection["class_name"]
        confidence = detection["confidence"]
        bbox = detection["bbox"]
        
        # 채도 추출
        saturation_values, region_info = extract_saturation_from_bbox(image_rgb, bbox)
        
        # 클러스터 분석
        cluster_result = analyze_clothing_colors(saturation_values)
        
        # 결과 구성
        result = {
            "region_id": region_id,
            "detection": {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "bbox": bbox
            },
            "region_info": region_info,
            "color_analysis": cluster_result,
            "saturation_score": cluster_result["saturation_score"]
        }
        
        return result
    
    def _calculate_overall_summary(self, region_results: List[Dict]) -> Dict:
        """전체 분석 결과 요약"""
        if not region_results:
            return {"error": "No regions analyzed"}
        
        # 각 영역별 대표 채도 수집 (클러스터별 절삭평균 중 최대값)
        representative_saturations = []
        for result in region_results:
            color_analysis = result["color_analysis"]
            if "representative_saturation" in color_analysis:
                representative_saturations.append(color_analysis["representative_saturation"])
        
        summary = {
            "total_regions": len(region_results),
            "representative_saturations": representative_saturations  # 각 영역별 대표 채도 리스트
        }
        
        return summary
    
    # _categorize_scores와 _classify_style 메서드는 제거 - 추후 다른 기준과 함께 구현 예정
    
    def get_detector_info(self) -> Dict:
        """감지기 정보 반환"""
        return self.detector.get_model_info()


def analyze_clothing_image(image_path: str, 
                          conf_threshold: Optional[float] = None,
                          model_name: Optional[str] = None) -> Dict:
    """
    의류 이미지 분석 편의 함수
    
    Args:
        image_path: 이미지 경로
        conf_threshold: 신뢰도 임계값
        model_name: 모델명
        
    Returns:
        분석 결과
    """
    analyzer = ClothingColorAnalyzer(model_name)
    return analyzer.analyze_image(image_path, conf_threshold)