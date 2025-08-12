"""
Colorful 분석 모듈
감지된 객체 영역의 색상을 분석하여 colorful 점수를 산출하는 모듈
"""

import cv2
import numpy as np
from typing import List, Dict, Any

from ..core.color_processing import analyze_region_with_clustering


class ColorfulAnalyzer:
    """Colorful 분석기 (색상 화려함 분석)"""
    
    def __init__(self):
        """분석기 초기화"""
        pass
    
    def analyze_detections(self, image_path: str, detections: List[Dict[str, Any]], 
                          verbose: bool = True) -> List[Dict[str, Any]]:
        """
        감지된 객체들의 색상을 분석
        
        Args:
            image_path: 이미지 파일 경로
            detections: 감지된 객체들의 리스트
            verbose: 상세 출력 여부
            
        Returns:
            색상 분석 결과가 포함된 객체 리스트
        """
        if not detections:
            if verbose:
                print("⚠️ 분석할 감지 결과가 없습니다.")
            return []
        
        try:
            # 이미지 로드 (색상 분석용)
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if verbose:
                print("🎨 각 영역별 색상 분석 중...")
            
            analyzed_results = []
            
            for i, detection in enumerate(detections):
                bbox = detection['bbox']
                confidence = detection['confidence']
                class_id = detection['class_id']
                class_name = detection.get('class_name', f'Class_{class_id}')
                
                if verbose:
                    print(f"\n📊 영역 {i+1} 분석:")
                    print(f"  클래스: {class_name} (ID: {class_id})")
                    print(f"  신뢰도: {confidence:.3f}")
                    print(f"  바운딩 박스: {bbox}")
                
                # 색상 분석
                color_analysis = analyze_region_with_clustering(
                    image_rgb, bbox, verbose=verbose
                )
                
                if "error" not in color_analysis:
                    result = {
                        'region_id': i,
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': bbox,
                        'score': color_analysis['saturation_score'],
                        'saturation_score': color_analysis['saturation_score'],
                        'max_saturation': color_analysis['largest_cluster_saturation'],
                        'n_clusters': color_analysis['n_clusters'],
                        'n_noise': color_analysis['n_noise'],
                        'labels': color_analysis['labels'],
                        'region_shape': color_analysis['region_shape']
                    }
                    analyzed_results.append(result)
                else:
                    if verbose:
                        print(f"  ⚠️ {color_analysis['error']}")
            
            return analyzed_results
            
        except Exception as e:
            if verbose:
                print(f"❌ 색상 분석 중 오류 발생: {e}")
            return []
    
    def load_image_rgb(self, image_path: str) -> np.ndarray:
        """
        이미지를 RGB 형식으로 로드
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            RGB 이미지 배열
        """
        image = cv2.imread(image_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
