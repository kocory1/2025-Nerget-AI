"""
색상 처리 모듈
이미지의 색상 변환 및 채도 추출 기능
"""

import cv2
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from scipy.stats import trim_mean
from sklearn.cluster import DBSCAN
from ..config.settings import IMAGE_PROCESSING_CONFIG


class ColorProcessor:
    """색상 처리기"""
    
    def __init__(self):
        self.max_pixels = IMAGE_PROCESSING_CONFIG["max_region_pixels"]
        self.resize_method = IMAGE_PROCESSING_CONFIG["resize_method"]
    
    def extract_region_colors(self, image: np.ndarray, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """
        바운딩 박스 영역에서 색상 정보 추출
        
        Args:
            image: RGB 이미지 (H, W, 3)
            bbox: 바운딩 박스 (x1, y1, x2, y2)
            
        Returns:
            추출된 영역의 이미지 (H', W', 3)
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # 바운딩 박스 클리핑
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # 영역 추출
        region = image[y1:y2, x1:x2]
        
        if region.size == 0:
            raise ValueError("Empty region extracted")
        
        # 크기 조정 (메모리 절약)
        region = self._resize_if_needed(region)
        
        return region
    
    def rgb_to_hsv_saturation(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        RGB 이미지를 HSV로 변환하고 채도(S) 값만 추출
        
        Args:
            rgb_image: RGB 이미지 (H, W, 3)
            
        Returns:
            채도 값 배열 (H*W,)
        """
        # RGB to HSV 변환
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        
        # 채도(S) 채널 추출 (인덱스 1)
        saturation_channel = hsv_image[:, :, 1]
        
        # 1차원 배열로 변환
        saturation_values = saturation_channel.flatten()
        
        return saturation_values
    
    def _resize_if_needed(self, region: np.ndarray) -> np.ndarray:
        """필요시 영역 크기 조정"""
        height, width = region.shape[:2]
        total_pixels = height * width
        
        if total_pixels <= self.max_pixels:
            return region
        
        # 비율 계산하여 리사이즈
        scale_factor = np.sqrt(self.max_pixels / total_pixels)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # 리사이즈 방법에 따라 처리
        if self.resize_method == "area":
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LINEAR
        
        resized_region = cv2.resize(region, (new_width, new_height), interpolation=interpolation)
        
        return resized_region
    
    def get_region_info(self, region: np.ndarray) -> dict:
        """영역 정보 반환"""
        height, width = region.shape[:2]
        total_pixels = height * width
        
        return {
            "dimensions": (height, width),
            "total_pixels": total_pixels,
            "is_resized": total_pixels < region.shape[0] * region.shape[1]
        }


def extract_saturation_from_bbox(image: np.ndarray, bbox: Tuple[float, float, float, float]) -> Tuple[np.ndarray, dict]:
    """
    바운딩 박스에서 채도 값 추출 편의 함수
    
    Args:
        image: RGB 이미지
        bbox: 바운딩 박스 좌표
        
    Returns:
        (채도 값 배열, 영역 정보)
    """
    processor = ColorProcessor()
    
    # 영역 추출
    region = processor.extract_region_colors(image, bbox)
    region_info = processor.get_region_info(region)
    
    # 채도 추출
    saturation_values = processor.rgb_to_hsv_saturation(region)
    
    return saturation_values, region_info


def analyze_region_with_clustering(image: np.ndarray, bbox: Tuple[float, float, float, float], 
                                 eps: float = 1, trim_proportion: float = 0.4, 
                                 verbose: bool = True) -> Dict[str, Any]:
    """
    바운딩 박스 영역의 채도를 DBSCAN 클러스터링으로 분석
    
    Args:
        image: RGB 이미지
        bbox: 바운딩 박스 (x1, y1, x2, y2)
        eps: DBSCAN epsilon 파라미터
        trim_proportion: 절삭평균 비율
        verbose: 상세 출력 여부
        
    Returns:
        분석 결과 딕셔너리
    """
    processor = ColorProcessor()
    
    # 영역 추출
    try:
        region = processor.extract_region_colors(image, bbox)
    except ValueError as e:
        return {"error": str(e)}
    
    if verbose:
        print(f"  🎯 바운딩 박스 영역 분석: {region.shape[0]}x{region.shape[1]} 픽셀")
    
    # RGB → HSV 변환
    hsv_region = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
    saturation_values = hsv_region[:, :, 1].flatten()
    
    if verbose:
        print(f"  🔍 분석할 픽셀 수: {len(saturation_values)}")
    
    # DBSCAN 클러스터링
    total_pixels = len(saturation_values)
    min_samples = total_pixels // 50  # 전체 픽셀의 2%
    
    saturation_data = saturation_values.reshape(-1, 1)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(saturation_data)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    if verbose:
        print(f"  🎯 발견된 클러스터: {n_clusters}개")
        print(f"  📊 노이즈 픽셀: {n_noise}개 ({n_noise/total_pixels*100:.1f}%)")
        print(f"  📏 DBSCAN 파라미터: eps={eps}, min_samples={min_samples}")
    
    # 클러스터 분석
    df_clusters = pd.DataFrame({
        'saturation': saturation_values,
        'cluster': labels
    })
    
    df_filtered = df_clusters[df_clusters['cluster'] != -1].copy()
    
    if len(df_filtered) == 0:
        return {"error": "유효한 클러스터가 없습니다."}
    
    # 클러스터별 크기와 절삭평균 계산
    cluster_sizes = df_filtered['cluster'].value_counts()
    trimmed_mean_saturation_per_cluster = {}
    
    for cluster_id in df_filtered['cluster'].unique():
        cluster_data = df_filtered[df_filtered['cluster'] == cluster_id]['saturation']
        
        if len(cluster_data) >= 10:  # 최소 10개 이상일 때만 절삭평균 적용
            trimmed_mean_sat = trim_mean(cluster_data, trim_proportion)
        else:
            trimmed_mean_sat = cluster_data.mean()
        
        trimmed_mean_saturation_per_cluster[cluster_id] = trimmed_mean_sat
    
    if verbose:
        print(f"  📈 클러스터별 분석 (절삭평균 기준):")
        for cluster_id in sorted(trimmed_mean_saturation_per_cluster.keys()):
            cluster_size = cluster_sizes[cluster_id]
            trimmed_mean_sat = trimmed_mean_saturation_per_cluster[cluster_id]
            print(f"    클러스터 {cluster_id}: 절삭평균 채도 {trimmed_mean_sat:.1f}, 크기 {cluster_size}개")
    
    # 가장 큰 클러스터 선택 (배경 제거)
    largest_cluster_id = cluster_sizes.idxmax()
    largest_cluster_avg_saturation = trimmed_mean_saturation_per_cluster[largest_cluster_id]
    
    if verbose:
        print(f"  🎯 가장 큰 클러스터 {largest_cluster_id} 선택 (크기: {cluster_sizes[largest_cluster_id]}, 절삭평균 채도: {largest_cluster_avg_saturation:.1f})")
    
    # 채도 점수 계산 (가장 큰 클러스터의 절삭평균 채도를 -1~1로 정규화)
    saturation_score = (largest_cluster_avg_saturation / 255.0) * 2 - 1
    
    if verbose:
        print(f"  📊 영역 분석 결과 (가장 큰 클러스터 절삭평균 기준):")
        print(f"    절삭평균 채도: {largest_cluster_avg_saturation:.1f}")
        print(f"    채도 점수: {saturation_score:.3f}")
        print(f"    화려함 점수: {saturation_score:.3f}")
    
    return {
        "region_shape": region.shape,
        "total_pixels": total_pixels,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "cluster_sizes": cluster_sizes.to_dict(),
        "trimmed_means": trimmed_mean_saturation_per_cluster,
        "largest_cluster_id": largest_cluster_id,
        "largest_cluster_size": cluster_sizes[largest_cluster_id],
        "largest_cluster_saturation": largest_cluster_avg_saturation,
        "saturation_score": saturation_score,
        "colorfulness_score": saturation_score,
        "labels": labels
    }