"""
색상 처리 모듈
- 바운딩 박스 전처리와 채도(S) 기반 통계 계산
- Colorful 파이프라인은 [x, y, S] 3D 특성에 대해 DBSCAN을 수행합니다.
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


def analyze_region_with_clustering(
    image: np.ndarray,
    bbox: Tuple[float, float, float, float],
    eps: float = 0.15,
    trim_proportion: float = 0.2,
    center_crop_ratio: float = 0.6,
    min_samples_ratio: float = 0.01,
    alpha: float = 1.0, # 위치 중요도
    beta: float = 1.5, # 채도 중요도 
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    [x, y, S] 3D 특성 공간에서 DBSCAN으로 군집화하고, 가장 큰 클러스터의 대표 채도로
    화려함 점수를 산출합니다.

    처리 단계(요약):
      1) 박스 중심부 크롭(center_crop_ratio) → 배경/경계 영향 감소
      2) HSV S 채널 추출 → 좌표 정규화(x/W, y/H)와 함께 3D 특성 [α·x, α·y, β·S] 구성
      3) DBSCAN(eps, min_samples)으로 3D 공간에서 군집화
      4) 각 클러스터 채도 분포의 절삭평균(또는 평균) 계산(trim_proportion)
      5) 가장 큰 클러스터의 대표 채도를 [-1, 1] 점수로 변환

    파라미터 가이드:
      - eps: 군집 반경. ↓ 세분화, ↑ 응집
      - min_samples_ratio: 총 픽셀 대비 최소 이웃 비율. ↓ 군집 형성 쉬움
      - alpha(α): 위치 가중치. ↑ 공간 거리 반영↑ (가까운 픽셀만 묶임)
      - beta(β): 채도 가중치. ↑ 채도 유사성 반영↑

    Args:
        image: RGB 이미지
        bbox: 바운딩 박스 (x1, y1, x2, y2)
        eps: DBSCAN epsilon(반경)
        trim_proportion: 대표 채도 계산 시 절삭 비율(0.0~0.5)
        center_crop_ratio: 박스 중심부 사용 비율(0~1)
        min_samples_ratio: min_samples = round(N * 비율)
        alpha: 위치(x,y) 축 가중치
        beta: 채도(S) 축 가중치
        verbose: 로그 출력 여부

    Returns:
        분석 결과 딕셔너리(대표 채도, 점수, 클러스터 요약 포함)
    """
    processor = ColorProcessor()
    
    # 영역 추출
    try:
        region = processor.extract_region_colors(image, bbox)
    except ValueError as e:
        return {"error": str(e)}
    
    # 가운데 영역만 사용 (center crop)
    # - 의도: 경계/배경 픽셀 영향 감소, 관심 대상의 내부 질감/색에 집중
    try:
        h, w = region.shape[:2]
        crop_ratio = float(center_crop_ratio)
        crop_ratio = max(0.1, min(1.0, crop_ratio))
        ch = int(round(h * crop_ratio))
        cw = int(round(w * crop_ratio))
        y1 = (h - ch) // 2
        x1 = (w - cw) // 2
        region = region[y1:y1+ch, x1:x1+cw]
    except Exception:
        pass

    if verbose:
        print(f"  🎯 바운딩 박스(센터 크롭) 영역 분석: {region.shape[0]}x{region.shape[1]} 픽셀")
    
    # RGB → HSV 변환
    # - 채도(S) 채널만 추출
    hsv_region = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
    saturation_map = hsv_region[:, :, 1].astype(np.float32)
    saturation_values = saturation_map.flatten()
    
    if verbose:
        print(f"  🔍 분석할 픽셀 수: {len(saturation_values)}")
    
    # 3D 특징 구성: [alpha*x_norm, alpha*y_norm, beta*S_norm]
    h_reg, w_reg = region.shape[:2]
    y_norm = np.linspace(0.0, 1.0, h_reg, dtype=np.float32)
    x_norm = np.linspace(0.0, 1.0, w_reg, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_norm, y_norm)
    s_norm = saturation_map / 255.0
    features = np.stack([
        alpha * x_grid,
        alpha * y_grid,
        beta * s_norm,
    ], axis=-1).reshape(-1, 3)

    # DBSCAN 클러스터링 (3D 특징 공간)
    # - min_samples: 영역 크기에 따라 비율로 산정(하한/상한 캡)
    total_pixels = len(saturation_values)
    ratio = float(min_samples_ratio)
    min_samples = int(max(5, min(5000, round(total_pixels * ratio))))

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = dbscan.fit_predict(features)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    if verbose:
        print(f"  🎯 발견된 클러스터: {n_clusters}개")
        print(f"  📊 노이즈 픽셀: {n_noise}개 ({n_noise/total_pixels*100:.1f}%)")
        print(f"  📏 DBSCAN 파라미터: eps={eps}, min_samples={min_samples}")
    
    # 클러스터 분석 (3D 특징 기반 결과)
    df_clusters = pd.DataFrame({
        'saturation': saturation_values,
        'cluster': labels
    })

    df_filtered = df_clusters[df_clusters['cluster'] != -1].copy()

    if len(df_filtered) == 0:
        return {"error": "유효한 클러스터가 없습니다."}

    cluster_sizes = df_filtered['cluster'].value_counts()
    trimmed_mean_saturation_per_cluster: Dict[int, float] = {}

    for cluster_id in df_filtered['cluster'].unique():
        cluster_data = df_filtered[df_filtered['cluster'] == cluster_id]['saturation']
        if len(cluster_data) >= 10 and trim_proportion > 0.0:
            trimmed_mean_sat = trim_mean(cluster_data, trim_proportion)
        else:
            trimmed_mean_sat = cluster_data.mean()
        trimmed_mean_saturation_per_cluster[int(cluster_id)] = float(trimmed_mean_sat)

    # 가장 큰 클러스터 선택
    largest_cluster_id = int(cluster_sizes.idxmax())
    largest_cluster_avg_saturation = float(trimmed_mean_saturation_per_cluster[largest_cluster_id])

    if verbose:
        desc = "절삭평균" if trim_proportion > 0.0 else "평균"
        print(f"  📈 클러스터별 분석 ({desc} 기준):")
        for cid in sorted(trimmed_mean_saturation_per_cluster.keys()):
            print(f"    클러스터 {cid}: {trimmed_mean_saturation_per_cluster[cid]:.1f}, 크기 {int(cluster_sizes[cid])}")
    
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
        "cluster_sizes": {int(k): int(v) for k, v in cluster_sizes.to_dict().items()},
        "trimmed_means": trimmed_mean_saturation_per_cluster,
        "largest_cluster_id": largest_cluster_id,
        "largest_cluster_size": int(cluster_sizes[largest_cluster_id]),
        "largest_cluster_saturation": largest_cluster_avg_saturation,
        "saturation_score": saturation_score,
        "colorfulness_score": saturation_score,
        "labels": labels
    }


def analyze_region_center_max(
    image: np.ndarray,
    bbox: Tuple[float, float, float, float],
    center_crop_ratio: float = 0.6,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    바운딩 박스의 중심 영역만 사용하여 최대 채도를 기반으로 점수 산출.

    Args:
        image: RGB 이미지
        bbox: (x1, y1, x2, y2)
        center_crop_ratio: 가운데 사용할 비율 (0~1)
        verbose: 로그 출력 여부

    Returns:
        분석 결과 딕셔너리 (클러스터링 관련 필드는 비워짐)
    """
    processor = ColorProcessor()

    # 영역 추출
    try:
        region = processor.extract_region_colors(image, bbox)
    except ValueError as e:
        return {"error": str(e)}

    # 센터 크롭
    try:
        h, w = region.shape[:2]
        crop_ratio = float(center_crop_ratio)
        crop_ratio = max(0.1, min(1.0, crop_ratio))
        ch = int(round(h * crop_ratio))
        cw = int(round(w * crop_ratio))
        y1 = (h - ch) // 2
        x1 = (w - cw) // 2
        region = region[y1:y1+ch, x1:x1+cw]
    except Exception:
        pass

    if verbose:
        print(f"  🎯 바운딩 박스(센터 크롭) 영역 분석: {region.shape[0]}x{region.shape[1]} 픽셀")

    # HSV 변환 및 채도 최대값
    hsv_region = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
    saturation_values = hsv_region[:, :, 1].astype(np.float32)
    total_pixels = int(saturation_values.size)
    if total_pixels == 0:
        return {"error": "Empty center region"}

    max_saturation = float(np.max(saturation_values))
    saturation_score = (max_saturation / 255.0) * 2 - 1

    if verbose:
        print(f"  📊 영역 분석 결과 (센터 최대 채도 기준):")
        print(f"    최대 채도: {max_saturation:.1f}")
        print(f"    채도 점수: {saturation_score:.3f}")

    return {
        "region_shape": region.shape,
        "total_pixels": total_pixels,
        "n_clusters": 0,
        "n_noise": 0,
        "cluster_sizes": {},
        "trimmed_means": {},
        "largest_cluster_id": None,
        "largest_cluster_size": None,
        "largest_cluster_saturation": max_saturation,
        "saturation_score": saturation_score,
        "colorfulness_score": saturation_score,
        "labels": None,
    }


def analyze_region_center_mean(
    image: np.ndarray,
    bbox: Tuple[float, float, float, float],
    center_crop_ratio: float = 0.6,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    바운딩 박스의 중심 영역만 사용하여 평균 채도를 기반으로 점수 산출.
    기존 파이프라인과의 호환성을 위해 'largest_cluster_saturation' 키에 평균을 매핑.
    """
    processor = ColorProcessor()

    # 영역 추출
    try:
        region = processor.extract_region_colors(image, bbox)
    except ValueError as e:
        return {"error": str(e)}

    # 센터 크롭
    try:
        h, w = region.shape[:2]
        crop_ratio = float(center_crop_ratio)
        crop_ratio = max(0.1, min(1.0, crop_ratio))
        ch = int(round(h * crop_ratio))
        cw = int(round(w * crop_ratio))
        y1 = (h - ch) // 2
        x1 = (w - cw) // 2
        region = region[y1:y1+ch, x1:x1+cw]
    except Exception:
        pass

    if verbose:
        print(f"  🎯 바운딩 박스(센터 크롭) 영역 분석: {region.shape[0]}x{region.shape[1]} 픽셀")

    # HSV 변환 및 채도 평균값
    hsv_region = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
    saturation_values = hsv_region[:, :, 1].astype(np.float32)
    total_pixels = int(saturation_values.size)
    if total_pixels == 0:
        return {"error": "Empty center region"}

    mean_saturation = float(np.mean(saturation_values))
    saturation_score = (mean_saturation / 255.0) * 2 - 1

    if verbose:
        print(f"  📊 영역 분석 결과 (센터 평균 채도 기준):")
        print(f"    평균 채도: {mean_saturation:.1f}")
        print(f"    채도 점수: {saturation_score:.3f}")

    return {
        "region_shape": region.shape,
        "total_pixels": total_pixels,
        "n_clusters": 0,
        "n_noise": 0,
        "cluster_sizes": {},
        "trimmed_means": {},
        "largest_cluster_id": None,
        "largest_cluster_size": None,
        "largest_cluster_saturation": mean_saturation,
        "saturation_score": saturation_score,
        "colorfulness_score": saturation_score,
        "labels": None,
    }