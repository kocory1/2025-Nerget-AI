"""
DBSCAN 클러스터링 모듈
채도 기반 의류 색상 클러스터링 수행
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.stats import trim_mean
from typing import Dict, List, Tuple, Optional

from ..config.settings import DBSCAN_CONFIG, COLOR_ANALYSIS_CONFIG


class ColorClusterAnalyzer:
    """색상 클러스터 분석기"""
    
    def __init__(self, eps: Optional[float] = None, min_samples_ratio: Optional[int] = None):
        """
        Args:
            eps: DBSCAN eps 파라미터 (None이면 설정값 사용)
            min_samples_ratio: 최소 샘플 비율 (None이면 설정값 사용)
        """
        self.eps = eps or DBSCAN_CONFIG["eps"]
        self.min_samples_ratio = min_samples_ratio or DBSCAN_CONFIG["min_samples_ratio"]
        self.trimmed_mean_ratio = COLOR_ANALYSIS_CONFIG["trimmed_mean_ratio"]
        self.min_cluster_size = COLOR_ANALYSIS_CONFIG["min_cluster_size"]
    
    def analyze_region_colors(self, saturation_values: np.ndarray) -> Dict:
        """
        영역의 채도 값들을 분석하여 클러스터링 수행
        
        Args:
            saturation_values: 채도 값 배열 (0-255)
            
        Returns:
            분석 결과 딕셔너리
        """
        total_pixels = len(saturation_values)
        
        # DBSCAN 파라미터 계산
        min_samples = max(1, total_pixels // self.min_samples_ratio)
        
        # 클러스터링 수행
        saturation_data = saturation_values.reshape(-1, 1)
        dbscan = DBSCAN(eps=self.eps, min_samples=min_samples)
        dbscan.fit(saturation_data)
        labels = dbscan.labels_
        
        # 클러스터 통계
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        noise_ratio = n_noise / total_pixels
        
        # 클러스터별 분석
        cluster_analysis = self._analyze_clusters(saturation_values, labels)
        
        # 클러스터별 절삭평균 중 최대값 선택
        max_cluster_info = self._select_max_average_cluster(cluster_analysis)
        
        return {
            "total_pixels": total_pixels,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_ratio": noise_ratio,
            "dbscan_params": {"eps": self.eps, "min_samples": min_samples},
            "cluster_analysis": cluster_analysis,
            "max_average_cluster": max_cluster_info,
            "representative_saturation": max_cluster_info.get("trimmed_mean", 0),
            "saturation_score": self._calculate_saturation_score(max_cluster_info.get("trimmed_mean", 0))
        }
    
    def _analyze_clusters(self, saturation_values: np.ndarray, labels: np.ndarray) -> List[Dict]:
        """클러스터별 상세 분석"""
        df_clusters = pd.DataFrame({
            'saturation': saturation_values,
            'cluster': labels
        })
        
        # 노이즈 제외
        df_filtered = df_clusters[df_clusters['cluster'] != -1].copy()
        
        if len(df_filtered) == 0:
            return []
        
        cluster_analysis = []
        cluster_sizes = df_filtered['cluster'].value_counts()
        
        for cluster_id in sorted(df_filtered['cluster'].unique()):
            cluster_data = df_filtered[df_filtered['cluster'] == cluster_id]['saturation']
            cluster_size = len(cluster_data)
            
            # 절삭평균 계산
            if cluster_size >= self.min_cluster_size:
                trimmed_mean_sat = trim_mean(cluster_data, self.trimmed_mean_ratio)
            else:
                trimmed_mean_sat = cluster_data.mean()
            
            cluster_analysis.append({
                "cluster_id": cluster_id,
                "size": cluster_size,
                "trimmed_mean": trimmed_mean_sat,
                "min_saturation": cluster_data.min(),
                "max_saturation": cluster_data.max(),
                "std_saturation": cluster_data.std()
            })
        
        return cluster_analysis
    
    def _select_max_average_cluster(self, cluster_analysis: List[Dict]) -> Dict:
        """클러스터별 절삭평균 중 최대값을 가진 클러스터 선택"""
        if not cluster_analysis:
            return {"error": "No valid clusters found"}
        
        # 절삭평균이 가장 높은 클러스터 찾기
        max_cluster = max(cluster_analysis, key=lambda x: x["trimmed_mean"])
        
        return {
            "cluster_id": max_cluster["cluster_id"],
            "size": max_cluster["size"],
            "trimmed_mean": max_cluster["trimmed_mean"],
            "min_saturation": max_cluster["min_saturation"],
            "max_saturation": max_cluster["max_saturation"],
            "std_saturation": max_cluster["std_saturation"]
        }
    
    def _calculate_saturation_score(self, max_saturation: float) -> float:
        """채도 점수 계산 (-1 ~ 1)"""
        return (max_saturation / 255.0) * 2 - 1


def analyze_clothing_colors(saturation_values: np.ndarray, 
                          eps: Optional[float] = None) -> Dict:
    """
    의류 색상 분석 편의 함수
    
    Args:
        saturation_values: 채도 값 배열
        eps: DBSCAN eps 파라미터 (선택사항)
        
    Returns:
        분석 결과
    """
    analyzer = ColorClusterAnalyzer(eps=eps)
    return analyzer.analyze_region_colors(saturation_values)