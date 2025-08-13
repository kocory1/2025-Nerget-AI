"""
시스템 설정 파일
DBSCAN 클러스터링 및 색상 분석 파라미터 정의
"""

# DBSCAN 클러스터링 설정
DBSCAN_CONFIG = {
    "eps": 1,                    # 클러스터 간 거리 (채도 차이 기준)
    "min_samples_ratio": 50,     # 전체 픽셀 대비 최소 샘플 비율 (1/50 = 2%)
    "noise_threshold": 0.8       # 노이즈 비율 임계값 (80% 이상 시 경고)
}

# 색상 분석 설정
COLOR_ANALYSIS_CONFIG = {
    "trimmed_mean_ratio": 0.4,   # 절삭평균 비율 (상하위 20%씩 제거)
    "min_cluster_size": 10,      # 절삭평균 적용 최소 클러스터 크기
    "saturation_normalize": True, # 채도 정규화 사용 여부
    "score_range": (-1, 1)       # 채도 점수 범위
}

# 이미지 처리 설정
IMAGE_PROCESSING_CONFIG = {
    "max_region_pixels": 50000,  # 최대 영역 픽셀 수 (메모리 절약)
    "resize_method": "area",     # 리사이즈 방법
    "color_space": "HSV"         # 색상 공간
}

# 객체 감지 설정
DETECTION_CONFIG = {
    "confidence_threshold": 0.8,  # 신뢰도 임계값
    "model_type": "yolos",       # 사용할 모델 (yolo/yolos)
    "device": "auto"             # 디바이스 (auto/cpu/cuda)
}

# 시각화 설정
VISUALIZATION_CONFIG = {
    "figure_size": (15, 10),     # 그림 크기
    "bbox_color": "red",         # 바운딩 박스 색상
    "bbox_thickness": 2,         # 바운딩 박스 두께
    "font_size": 12,            # 폰트 크기
    "show_confidence": True      # 신뢰도 표시 여부
}

# API 설정
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,              # 개발 모드에서 자동 리로드
    "workers": 1,                # 워커 프로세스 수
    "max_file_size": 10 * 1024 * 1024  # 최대 파일 크기 (10MB)
}

def get_dbscan_params(total_pixels: int) -> dict:
    """픽셀 수에 따른 DBSCAN 파라미터 반환"""
    min_samples = max(1, total_pixels // DBSCAN_CONFIG["min_samples_ratio"])
    return {
        "eps": DBSCAN_CONFIG["eps"],
        "min_samples": min_samples
    }

def get_analysis_summary() -> dict:
    """현재 분석 설정 요약 반환"""
    return {
        "dbscan_eps": DBSCAN_CONFIG["eps"],
        "trimmed_mean_ratio": COLOR_ANALYSIS_CONFIG["trimmed_mean_ratio"],
        "confidence_threshold": DETECTION_CONFIG["confidence_threshold"],
        "max_region_pixels": IMAGE_PROCESSING_CONFIG["max_region_pixels"]
    }