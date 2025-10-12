"""
Colorful 분석 모듈
감지된 박스별 색상 채도를 분석하여 이미지 레벨 colorful 점수 산출의 근거를 제공합니다.
"""

from typing import List, Dict, Any
import numpy as np
from PIL import Image


def _load_image_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def _saturation_from_rgb(rgb: np.ndarray) -> float:
    # 간단한 채도 근사: HSV 변환 없이 RGB 분산 기반 근사 (0~1 범위)
    # 표준화: 채널별 [0,1] 스케일 후 표준편차를 채도로 사용
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return 0.0
    arr = rgb.astype(np.float32) / 255.0
    std = float(arr.reshape(-1, 3).std())
    # 안정화 및 상한
    return max(0.0, min(1.0, std * 1.5))


class ColorfulAnalyzer:
    """Colorful 분석기 (채도 근사 기반)

    실제 프로덕션 알고리즘 대체 전까지 임시 경량 구현입니다.
    """

    def load_image_rgb(self, path: str) -> np.ndarray:
        return _load_image_rgb(path)

    def analyze_detections(
        self,
        image_path: str,
        detections: List[Dict[str, Any]],
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        if not detections:
            return []

        rgb = _load_image_rgb(image_path)
        h, w, _ = rgb.shape
        results: List[Dict[str, Any]] = []
        for i, det in enumerate(detections):
            # bbox = [x1,y1,x2,y2] 가정, 경계 보정
            bbox = det.get("bbox") or [0, 0, w, h]
            x1, y1, x2, y2 = [int(max(0, v)) for v in bbox]
            x1 = min(x1, w - 1)
            x2 = min(max(x2, x1 + 1), w)
            y1 = min(y1, h - 1)
            y2 = min(max(y2, y1 + 1), h)

            crop = rgb[y1:y2, x1:x2]
            s = _saturation_from_rgb(crop)

            enriched = dict(det)
            enriched["region_id"] = i
            enriched["score"] = float(round(s, 4))
            results.append(enriched)

        return results


