"""
Maximal 분석 모듈
YOLOS 감지 결과를 받아 핵심 아이템 개수 기반으로 미니멀/맥시멀을 판단
"""

from typing import List, Dict, Any, Tuple
import math

from ..config.labels import is_core_item, classify_style_by_core_count, compute_maximal_score


class MaximalAnalyzer:
    """Maximal 분석기 (핵심 아이템 카운트 기반)"""

    def __init__(self, threshold: int = 5):
        self.threshold = int(threshold)

    def analyze_detections(self, detections: List[Dict[str, Any]], verbose: bool = True) -> Dict[str, Any]:
        """
        감지 결과를 입력 받아 스타일을 분류
        Args:
            detections: YOLOS 감지 결과 리스트 [{class_id,int, class_name,str, confidence,float, bbox,[...]}]
            verbose: 로깅 여부
        Returns:
            {
              "predicted_style": "minimal"|"maximal",
              "core_item_count": int,
              "core_ratio": float,
              "threshold": int,
              "per_region": [ {region_id, class_id, class_name, is_core_item, confidence, bbox}, ... ]
            }
        """
        # 1) 중복/쌍 처리: 동일 클래스 다중 박스는 1개로 캡 (간단 캡 방식)
        #    신발(23) 등 좌우 쌍으로 감지되는 항목이 과대 카운트되는 것을 방지
        seen_classes = set()
        dedup_labels = []
        for det in detections:
            cid = det.get("class_id")
            if cid in seen_classes:
                continue
            seen_classes.add(cid)
            dedup_labels.append(cid)

        labels = dedup_labels
        style, core_count, core_ratio = classify_style_by_core_count(labels, threshold=self.threshold)
        score = compute_maximal_score(core_count, self.threshold)
        # 소수점 4자리 절삭
        factor = 10 ** 4
        score = math.trunc(score * factor) / float(factor)

        per_region = []
        for i, det in enumerate(detections):
            per_region.append({
                "region_id": i,
                "class_id": det.get("class_id"),
                "class_name": det.get("class_name"),
                "is_core_item": bool(is_core_item(det.get("class_id"))),
                "confidence": det.get("confidence"),
                "bbox": det.get("bbox"),
            })

        if verbose:
            print(f"🔢 핵심 아이템 개수: {core_count} (threshold={self.threshold}) → style={style}")

        return {
            "predicted_style": style,
            "core_item_count": core_count,
            "core_ratio": core_ratio,
            "threshold": self.threshold,
            "maximal_score": score,  # [-1,1]에서 미니멀(-1~0], 맥시멀[0~1]
            "per_region": per_region,
        }


