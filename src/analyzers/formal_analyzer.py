"""
Formal 분석 모듈
감지된 객체 라벨을 기반으로 포멀/캐주얼 점수를 산출 (-1/0/1)
"""

from typing import List, Dict, Any

from ..core.formal_processing import score_detection_formality


class FormalAnalyzer:
    """Formal 분석기 (포멀/캐주얼 분석)"""

    def __init__(self):
        """분석기 초기화"""
        pass

    def analyze_detections(self, detections: List[Dict[str, Any]], verbose: bool = True) -> List[Dict[str, Any]]:
        """
        감지된 객체들의 포멀/캐주얼 점수를 분석

        Args:
            detections: 감지된 객체들의 리스트
            verbose: 상세 출력 여부

        Returns:
            포멀 점수가 포함된 객체 리스트
        """
        if not detections:
            if verbose:
                print("⚠️ 분석할 감지 결과가 없습니다.")
            return []

        analyzed_results: List[Dict[str, Any]] = []

        for i, detection in enumerate(detections):
            enriched = score_detection_formality(detection)
            enriched.setdefault("region_id", i)
            if verbose:
                print(f"\n📊 Formal region {i+1}:")
                print(f"  class: {enriched.get('class_name')} (ID: {enriched.get('class_id')})")
                print(f"  conf: {enriched.get('confidence', 0.0):.3f}")
                print(f"  formal: {enriched.get('formal_score')} ({enriched.get('formal_label')})")
            analyzed_results.append(enriched)

        return analyzed_results


