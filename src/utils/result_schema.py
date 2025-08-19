"""
결과 스키마 유틸
- 파이프라인 간 표준 반환 형식 생성 및 보조 함수
"""

from typing import Dict, Any, List, Optional
import math


STANDARD_SCORE_KEYS = ("colorful", "maximal", "formal")


def _truncate_float(value: Optional[float], decimals: int = 4) -> Optional[float]:
    """부동소수점을 소수점 N자리에서 절삭(0쪽으로)합니다. None은 그대로 반환합니다."""
    if value is None:
        return None
    factor = 10 ** decimals
    return math.trunc(float(value) * factor) / float(factor)


def build_image_level_scores(
    colorful: Optional[float] = None,
    maximal: Optional[float] = None,
    formal: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """이미지 레벨 점수 딕셔너리 생성 (키 일관성 보장, 소수점 4자리 절삭)."""
    return {
        "colorful": _truncate_float(colorful, 4),
        "maximal": _truncate_float(maximal, 4),
        "formal": _truncate_float(formal, 4),
    }


def scores_dict_to_vector(scores: Dict[str, Optional[float]]) -> List[float]:
    """이미지 레벨 점수 딕셔너리를 [colorful, maximal, formal] 순서의 리스트로 변환."""
    return [float(scores.get(k) or 0.0) for k in STANDARD_SCORE_KEYS]

def vector_to_scores_dict(vec: List[float]) -> Dict[str, float]:
    """[colorful, maximal, formal] 벡터를 딕셔너리로 변환."""
    return {k: float(vec[i]) for i, k in enumerate(STANDARD_SCORE_KEYS)}


def build_result_schema(
    pipeline_type: str,
    image_path: str,
    image_level_scores: Dict[str, Optional[float]],
    success: bool = True,
    error: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
    *,
    include_detections: bool = False,
    detections: Optional[List[Dict[str, Any]]] = None,
    **extras: Any,
) -> Dict[str, Any]:
    """표준 결과 스키마 생성.

    반환 형식:
    {
      success: bool,
      pipeline_type: str,
      image_path: str,
      detections: [...],
      image_level_scores: {colorful, maximal, formal},
      meta: {...},
      error?: str,
      ...extras
    }
    """
    meta = meta or {}
    # 점수 키 일관성 보장
    for k in STANDARD_SCORE_KEYS:
        image_level_scores.setdefault(k, None)

    result: Dict[str, Any] = {
        "success": bool(success),
        "pipeline_type": pipeline_type,
        "image_path": image_path,
        "image_level_scores": image_level_scores,
        "meta": meta,
    }
    if include_detections:
        result["detections"] = detections or []
    if error is not None:
        result["error"] = str(error)

    # 추가 필드(예: image_rgb 등) 필요 시 extras로 전달
    result.update(extras)
    return result


def failure_schema(pipeline_type: str, image_path: str, error: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """실패 스키마 생성."""
    return build_result_schema(
        pipeline_type=pipeline_type,
        image_path=image_path,
        image_level_scores=build_image_level_scores(None, None, None),
        success=False,
        error=error,
        meta=meta or {},
    )


