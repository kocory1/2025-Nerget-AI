"""
결과 스키마 유틸
- 파이프라인 간 표준 반환 형식 생성 및 보조 함수
"""

from typing import Dict, Any, List, Optional


STANDARD_SCORE_KEYS = ("colorful", "maximal", "formal")


def build_image_level_scores(
    colorful: Optional[float] = None,
    maximal: Optional[float] = None,
    formal: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """이미지 레벨 점수 딕셔너리 생성 (키 일관성 보장)."""
    return {
        "colorful": colorful,
        "maximal": maximal,
        "formal": formal,
    }


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


