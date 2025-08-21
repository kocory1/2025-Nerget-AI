from __future__ import annotations

from typing import Optional

from src.pipelines.unified_pipeline import UnifiedPipeline


_pipeline: Optional[UnifiedPipeline] = None


def get_pipeline() -> UnifiedPipeline:
    global _pipeline
    if _pipeline is None:
        try:
            _pipeline = UnifiedPipeline()
            print("UnifiedPipeline 초기화 완료 (services.model_manager)")
        except Exception as e:
            _pipeline = None
            print(f"UnifiedPipeline 초기화 실패: {e}")
            raise
    return _pipeline


def is_pipeline_ready() -> bool:
    try:
        p = get_pipeline()
        return bool(p and p.is_ready())
    except Exception:
        return False


