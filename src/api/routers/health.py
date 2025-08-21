from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.services.model_manager import is_pipeline_ready
from src.db.mysql import ping


router = APIRouter(prefix="", tags=["health"])


@router.get("/health")
async def health_check():
    yolo_status = "available" if is_pipeline_ready() else "unavailable"
    return {
        "status": "healthy",
        "message": "서버가 정상적으로 실행 중입니다.",
        "yolo_model": yolo_status,
    }


@router.get("/db/health")
async def db_health():
    try:
        ok = await ping()
        if not ok:
            raise RuntimeError("ping failed")
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB health check failed: {e}")


