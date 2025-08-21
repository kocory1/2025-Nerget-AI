from __future__ import annotations

import os
import shutil
import tempfile
import time

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.services.model_manager import get_pipeline
from src.config.settings import API_CONFIG


router = APIRouter(prefix="/images", tags=["images"])


@router.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    conf_threshold: float = 0.8,
):
    pipeline = get_pipeline()
    if pipeline is None or not pipeline.is_ready():
        raise HTTPException(status_code=503, detail="모델이 준비되지 않았습니다.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")

    temp_file = None
    try:
        t0 = time.perf_counter()
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_file = tmp.name

        # 파일 크기 제한 확인
        try:
            file_size = os.path.getsize(temp_file)
            if file_size > API_CONFIG.get("max_file_size", 10 * 1024 * 1024):
                raise HTTPException(status_code=413, detail="업로드 파일이 너무 큽니다.")
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail="파일을 처리할 수 없습니다.")

        t_load = time.perf_counter()
        result = pipeline.detect_and_analyze(temp_file, conf_threshold=conf_threshold, verbose=True)
        t_done = time.perf_counter()
        scores = result.get("image_level_scores", {})

        col = float(scores.get("colorful") or 0.0)
        den = float(scores.get("maximal") or 0.0)
        frm = float(scores.get("formal") or 0.0)

        return {
            "filename": file.filename,
            "vector": [col, den, frm],
            "message": "분석 완료",
            "timing_ms": {
                "io_copy": round((t_load - t0) * 1000, 1),
                "pipeline": round((t_done - t_load) * 1000, 1),
                "total": round((t_done - t0) * 1000, 1),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 중 오류가 발생했습니다: {str(e)}")
    finally:
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)


