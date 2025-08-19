from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys
import tempfile
import shutil

# 프로젝트 루트를 Python 경로에 추가 (.. / .. / .. = 프로젝트 루트)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 최신 파이프라인/감지기 사용
from src.pipelines.unified_pipeline import UnifiedPipeline
from src.config.settings import API_CONFIG
import time

app = FastAPI(
    title="너겟 AI API",
    description="한이음 너겟 AI 프로젝트 API - YOLOS 기반 의류 감지 및 색상 분석",
    version="0.6"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한하세요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 통합 파이프라인 초기화 (전역)
pipeline: UnifiedPipeline | None = None
try:
    pipeline = UnifiedPipeline()
    print("UnifiedPipeline 초기화 완료")
except Exception as e:
    pipeline = None
    print(f"UnifiedPipeline 초기화 실패: {e}")

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {"message": "너겟 AI API에 오신 것을 환영합니다!"}

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    yolo_status = "available" if (pipeline and pipeline.is_ready()) else "unavailable"
    return {
        "status": "healthy", 
        "message": "서버가 정상적으로 실행 중입니다.",
        "yolo_model": yolo_status
    }


@app.post("/analyze-clothing")
async def analyze_clothing(
    file: UploadFile = File(...),
    conf_threshold: float = 0.8,
):
    """
    통합 파이프라인 기반 분석 엔드포인트
    - 반환: 벡터만
      * vector: [colorful, maximal, formal]
    """
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
                "total": round((t_done - t0) * 1000, 1)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 중 오류가 발생했습니다: {str(e)}")
    finally:
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # 개발 모드에서 자동 리로드
    ) 