from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import sys
import tempfile
import shutil

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 최신 파이프라인/감지기 사용
from src.pipelines.colorful_pipeline import ColorfulPipeline

app = FastAPI(
    title="너겟 AI API",
    description="한이음 너겟 AI 프로젝트 API - YOLOS 기반 의류 감지 및 색상 분석",
    version="1.1.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한하세요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 최신 Colorful 파이프라인 초기화 (전역)
pipeline: ColorfulPipeline | None = None
try:
    pipeline = ColorfulPipeline()
    print("ColorfulPipeline 초기화 완료")
except Exception as e:
    pipeline = None
    print(f"ColorfulPipeline 초기화 실패: {e}")

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

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """이미지 업로드 엔드포인트"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
    
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "message": "이미지가 성공적으로 업로드되었습니다."
    }

@app.post("/analyze-clothing")
async def analyze_clothing(
    file: UploadFile = File(...),
    conf_threshold: float = 0.8,
):
    """
    의류 감지 및 색상 분석 엔드포인트 (DBSCAN 기반 ColorfulPipeline)
    """
    if pipeline is None or not pipeline.is_ready():
        raise HTTPException(status_code=503, detail="모델이 준비되지 않았습니다.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")

    temp_file = None
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_file = tmp.name

        result = pipeline.detect_and_analyze(temp_file, conf_threshold=conf_threshold, verbose=False)

        return {
            "filename": file.filename,
            "analysis": result,
            "message": "의류 분석이 완료되었습니다. (ColorfulPipeline 기반)"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 중 오류가 발생했습니다: {str(e)}")
    finally:
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)

@app.post("/analyze-clothing-dbscan")
async def analyze_clothing_dbscan(
    file: UploadFile = File(...),
    conf_threshold: float = 0.8,
):
    """
    의류 감지 및 색상 분석 엔드포인트 (DBSCAN 기반 ColorfulPipeline)
    """
    return await analyze_clothing(file=file, conf_threshold=conf_threshold)

@app.post("/detect-only")
async def detect_only(file: UploadFile = File(...), conf_threshold: float = 0.8):
    """의류 감지만 수행하는 엔드포인트"""
    if pipeline is None or not pipeline.is_ready():
        raise HTTPException(status_code=503, detail="모델이 준비되지 않았습니다.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")

    temp_file = None
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_file = tmp.name

        detected_boxes = pipeline.detector.detect_objects(temp_file, conf_threshold=conf_threshold, verbose=False)

        return {
            "filename": file.filename,
            "detected_regions": detected_boxes,
            "count": len(detected_boxes),
            "message": "의류 감지가 완료되었습니다."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"감지 중 오류가 발생했습니다: {str(e)}")
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