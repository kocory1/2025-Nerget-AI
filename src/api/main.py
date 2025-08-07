from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import sys
import tempfile
import shutil

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# YOLO 색상 클러스터링 모듈 import
try:
    from yolo_color_clustering import YOLOColorClustering
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLO 모듈을 불러올 수 없습니다. ultralytics 패키지가 설치되어 있는지 확인하세요.")

app = FastAPI(
    title="너겟 AI API",
    description="한이음 너겟 AI 프로젝트 API - YOLO 기반 의류 감지 및 색상 분석",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한하세요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# YOLO 분석기 초기화 (전역 변수로)
yolo_analyzer = None
if YOLO_AVAILABLE:
    try:
        # 학습된 모델 경로 확인
        model_path = "runs/detect/train6/weights/best.pt"
        if os.path.exists(model_path):
            yolo_analyzer = YOLOColorClustering(model_path)
            print("YOLO 모델이 성공적으로 로드되었습니다.")
        else:
            print(f"Warning: YOLO 모델 파일을 찾을 수 없습니다: {model_path}")
    except Exception as e:
        print(f"YOLO 모델 로드 중 오류: {e}")

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {"message": "너겟 AI API에 오신 것을 환영합니다!"}

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    yolo_status = "available" if yolo_analyzer else "unavailable"
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
    
    # 파일 저장 로직 (필요시 구현)
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": file.size,
        "message": "이미지가 성공적으로 업로드되었습니다."
    }

@app.post("/analyze-clothing")
async def analyze_clothing(
    file: UploadFile = File(...),
    conf_threshold: float = 0.5,
    n_colors: int = 5
):
    """
    의류 감지 및 색상 분석 엔드포인트 (K-means 기반)
    
    Args:
        file: 업로드된 이미지 파일
        conf_threshold: YOLO 신뢰도 임계값 (0.0 ~ 1.0)
        n_colors: 클러스터링할 색상 개수
    """
    if not YOLO_AVAILABLE or not yolo_analyzer:
        raise HTTPException(
            status_code=503, 
            detail="YOLO 모델을 사용할 수 없습니다. 모델이 로드되지 않았습니다."
        )
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
    
    # 임시 파일로 저장
    temp_file = None
    try:
        # 임시 파일 생성
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_file = tmp.name
        
        # YOLO 분석 수행 (K-means 기반)
        results = yolo_analyzer.analyze_image(
            temp_file, 
            conf_threshold=conf_threshold,
            n_colors=n_colors
        )
        
        # 결과 정리 (numpy 배열을 리스트로 변환)
        cleaned_results = {
            "detected_regions": results["detected_regions"],
            "overall_style": results["overall_style"],
            "analysis_results": []
        }
        
        for result in results["analysis_results"]:
            cleaned_result = {
                "region_id": result["region_id"],
                "bbox": result["bbox"],
                "confidence": result["confidence"],
                "showy_ratio": result["showy_ratio"],
                "style": result["style"],
                "colors": result["colors"],
                "color_ratios": result["color_ratios"]
            }
            cleaned_results["analysis_results"].append(cleaned_result)
        
        return {
            "filename": file.filename,
            "analysis": cleaned_results,
            "message": "의류 분석이 완료되었습니다. (K-means 기반)"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 중 오류가 발생했습니다: {str(e)}")
    
    finally:
        # 임시 파일 정리
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)

@app.post("/analyze-clothing-dbscan")
async def analyze_clothing_dbscan(
    file: UploadFile = File(...),
    conf_threshold: float = 0.5,
    eps: float = 30.0,
    min_samples: int = 10
):
    """
    의류 감지 및 색상 분석 엔드포인트 (DBSCAN 기반)
    
    Args:
        file: 업로드된 이미지 파일
        conf_threshold: YOLO 신뢰도 임계값 (0.0 ~ 1.0)
        eps: DBSCAN의 이웃 반경
        min_samples: DBSCAN의 최소 샘플 수
    """
    if not YOLO_AVAILABLE or not yolo_analyzer:
        raise HTTPException(
            status_code=503, 
            detail="YOLO 모델을 사용할 수 없습니다. 모델이 로드되지 않았습니다."
        )
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
    
    # 임시 파일로 저장
    temp_file = None
    try:
        # 임시 파일 생성
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_file = tmp.name
        
        # YOLO 분석 수행 (DBSCAN 기반)
        results = yolo_analyzer.analyze_image_dbscan(
            temp_file, 
            conf_threshold=conf_threshold,
            eps=eps,
            min_samples=min_samples
        )
        
        # 결과 정리 (numpy 배열을 리스트로 변환)
        cleaned_results = {
            "detected_regions": results["detected_regions"],
            "overall_style": results["overall_style"],
            "overall_score": results["overall_score"],
            "analysis_results": []
        }
        
        for result in results["analysis_results"]:
            cleaned_result = {
                "region_id": result["region_id"],
                "bbox": result["bbox"],
                "confidence": result["confidence"],
                "style": result["style"],
                "normalized_score": result["normalized_score"],
                "clusters": result["clusters"],
                "analysis": result["analysis"]
            }
            cleaned_results["analysis_results"].append(cleaned_result)
        
        return {
            "filename": file.filename,
            "analysis": cleaned_results,
            "message": "의류 분석이 완료되었습니다. (DBSCAN 기반)"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 중 오류가 발생했습니다: {str(e)}")
    
    finally:
        # 임시 파일 정리
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)

@app.post("/detect-only")
async def detect_only(file: UploadFile = File(...), conf_threshold: float = 0.5):
    """
    의류 감지만 수행하는 엔드포인트
    """
    if not YOLO_AVAILABLE or not yolo_analyzer:
        raise HTTPException(
            status_code=503, 
            detail="YOLO 모델을 사용할 수 없습니다."
        )
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
    
    temp_file = None
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_file = tmp.name
        
        detected_boxes = yolo_analyzer.detect_clothing(temp_file, conf_threshold)
        
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