from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys

# 프로젝트 루트를 Python 경로에 추가 (.. / .. / .. = 프로젝트 루트)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.config.settings import API_CONFIG
from src.api.routers.health import router as health_router
from src.api.routers.images import router as images_router

app = FastAPI(
    title="너겟 AI API",
    description="한이음 너겟 AI 프로젝트 API - YOLOS 기반 의류 감지 및 색상 분석",
    version="0.7"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한하세요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {"message": "너겟 AI API에 오신 것을 환영합니다!"}


# Include routers
app.include_router(health_router)
app.include_router(images_router)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # 개발 모드에서 자동 리로드
    ) 