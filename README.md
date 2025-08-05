# 너겟 AI 프로젝트

한이음 너겟 AI 프로젝트입니다.

## 설치 및 실행

### 1. 가상환경 활성화
```bash
source venv/bin/activate
```

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. FastAPI 서버 실행
```bash
python main.py
```
또는
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Jupyter 노트북 실행
```bash
jupyter notebook
```

## API 엔드포인트

- `GET /`: 루트 엔드포인트
- `GET /health`: 헬스 체크
- `POST /upload-image`: 이미지 업로드

## API 문서

서버 실행 후 다음 URL에서 API 문서를 확인할 수 있습니다:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 프로젝트 구조

```
2025-Nerget-AI/
├── main.py                 # FastAPI 메인 애플리케이션
├── requirements.txt        # Python 패키지 의존성
├── README.md              # 프로젝트 설명서
├── venv/                  # 가상환경
├── dataset/               # 데이터셋
├── runs/                  # 실행 결과
├── second.ipynb           # Jupyter 노트북
├── second_ml.ipynb        # 머신러닝 노트북
├── classfi_color.py       # 색상 분류 모듈
├── test.py                # 테스트 파일
└── style_labels_balanced_100.csv  # 스타일 라벨 데이터
``` 