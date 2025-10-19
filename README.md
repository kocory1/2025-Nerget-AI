
## ✨ 주요 기능

### 🔍 AI 기반 의류 감지
- **46개 패션 카테고리** 자동 감지 (드레스, 스커트, 가방, 신발 등)
- 신뢰도 임계값 설정 가능(기본 0.4, 설정에서 조정)
- **실시간 처리** 및 배치 분석 지원

### 🎯 3가지 스타일 분류기
- **🌈 Colorful**: 색상의 화려함 분석 (x,y,S 3D DBSCAN + Trimmed Mean, 이미지 점수=박스별 최대, 밝은 클러스터 우선 선택 + p90 폴백)
- **🔥 Maximal**: 맥시멀/미니멀 성향 분석 (핵심 아이템 개수 기반)
- **👔 Formal**: 포멀/캐주얼 격식성 분석 (클래스 중복 제거 합계, tanh 스무딩)

### 🏗️ 엔터프라이즈 아키텍처
- **마이크로서비스** 구조로 개별 모듈 독립 배포 가능
- **RESTful API** 서버 (FastAPI 기반)
- **도커 컨테이너** 지원 (곧 출시)
- **수평 확장** 가능한 설계

## 🏗️ 시스템 아키텍처

```
2025-Nerget-AI/
├── 🧠 src/                          # 핵심 AI 모듈
│   ├── 🚀 pipelines/                # 분석 파이프라인
│   │   ├── colorful_pipeline.py     # 🌈 Colorful 분석 파이프라인 (감지 결과 주입형)
│   │   ├── maximal_pipeline.py      # 🔥 Maximal 분석 파이프라인 
│   │   ├── formal_pipeline.py       # 👔 Formal 분석 파이프라인
│   │   ├── unified_pipeline.py      # 🎯 통합 분석 파이프라인
│   │   └── base_pipeline.py         # 📋 파이프라인 베이스 클래스
│   ├── 🔬 analyzers/                # 분석기 모듈
│   │   ├── colorful_analyzer.py     # 🌈 색상 화려함 분석기
│   │   └── formal_analyzer.py       # 👔 포멀/캐주얼 분석기
│   ├── 🤖 detectors/                # 객체 감지 모듈
│   │   └── object_detector.py       # 🔍 YOLO 객체 감지기
│   ├── ⚙️ utils/                    # 공통 유틸리티
│   │   └── file_utils.py            # 🔎 이미지 경로 검증
│   ├── 🖼️ visualizers/              # 시각화 모듈
│   │   ├── image_visualizer.py      # 🌈 Colorful 결과 시각화
│   │   ├── formal_visualizer.py     # 👔 Formal 결과 시각화
│   │   └── plotting.py              # 공통 플로팅 유틸
│   ├── 🛎️ services/                 # 서비스 레이어
│   │   ├── faiss_index_store.py     # 🔎 FAISS 인덱스 빌드/로드/검색
│   │   ├── reco_reranker.py         # ♻️ MMR 기반 재랭킹
│   │   └── model_manager.py         # 📦 모델 로딩/관리(옵션)
│   ├── ⚙️ core/                     # 핵심 알고리즘
│   │   ├── color_processing.py      # 🌈 색상 처리 & x,y,S DBSCAN
│   │   └── formal_processing.py     # 👔 포멀 스코어링(dedup + tanh 스무딩)
│   ├── ⚙️ config/                   # 시스템 설정
│   │   ├── settings.py              # 🔧 글로벌 설정
│   │   └── labels.py               # 🏷️ 카테고리 라벨
│   ├── 🤖 models/                   # 기존 모델 (호환성)
│   │   └── yolos_detector.py        # 🔍 YOLO 모델 래퍼
│   └── 🌐 api/                      # REST API 서버
│       ├── main.py                 # 🚀 FastAPI 애플리케이션 (라우터 포함)
│       └── routers/                # 라우터 모듈
│           ├── health.py           # /health, /db/health
│           ├── images.py           # /images/analyze
│           └── reco.py             # /reco/* (FAISS 기반 추천)
├── 📊 scripts/                      # 실행 스크립트
│   └── check_yolo_labels.py        # ✅ 라벨 검증
├── 🧪 tests/                        # 테스트 스위트
│   ├── test_colorful_pipeline.py    # 🌈 Colorful 테스트(고정 이미지 우선 + 랜덤)
│   ├── test_formal_pipeline.py      # 👔 Formal 테스트(고정 이미지 우선 + 라벨 프리뷰)
│   └── test_unified_pipeline.py     # 🎯 통합 파이프라인 테스트
├── 📚 docs/                         # 기술 문서
│   ├── project_structure.md        # 📋 프로젝트 구조
│   └── *.ipynb                    # 📓 분석 노트북
└── 🗃️ dataset/                      # 분류기 개발용 샘플 데이터
    ├── minimal/ (100장)           # 🎯 기본 테스트 세트
    └── maximal/ (100장)           # 🚀 확장 검증 세트
```

## 🚀 시작방법

### 1️⃣ 환경 설정

```bash
# 가상환경 활성화 
source venv/bin/activate

# 필수 의존성 설치
pip install -r requirements.txt

# YOLOS(Fashionpedia) 감지를 위해 transformers가 필요합니다 (requirements에 없다면 설치)
pip install "transformers>=4.21.0"
```

### 2️⃣ 기본 실행

```bash
# 🌈 Colorful 파이프라인 테스트 (랜덤 이미지)
python tests/test_colorful_pipeline.py

# 👔 Formal 파이프라인 테스트 (랜덤 샘플 1장 시각화)
python tests/test_formal_pipeline.py

# ✅ YOLO 라벨 검증
python scripts/check_yolo_labels.py

# 🎯 통합 분석 테스트
python tests/test_unified_pipeline.py
```

### 3️⃣ API 서버 시작

```bash
# 🌐 FastAPI 개발 서버 (자동 리로드)
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# 📱 API 문서 확인
# http://localhost:8000/docs
```

### 4️⃣ 추천 API(FAISS) 빠른 사용법

```bash
# 1) 서버 실행 후 인덱스 로드/재구축 (RDS MySQL에서 v1..v4 읽기, 실패 시 파일 폴백)
curl -X POST http://localhost:8000/reco/reload

# 2) 헬스 체크(인덱스 로딩 여부 및 벡터 수)
curl http://localhost:8000/reco/health

# 3) 사용자 벡터 기반 추천 요청 (v1,v2,v3는 검색, v4는 다양성)
curl -X POST http://localhost:8000/reco/by-user-vector \
  -H 'Content-Type: application/json' \
  -d '{"vector": [0.5, -0.2, 0.8, 0.3], "k": 30}'
```

- 인덱스 엔진: FAISS IndexFlatIP (L2 정규화된 내적 = 코사인 유사도)
- 인덱스 파일: `src/faiss/index_ip.faiss`(벡터), `src/faiss/idmap.json`(메타데이터)
- v4(다양성) → λ, 후보군 크기(C) 매핑:
  - λ = 0.6 − 0.3·v4, 이후 [0.3, 0.9]로 클램핑
  - C = ceil(k · (1 + 0.4·v4)), 이후 [ceil(0.6k), ceil(1.4k)]로 클램핑
- 초기 후보는 FAISS로 검색하고, 최종 결과는 MMR로 재랭킹합니다.
- `POST /reco/reload`는 RDS에서 최신 벡터를 읽어 인덱스를 갱신하며, 0건/오류 시 디스크의 인덱스/메타로 폴백합니다.

## 🌐 API 설명

- GET `/health`: 서버 상태 확인
- POST `/images/analyze`: 단일 이미지 통합 분석 실행
- GET `/reco/health`: 추천 인덱스 로딩 여부/카운트
- POST `/reco/reload`: RDS MySQL에서 인덱스 재구축(오류/0건 시 파일 폴백)
- POST `/reco/by-user-vector`: 사용자 벡터 기반 추천
  - 요청: `{ "vector": [v1, v2, v3, v4], "k": 30 }`
  - 응답: `items[]`에 `id, localPath, v1..v4, score_ip, score_mmr, rank`

 

## 🔬 핵심 알고리즘

### 🤖 객체 감지 엔진
- **모델**: YOLOS Fashionpedia (HuggingFace Transformers)
- **입력**: RGB 이미지 (자동 리사이즈)
- **출력**: 46개 패션 카테고리 감지 결과
- **성능**: 기본 신뢰도 0.4 기준(설정 가능)

### 🎨 색상 분석 알고리즘(요약)
1) RGB→HSV, 채도(S) 추출, 박스 중심부 크롭
2) 3D 특징 [α·x_norm, α·y_norm, β·S_norm] 구성
3) DBSCAN(eps, min_samples)으로 군집 라벨링
4) 각 클러스터의 채도 분포에 대해 Trimmed Mean(절삭평균) 계산
5) 가장 밝은 클러스터의 대표 채도를 [-1,1] 점수로 변환(작은 면적이면 p90 폴백), 이미지 점수는 박스 점수 중 최대값


## 📄 라이센스

이 프로젝트는 **MIT 라이센스** 하에 배포됩니다.

---

## 🆕 버전 히스토리
### 0.9
- 리로드 동작을 RDS 우선(실패 시 파일 폴백)으로 변경
- 인덱스 기본 경로를 `src/faiss`로 변경(퍼시스턴스 포함)
- CI/CD 추가: `.github/workflows/deploy.yml`

### 0.8
- 추천 API 추가: `/reco/health`, `/reco/reload`, `/reco/by-user-vector`
- FAISS 통합: IndexFlatIP + L2 정규화, `index_ip.faiss`/`idmap.json` 퍼시스턴스
- MMR 재랭킹 도입: v4 다양성 → λ(0.3~0.9), 후보군 크기 C(0.6k~1.4k)
- Formal 점수 정밀도 개선: float 유지, conf 기본값 0.4로 정렬
- YOLOS 이미지 로딩 경로 개선: PIL 중심으로 안정성 향상(cv2 의존 완화)
- 의존성 추가: `faiss-cpu`
### 0.7
- ✅: FastAPI 라우터 분리(`api/routers/health.py`, `api/routers/images.py`)
- ✅: 라벨 점수 합산 후 클래스 중복 제거 + tanh 스무딩 적용, conf≥0.8 유지
- ✅: `test_colorful_pipeline.py`/`test_formal_pipeline.py`에 고정 샘플 우선/라벨 프리뷰 추가
- ✅: 최소 연결 유틸 추가(`/db/health`), 설정은 환경변수(`DB_HOST` 등) 기반

### 0.6
- ✅ 성능 개선: 통합 파이프라인 병렬화 + 단일 감지 공유로 평균 처리시간 약 2.2초 → 0.8초 (63.6% 지연 감소)
- ✅ 구조 개선: `ColorfulPipeline`에서 감지기 분리, `UnifiedPipeline`이 단일 감지기 소유 
- ✅ Colorful 로직 보강: 밝은 클러스터 우선 + p90 폴백, 시각화/로그 개선


### 0.5
- ✅ Unified 파이프라인 구현
- ✅ Colorful 파이프라인 수정 (위치정보 + 채도의 3차원 기반 DBSCAN)
- ✅ Processor 디렉토리 제거
- ✅ utils/file_utils.py : 파일 입력 검증 로직 일원화
- ✅ utils/result_schema.py : - 파이프라인 간 표준 반환 형식 일원화

### 0.4
- ✅ Formal 파이프라인 구현(라벨 기반 -1/0/1, conf≥0.8 평균)
- ✅ FormalVisualizer 추가 및 테스트 연동(랜덤 샘플 시각화)
- ✅ visualizers 디렉토리 정리(공통 `plotting.py`), utils 내 시각화 제거
- ✅ API 레거시 의존 제거 및 최신 파이프라인 연동
- ✅ 레거시 코드 정리(`core/clustering.py`, `models/color_analyzer.py` 제거)

### 0.3
- ✅ 3분류기 아키텍처 스캐폴딩(Colorful/Maximal/Formal)
- ✅ 모듈화 리팩터: 파이프라인별 독립 구조로 분리
- ✅ UnifiedPipeline 도입(결과 통합)
- ✅ Colorful 파이프라인 완성(DBSCAN+Trimmed Mean)
- ✅ 테스트 구조 정리(`tests/` 분리)

### 0.2
- ✅ 모듈화 진행 및 통합 감지 파이프라인(초기 DetectionPipeline)
- ✅ FastAPI 기반 API 초안 제공

### 0.1
- ✅ 초기 프로토타입: YOLO 객체 감지, DBSCAN 색상 클러스터링, 기본 시각화

---
