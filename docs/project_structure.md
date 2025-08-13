# 🎨 Nerget AI - 의류 색상 분석 프로젝트 구조

## 📂 디렉토리 구조

```
2025-Nerget-AI/
├── 📁 src/                           # 소스 코드
│   ├── 📁 models/                    # AI 모델 관련
│   │   ├── yolo_detector.py          # YOLO 의류 감지
│   │   ├── yolos_detector.py         # YOLOS Fashionpedia 감지
│   │   └── color_analyzer.py         # 색상 분석 엔진
│   ├── 📁 core/                      # 핵심 분석 로직
│   │   ├── clustering.py             # DBSCAN 클러스터링
│   │   ├── color_processing.py       # HSV 색상 처리
│   │   └── scoring.py                # 채도 점수 계산
│   ├── 📁 utils/                     # 유틸리티
│   │   ├── image_utils.py            # 이미지 전처리
│   │   ├── visualization.py          # 결과 시각화
│   │   └── file_utils.py             # 파일 처리
│   ├── 📁 api/                       # API 관련
│   │   ├── main.py                   # FastAPI 메인
│   │   ├── endpoints.py              # API 엔드포인트
│   │   └── models.py                 # API 데이터 모델
│   └── 📁 config/                    # 설정 파일
│       ├── labels.py                 # 의류 라벨 정의
│       ├── settings.py               # 시스템 설정
│       └── constants.py              # 상수 정의
├── 📁 tests/                         # 테스트 파일
│   ├── test_basic_analysis.py        # 기본 분석 테스트
│   ├── test_performance.py           # 성능 테스트
│   └── test_accuracy.py              # 정확도 테스트
├── 📁 data/                          # 데이터 관련
│   ├── 📁 dataset/                   # 기존 데이터셋
│   ├── 📁 models/                    # 학습된 모델 파일
│   └── 📁 samples/                   # 샘플 이미지
├── 📁 docs/                          # 문서
│   ├── API_guide.md                  # API 사용 가이드
│   ├── algorithm_guide.md            # 알고리즘 설명
│   └── deployment_guide.md           # 배포 가이드
├── 📁 scripts/                       # 실행 스크립트
│   ├── run_server.py                 # 서버 실행
│   ├── run_tests.py                  # 테스트 실행
│   └── setup.py                      # 환경 설정
├── requirements.txt                  # 의존성
├── .env                             # 환경 변수
├── .gitignore                       # Git 무시 파일
└── README.md                        # 프로젝트 설명
```

## 🎯 주요 특징

### 1. **모듈화된 구조**
- 각 기능별로 명확히 분리
- 재사용 가능한 컴포넌트
- 테스트 용이성

### 2. **설정 분리**
- 의류 라벨 정의 분리
- 시스템 설정 외부화
- 환경별 설정 관리

### 3. **확장 가능성**
- 새로운 모델 쉽게 추가
- API 엔드포인트 확장 용이
- 테스트 케이스 추가 간편

### 4. **유지보수성**
- 명확한 책임 분리
- 문서화 체계
- 버전 관리 용이