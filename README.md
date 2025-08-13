
## ✨ 주요 기능

### 🔍 AI 기반 의류 감지
- **46개 패션 카테고리** 자동 감지 (드레스, 스커트, 가방, 신발 등)
- **0.8+ 신뢰도** 기반 정확한 객체 인식
- **실시간 처리** 및 배치 분석 지원

### 🎯 3가지 스타일 분류기
- **🌈 Colorful**: 색상의 화려함과 생동감 분석
- **🔥 Maximal**: 맥시멀/미니멀 성향 분석 (개발 예정)
- **👔 Formal**: 포멀/캐주얼 격식성 분석 (개발 예정)

### 🎨 고급 색상 분석 (Colorful)
- **HSV 색상 공간** 기반 정밀 분석
- **DBSCAN 클러스터링**으로 노이즈 제거 및 대표 색상 추출
- **Trimmed Mean** 알고리즘으로 robust한 색상 값 계산

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
│   │   ├── colorful_pipeline.py     # 🌈 Colorful 분석 파이프라인
│   │   ├── maximal_pipeline.py      # 🔥 Maximal 분석 파이프라인 (TODO)
│   │   ├── formal_pipeline.py       # 👔 Formal 분석 파이프라인 (TODO)
│   │   ├── unified_pipeline.py      # 🎯 통합 분석 파이프라인
│   │   └── base_pipeline.py         # 📋 파이프라인 베이스 클래스
│   ├── 🔬 analyzers/                # 분석기 모듈
│   │   └── colorful_analyzer.py     # 🌈 색상 화려함 분석기
│   ├── 🤖 detectors/                # 객체 감지 모듈
│   │   └── object_detector.py       # 🔍 YOLO 객체 감지기
│   ├── ⚡ processors/               # 결과 처리 모듈
│   │   └── result_processor.py      # 📊 분석 결과 처리기
│   ├── 🖼️ visualizers/              # 시각화 모듈
│   │   └── image_visualizer.py      # 📈 이미지 시각화기
│   ├── ⚙️ core/                     # 핵심 알고리즘
│   │   ├── color_processing.py      # 🌈 색상 처리 & 클러스터링
│   │   └── clustering.py            # 📊 클러스터링 알고리즘
│   ├── 🛠️ utils/                    # 유틸리티
│   │   └── visualization.py         # 📈 시각화 유틸리티
│   ├── ⚙️ config/                   # 시스템 설정
│   │   ├── settings.py              # 🔧 글로벌 설정
│   │   └── labels.py               # 🏷️ 카테고리 라벨
│   ├── 🤖 models/                   # 기존 모델 (호환성)
│   │   ├── detection_pipeline.py    # (deprecated)
│   │   ├── color_analyzer.py        # (deprecated)  
│   │   └── yolos_detector.py        # 🔍 YOLO 모델 래퍼
│   └── 🌐 api/                      # REST API 서버
│       └── main.py                 # 🚀 FastAPI 애플리케이션
├── 📊 scripts/                      # 실행 스크립트
│   ├── quick_test.py               # ⚡ 빠른 테스트
│   ├── check_yolo_labels.py        # ✅ 라벨 검증
│   └── classfi_color.py           # 🎨 색상 분류
├── 🧪 tests/                        # 테스트 스위트
│   └── test_yolo_color_simple_modular.py  # 🌈 Colorful 파이프라인 테스트
├── 📚 docs/                         # 기술 문서
│   ├── project_structure.md        # 📋 프로젝트 구조
│   └── *.ipynb                    # 📓 분석 노트북
└── 🗃️ dataset/                      # 분류기 개발용 샘플 데이터
    ├── minimal/ (100장)           # 🎯 기본 테스트 세트 (Colorful)
    └── maximal/ (100장)           # 🚀 확장 검증 세트 (Maximal)
```

## 🚀 빠른 시작

### 1️⃣ 환경 설정

```bash
# 가상환경 활성화 (권장)
source venv/bin/activate

# 필수 의존성 설치
pip install -r requirements.txt
```

### 2️⃣ 기본 실행

```bash
# 🌈 Colorful 파이프라인 테스트 (권장)
python tests/test_yolo_color_simple_modular.py

# ✅ YOLO 라벨 검증
python scripts/check_yolo_labels.py

# 🎯 통합 분석 테스트 (추후 지원)
# python tests/test_unified_pipeline.py
```

### 3️⃣ API 서버 시작

```bash
# 🌐 FastAPI 개발 서버 (자동 리로드)
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# 📱 API 문서 확인
# http://localhost:8000/docs
```

## 💻 프로그래밍 가이드

### 기본 사용법

```python
# 🌈 개별 분류기 사용
from src.pipelines.colorful_pipeline import ColorfulPipeline

pipeline = ColorfulPipeline()
result = pipeline.detect_and_analyze("image.jpg", conf_threshold=0.8)

if result.get("success"):
    print(f"감지된 객체: {len(result['detections'])}개")
    pipeline.visualize_results(result)
```

```python
# 🎯 통합 분석 사용 (3가지 분류기)
from src.pipelines.unified_pipeline import UnifiedPipeline

unified = UnifiedPipeline()
result = unified.detect_and_analyze("image.jpg", conf_threshold=0.8)

if result.get("success"):
    for detection in result["detections"]:
        scores = detection["scores"]
        print(f"Colorful: {scores['colorful_score']:.3f}")
        print(f"Maximal: {scores['maximal_score']:.3f}")  # TODO
        print(f"Formal: {scores['formal_score']:.3f}")    # TODO
```

### 고급 모듈 사용

```python
# 🎨 색상 분석만 사용
from src.core.color_processing import analyze_region_with_clustering
import cv2

image = cv2.imread("image.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
bbox = [x1, y1, x2, y2]

color_result = analyze_region_with_clustering(image_rgb, bbox)
print(f"🎯 채도 점수: {color_result['saturation_score']:.3f}")
print(f"📊 스타일 분류: {color_result['style_classification']}")
```

```python
# 🔍 YOLO 감지만 사용
from src.detectors.object_detector import ObjectDetector

detector = ObjectDetector()
detections = detector.detect_objects(image_path, conf_threshold=0.8)

for detection in detections:
    print(f"🏷️ {detection['label']}: {detection['confidence']:.3f}")
```

```python
# 📈 시각화만 사용
from src.visualizers.image_visualizer import ImageVisualizer

visualizer = ImageVisualizer()
visualizer.visualize_analysis_results(analysis_result, verbose=True)
```

## 🔬 핵심 알고리즘

### 🤖 객체 감지 엔진
- **모델**: YOLOS Fashionpedia (HuggingFace Transformers)
- **입력**: RGB 이미지 (자동 리사이즈)
- **출력**: 46개 패션 카테고리 감지 결과
- **성능**: 신뢰도 0.8+ 기준 고정밀 감지

### 🎨 색상 분석 알고리즘
```python
# 1. RGB → HSV 변환
hsv_image = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)

# 2. 채도(S) 채널 추출 및 DBSCAN 클러스터링
clustering = DBSCAN(eps=1, min_samples=total_pixels//50)
clusters = clustering.fit_predict(saturation_values)

# 3. 최대 클러스터 선택 (노이즈 제거)
largest_cluster = max(clusters, key=cluster_size)

# 4. Trimmed Mean으로 robust한 대표값 계산
trimmed_mean = trim_mean(cluster_values, proportiontocut=0.2)

# 5. 정규화된 채도 점수 계산
saturation_score = (trimmed_mean / 255.0) * 2 - 1
```


## 📊 분석 결과 해석

### Colorful 파이프라인 출력 예시
```
🌈 Colorful 파이프라인 테스트 (YOLO + DBSCAN 색상 분석)
======================================================================
1. YOLOS 분석기 초기화 중... (Fashionpedia 모델)
✅ YOLOS 분석기 초기화 완료
📸 테스트 이미지: test_image.jpg

🔍 YOLO 객체 감지 중...
✅ 감지된 객체: 3개

🎨 각 영역별 색상 분석 중...

📊 영역 1 분석:
  클래스: bag, wallet (ID: 24)
  신뢰도: 0.988
  🎯 발견된 클러스터: 1개
  📊 노이즈 픽셀: 24,913개 (93.0%)
  🎯 절삭평균 채도: 255.0
  📊 채도 점수: 1.000 (Showy)

📊 영역 2 분석:
  클래스: dress (ID: 10)
  신뢰도: 0.802
  🎯 절삭평균 채도: 24.1
  📊 채도 점수: -0.811 (Basic)

📋 전체 분석 결과 요약:
분석된 영역 수: 3
전체 평균 점수: -0.209

🖼️ 결과 시각화...
✅ 시각화 완료
🎉 Colorful 파이프라인 테스트 완료!
```

### 통합 파이프라인 출력 예시 (예정)
```json
{
  "success": true,
  "pipeline_type": "unified",
  "detections": [
    {
      "region_id": 0,
      "class_name": "dress",
      "scores": {
        "colorful_score": -0.811,
        "maximal_score": 0.0,    // TODO
        "formal_score": 0.0      // TODO
      }
    }
  ],
  "overall_scores": {
    "colorful_score": -0.209,
    "maximal_score": 0.0,
    "formal_score": 0.0
  }
}
```

## 🛠️ 개발자 가이드

### 새로운 분류기 추가 (예: Shape 분류기)

1. **새 분석기 생성**
```python
# src/analyzers/shape_analyzer.py
from .base_analyzer import BaseAnalyzer

class ShapeAnalyzer(BaseAnalyzer):
    def analyze_detections(self, image_path, detections):
        # Shape 분석 로직 구현
        return shape_results
```

2. **새 파이프라인 생성**
```python
# src/pipelines/shape_pipeline.py
from .base_pipeline import BasePipeline
from ..analyzers.shape_analyzer import ShapeAnalyzer

class ShapePipeline(BasePipeline):
    def __init__(self):
        self.analyzer = ShapeAnalyzer()
    
    def detect_and_analyze(self, image_path, **kwargs):
        # Shape 분석 로직
        return shape_result
```

3. **통합 파이프라인에 추가**
```python
# src/pipelines/unified_pipeline.py 수정
from .shape_pipeline import ShapePipeline

class UnifiedPipeline:
    def __init__(self):
        self.shape_pipeline = ShapePipeline()  # 추가
        # 기존 파이프라인들...
```

### 설정 커스터마이징

```python
# src/config/settings.py
class Settings:
    # 이미지 처리 설정
    MAX_IMAGE_SIZE = 800
    
    # 클러스터링 설정
    DBSCAN_EPS = 1
    DBSCAN_MIN_SAMPLES_RATIO = 50
    
    # 신뢰도 임계값
    DEFAULT_CONFIDENCE_THRESHOLD = 0.8
    
    # 시각화 설정
    VISUALIZATION_DPI = 150
    FIGURE_SIZE = (15, 10)
```

### API 엔드포인트 확장

```python
# src/api/main.py에 새 엔드포인트 추가
@app.post("/analyze/batch")
async def analyze_batch_images(files: List[UploadFile]):
    """배치 이미지 분석 API"""
    results = []
    for file in files:
        # 배치 처리 로직
        result = pipeline.detect_and_analyze(file)
        results.append(result)
    return {"batch_results": results}
```

## 📈 성능 최적화

### 메모리 최적화
- **이미지 자동 리사이즈**: 800px 이상 이미지 자동 축소
- **배치 처리**: 메모리 효율적인 순차 처리
- **모델 캐싱**: 초기 로딩 후 메모리 상주

### 처리 속도 최적화
- **병렬 처리**: 다중 영역 분석 시 멀티프로세싱 지원
- **GPU 가속**: CUDA 지원 (설정 시)
- **지연 로딩**: 필요한 모듈만 선택적 로딩

### 확장성 최적화
- **마이크로서비스**: 모듈별 독립 배포 가능
- **로드 밸런싱**: 다중 인스턴스 지원
- **캐시 시스템**: Redis 연동 예정

## 🔍 데이터셋 정보

### 검증 데이터셋
- **Minimal Dataset**: 100장 (기본 기능 테스트)
- **Maximal Dataset**: 100장 (성능 벤치마크)
- **라벨링**: 수동 검증된 ground truth 포함

### 지원 이미지 형식
- **형식**: JPG, PNG, JPEG
- **해상도**: 최소 224x224, 권장 800x800+
- **색상**: RGB 컬러 이미지

## 📦 의존성 관리

### 핵심 패키지
```
🤖 AI/ML:
├── transformers >= 4.21.0    # YOLOS 모델
├── torch >= 1.12.0          # 딥러닝 프레임워크
├── ultralytics >= 8.0.0     # YOLO 유틸리티
└── scikit-learn >= 1.1.0    # 클러스터링

🖼️ 이미지 처리:
├── opencv-python >= 4.6.0   # 컴퓨터 비전
├── Pillow >= 9.2.0          # 이미지 처리
└── matplotlib >= 3.5.0      # 시각화

🌐 웹 서비스:
├── fastapi >= 0.85.0        # API 프레임워크
├── uvicorn >= 0.18.0        # ASGI 서버
└── python-multipart         # 파일 업로드

📊 데이터 처리:
├── numpy >= 1.21.0          # 수치 연산
├── pandas >= 1.4.0          # 데이터 분석
└── scipy >= 1.9.0           # 과학 계산
```

전체 의존성은 `requirements.txt` 참조.


## 📄 라이센스

이 프로젝트는 **MIT 라이센스** 하에 배포됩니다.

---

## 🆕 버전 히스토리

### v3.0 (Current) - 3가지 분류기 구조 완성
- ✅ **3가지 분류기 아키텍처**: Colorful, Maximal, Formal 분류기 설계
- ✅ **모듈화 리팩토링**: 파이프라인별 독립적 구조로 분리
- ✅ **통합 파이프라인**: UnifiedPipeline으로 3가지 분석 통합
- ✅ **Colorful 파이프라인**: 색상 화려함 분석 완전 구현
- ✅ **확장 가능한 설계**: 새로운 분류기 쉽게 추가 가능
- ✅ **테스트 구조 정리**: tests/ 디렉토리로 테스트 파일 이동

### v2.0 - 모듈화 완성
- ✅ **완전 모듈화**: 모든 기능을 재사용 가능한 모듈로 분리
- ✅ **통합 파이프라인**: DetectionPipeline 클래스로 원스톱 분석
- ✅ **API 서버**: FastAPI 기반 RESTful API 제공

### v1.0 - 초기 프로토타입
- ✅ YOLO 객체 감지 기본 구현
- ✅ DBSCAN 색상 클러스터링
- ✅ 기본 시각화 기능

---
