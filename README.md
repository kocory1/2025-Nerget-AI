
## ✨ 주요 기능

### 🔍 AI 기반 의류 감지
- **46개 패션 카테고리** 자동 감지 (드레스, 스커트, 가방, 신발 등)
- **0.8+ 신뢰도** 기반 정확한 객체 인식
- **실시간 처리** 및 배치 분석 지원

### 🎨 고급 색상 분석
- **HSV 색상 공간** 기반 정밀 분석
- **DBSCAN 클러스터링**으로 노이즈 제거 및 대표 색상 추출
- **Trimmed Mean** 알고리즘으로 robust한 색상 값 계산

### 📊 스타일 인텔리전스
- **Basic** (-1.0 ~ -0.3): 미니멀하고 차분한 스타일
- **Neutral** (-0.3 ~ 0.3): 균형 잡힌 중간 톤 스타일  
- **Showy** (0.3 ~ 1.0): 화려하고 역동적인 스타일

### 🏗️ 엔터프라이즈 아키텍처
- **마이크로서비스** 구조로 개별 모듈 독립 배포 가능
- **RESTful API** 서버 (FastAPI 기반)
- **도커 컨테이너** 지원 (곧 출시)
- **수평 확장** 가능한 설계

## 🏗️ 시스템 아키텍처

```
2025-Nerget-AI/
├── 🧠 src/                          # 핵심 AI 모듈
│   ├── 🤖 models/                   # AI 모델 & 파이프라인
│   │   ├── detection_pipeline.py    # 🎯 메인 감지 파이프라인
│   │   ├── color_analyzer.py        # 🎨 색상 분석 엔진
│   │   └── yolos_detector.py        # 🔍 YOLO 객체 감지기
│   ├── ⚙️ core/                     # 핵심 알고리즘
│   │   ├── color_processing.py      # 🌈 색상 처리 & 클러스터링
│   │   └── clustering.py            # 📊 클러스터링 알고리즘
│   ├── 🛠️ utils/                    # 유틸리티 & 시각화
│   │   └── visualization.py         # 📈 결과 시각화 엔진
│   ├── ⚙️ config/                   # 시스템 설정
│   │   ├── settings.py              # 🔧 글로벌 설정
│   │   └── labels.py               # 🏷️ 카테고리 라벨
│   └── 🌐 api/                      # REST API 서버
│       └── main.py                 # 🚀 FastAPI 애플리케이션
├── 📊 scripts/                      # 실행 스크립트
│   ├── quick_test.py               # ⚡ 빠른 테스트
│   ├── check_yolo_labels.py        # ✅ 라벨 검증
│   └── classfi_color.py           # 🎨 색상 분류
├── 🧪 tests/                        # 테스트 스위트
│   └── test_runner.py              # 🏃 테스트 러너
├── 📚 docs/                         # 기술 문서
│   ├── project_structure.md        # 📋 프로젝트 구조
│   └── *.ipynb                    # 📓 분석 노트북
├── 🗃️ dataset/                      # 학습/검증 데이터
│   ├── minimal/ (100장)           # 🎯 기본 테스트 세트
│   └── maximal/ (100장)           # 🚀 확장 검증 세트
└── 🎯 test_yolo_color_simple.py    # 메인 실행 파일
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
# 🎯 메인 테스트 실행 (권장)
python test_yolo_color_simple.py

# ⚡ 빠른 테스트
python scripts/quick_test.py

# 🔍 종합 테스트 (의존성 체크 포함)
python tests/test_runner.py
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
from src.models.detection_pipeline import DetectionPipeline

# 🚀 파이프라인 초기화
pipeline = DetectionPipeline()

# 🔍 이미지 분석 실행
result = pipeline.detect_and_analyze(
    image_path="path/to/image.jpg", 
    conf_threshold=0.8
)

# 📊 결과 확인
if result.get("success"):
    print(f"감지된 객체: {len(result['detections'])}개")
    
    # 🎨 시각화
    pipeline.visualize_results(result)
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
from src.models.yolos_detector import YOLOSDetector

detector = YOLOSDetector()
detections = detector.detect(image_path, confidence_threshold=0.8)

for detection in detections:
    print(f"🏷️ {detection['label']}: {detection['confidence']:.3f}")
```

```python
# 📈 시각화만 사용
from src.utils.visualization import visualize_detection_results

visualized_image = visualize_detection_results(
    image_rgb, 
    detection_results,
    show_confidence=True,
    show_style_classification=True
)
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

### 📊 스타일 분류 로직
```python
def classify_style(saturation_score):
    if saturation_score <= -0.3:
        return "Basic"     # 미니멀, 차분함
    elif saturation_score <= 0.3:
        return "Neutral"   # 균형감, 안정감
    else:
        return "Showy"     # 화려함, 역동성
```

## 📊 분석 결과 해석

### 출력 예시
```
🎯 Nerget AI 패션 분석 결과

📷 이미지: test_image.jpg
🔍 총 감지 객체: 2개

📊 영역 1 분석:
  🏷️ 클래스: dress (ID: 10)
  🎯 신뢰도: 0.878
  📍 위치: [120, 45, 340, 580]
  🎨 색상 분석:
    └─ 발견된 클러스터: 2개
    └─ 선택된 클러스터: 1 (크기: 30,980픽셀)
    └─ 절삭평균 채도: 26.6/255
    └─ 채도 점수: -0.791
  📊 스타일 분류: Basic ✨

📊 영역 2 분석:
  🏷️ 클래스: bag (ID: 27)
  🎯 신뢰도: 0.923
  📍 위치: [450, 200, 520, 350]
  🎨 색상 분석:
    └─ 발견된 클러스터: 1개
    └─ 선택된 클러스터: 0 (크기: 8,450픽셀)
    └─ 절삭평균 채도: 180.3/255
    └─ 채도 점수: 0.415
  📊 스타일 분류: Showy 🌟

💫 전체 분석 완료! 시각화 결과를 확인하세요.
```

## 🛠️ 개발자 가이드

### 새로운 분석 모듈 추가

1. **색상 분석 알고리즘 확장**
```python
# src/core/color_processing.py에 새 함수 추가
def analyze_texture_patterns(image_region, bbox):
    """텍스처 패턴 분석 함수"""
    # 새로운 분석 로직 구현
    return texture_result
```

2. **파이프라인에 통합**
```python
# src/models/detection_pipeline.py 수정
def detect_and_analyze(self, image_path, **kwargs):
    # 기존 분석 + 새로운 분석 추가
    texture_result = analyze_texture_patterns(region, bbox)
    result['texture_analysis'] = texture_result
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

## 🚀 로드맵

### v3.0 (2024 Q2)
- [ ] 🐳 Docker 컨테이너 지원
- [ ] ☁️ 클라우드 배포 가이드 (AWS, GCP)
- [ ] 📱 모바일 앱 API 지원
- [ ] 🔄 실시간 스트리밍 분석

### v3.1 (2024 Q3)
- [ ] 🧠 추가 AI 모델 통합 (세그멘테이션)
- [ ] 🎨 고급 색상 분석 (색조, 명도 포함)
- [ ] 📊 사용자 맞춤 스타일 분류
- [ ] 🔍 텍스처 및 패턴 분석

### v4.0 (2024 Q4)
- [ ] 🌐 프론트엔드 웹 인터페이스
- [ ] 📈 실시간 대시보드
- [ ] 🤖 ChatGPT 연동 패션 조언
- [ ] 🛒 전자상거래 플랫폼 연동

## 🤝 기여하기

### 개발 참여
1. **이슈 리포팅**: 버그 발견 시 상세한 재현 단계와 함께 이슈 생성
2. **기능 제안**: 새로운 기능 아이디어를 이슈로 제안
3. **코드 기여**: 
   ```bash
   git checkout -b feature/새기능명
   # 개발 작업
   git commit -am "feat: 새기능 추가"
   git push origin feature/새기능명
   # Pull Request 생성
   ```

### 개발 가이드라인
- **코드 스타일**: PEP 8 준수
- **테스트**: 새 기능은 반드시 테스트 코드 포함
- **문서화**: docstring 및 README 업데이트
- **성능**: 기존 성능 저하 없이 개선

## 📞 지원 및 문의

- **기술 문의**: GitHub Issues 
- **버그 리포트**: GitHub Issues (bug 라벨)
- **기능 요청**: GitHub Issues (enhancement 라벨)

## 📄 라이센스

이 프로젝트는 **MIT 라이센스** 하에 배포됩니다.

---

## 🆕 버전 히스토리

### v2.0 (Current) - 모듈화 완성
- ✅ **완전 모듈화**: 모든 기능을 재사용 가능한 모듈로 분리
- ✅ **통합 파이프라인**: DetectionPipeline 클래스로 원스톱 분석
- ✅ **코드 최적화**: 283줄 → 41줄로 메인 실행 파일 간소화
- ✅ **API 서버**: FastAPI 기반 RESTful API 제공
- ✅ **확장성 향상**: 플러그인 방식으로 새 기능 추가 용이

### v1.0 - 초기 프로토타입
- ✅ YOLO 객체 감지 기본 구현
- ✅ DBSCAN 색상 클러스터링
- ✅ 기본 시각화 기능

---

*🎨 **Nerget AI**와 함께 패션의 미래를 만들어가세요!*