import cv2
import numpy as np
from PIL import Image

def classify_bs_by_color(image_path, sat_thresh=40, showy_ratio_thresh=0.15):
    # 1. 이미지 로드
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. HSV로 변환
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # 3. 채도가 일정 기준 이상인 픽셀 비율 측정
    total_pixels = s.size
    showy_pixels = np.sum(s > sat_thresh)
    showy_ratio = showy_pixels / total_pixels

    print(f"채도 높은 픽셀 비율: {showy_ratio:.3f}")

    # 4. 비율에 따라 B/S 분류
    if showy_ratio > showy_ratio_thresh:
        return 'S'  # Showy
    else:
        return 'B'  # Basic

# ✅ 테스트 실행
image_path = "./datasets/train_images/MEN-Denim-id_00000080-01_7_additional.png"

bs_result = classify_bs_by_color(image_path)
print(f"➡️ Rule-based 색상 분석 결과: {bs_result}")
