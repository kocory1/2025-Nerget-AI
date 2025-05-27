from ultralytics import YOLO
import cv2

# 옷 탐지를 위한 YOLOv8 객체 탐지 모델
model = YOLO("yolo11n-cls.pt")  # 또는 yolov8n.pt 등 Detection 모델

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 탐지
    results = model(frame)  # Detection 모델이면 bounding box 반환
    annotated = results[0].plot()
    cv2.imshow("Clothing Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
