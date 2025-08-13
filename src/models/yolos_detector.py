"""
YOLOS Fashionpedia 모델 래퍼
의류 객체 감지를 위한 YOLOS 모델 인터페이스
"""

import cv2
import numpy as np
from PIL import Image
from transformers import YolosImageProcessor, YolosForObjectDetection
import torch
from typing import List, Dict, Tuple, Optional

from ..config.labels import get_label_name, should_analyze_color
from ..config.settings import DETECTION_CONFIG


class YOLOSDetector:
    """YOLOS Fashionpedia 모델 래퍼 클래스"""
    
    def __init__(self, model_name: str = "valentinafeve/yolos-fashionpedia"):
        """
        Args:
            model_name: 사용할 YOLOS 모델명
        """
        self.model_name = model_name
        self.confidence_threshold = DETECTION_CONFIG["confidence_threshold"]
        self.device = self._get_device()
        
        self.image_processor = None
        self.model = None
        self._load_model()
    
    def _get_device(self) -> str:
        """사용할 디바이스 결정"""
        device_setting = DETECTION_CONFIG["device"]
        
        if device_setting == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        else:
            return device_setting
    
    def _load_model(self):
        """모델 로드"""
        try:
            print(f"🔄 YOLOS 모델 로딩 중... ({self.model_name})")
            
            self.image_processor = YolosImageProcessor.from_pretrained(self.model_name)
            self.model = YolosForObjectDetection.from_pretrained(self.model_name)
            
            # GPU로 이동 (가능한 경우)
            if self.device == "cuda":
                self.model = self.model.to("cuda")
            
            print("✅ YOLOS 모델 로딩 완료")
            
        except Exception as e:
            print(f"❌ YOLOS 모델 로딩 실패: {e}")
            raise
    
    def detect_clothing(self, image_path: str, conf_threshold: Optional[float] = None) -> List[Dict]:
        """
        의류 객체 감지
        
        Args:
            image_path: 이미지 파일 경로
            conf_threshold: 신뢰도 임계값 (None이면 기본값 사용)
            
        Returns:
            감지된 객체 리스트 [{"class_id": int, "class_name": str, "confidence": float, "bbox": [x1,y1,x2,y2]}, ...]
        """
        threshold = conf_threshold or self.confidence_threshold
        
        # 이미지 로드
        image = Image.open(image_path).convert("RGB")
        
        # 전처리
        inputs = self.image_processor(images=image, return_tensors="pt")
        
        # GPU로 이동 (가능한 경우)
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # 추론
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 후처리
        target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
        if self.device == "cuda":
            target_sizes = target_sizes.to("cuda")
            
        results = self.image_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold
        )[0]
        
        # 결과 정리
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            class_id = label.item()
            confidence = score.item()
            bbox = box.cpu().numpy().tolist()  # [x1, y1, x2, y2]
            class_name = get_label_name(class_id)
            
            # 색상 분석 대상만 포함
            if should_analyze_color(class_id):
                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": bbox
                })
        
        return detections
    
    def load_image_as_rgb(self, image_path: str) -> np.ndarray:
        """
        이미지를 RGB 배열로 로드
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            RGB 이미지 배열 (H, W, 3)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # BGR to RGB 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_rgb
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "model_type": "YOLOS Fashionpedia"
        }