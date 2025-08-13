"""
객체 감지 모듈
YOLO 모델을 사용한 의류 객체 감지
"""

from typing import List, Dict, Any, Optional


class ObjectDetector:
    """YOLO 기반 객체 감지기"""
    
    def __init__(self):
        """감지기 초기화"""
        self.analyzer = None
        self._init_yolo_analyzer()
    
    def _init_yolo_analyzer(self) -> None:
        """YOLO 분석기 초기화"""
        try:
            # YOLOSDetector 사용 (실제 존재하는 클래스)
            from ..models.yolos_detector import YOLOSDetector
            self.analyzer = YOLOSDetector()
            print("✅ YOLOS 분석기 초기화 완료 (Fashionpedia 모델 사용)")
        except ImportError as e:
            print(f"❌ YOLO 모듈 import 실패: {e}")
            self.analyzer = None
        except Exception as e:
            print(f"❌ YOLO 분석기 초기화 실패: {e}")
            self.analyzer = None
    
    def detect_objects(self, image_path: str, conf_threshold: float = 0.8, 
                      verbose: bool = True) -> List[Dict[str, Any]]:
        """
        이미지에서 객체를 감지
        
        Args:
            image_path: 이미지 파일 경로
            conf_threshold: 신뢰도 임계값
            verbose: 상세 출력 여부
            
        Returns:
            감지된 객체들의 리스트
        """
        if self.analyzer is None:
            if verbose:
                print("❌ YOLO 분석기가 초기화되지 않았습니다.")
            return []
        
        try:
            if verbose:
                print("🔍 YOLO 객체 감지 중...")
            
            detected_boxes = self.analyzer.detect_clothing(
                image_path, 
                conf_threshold=conf_threshold
            )
            
            if verbose:
                print(f"✅ 감지된 객체: {len(detected_boxes)}개")
            
            return detected_boxes
            
        except Exception as e:
            if verbose:
                print(f"❌ 객체 감지 중 오류 발생: {e}")
            return []
    
    def is_ready(self) -> bool:
        """감지기가 사용 가능한지 확인"""
        return self.analyzer is not None
