"""
추상 파이프라인 베이스 클래스
모든 파이프라인의 공통 인터페이스를 정의
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BasePipeline(ABC):
    """파이프라인 추상 베이스 클래스"""
    
    @abstractmethod
    def detect_and_analyze(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """
        이미지 감지 및 분석 메인 메서드
        
        Args:
            image_path: 이미지 파일 경로
            **kwargs: 추가 파라미터
            
        Returns:
            분석 결과 딕셔너리
        """
        pass
    
    @abstractmethod
    def visualize_results(self, analysis_result: Dict[str, Any]) -> None:
        """
        분석 결과 시각화
        
        Args:
            analysis_result: 분석 결과 딕셔너리
        """
        pass
    
    def get_style_classification(self, score: float) -> str:
        """
        점수에 따른 스타일 분류
        
        Args:
            score: 색상 점수
            
        Returns:
            스타일 분류 ("Basic", "Neutral", "Showy")
        """
        if score < -0.3:
            return "Basic"
        elif score <= 0.3:
            return "Neutral"
        else:
            return "Showy"
