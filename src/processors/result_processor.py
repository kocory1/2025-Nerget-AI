"""
결과 처리 모듈
분석 결과를 처리하고 집계하는 모듈
"""

import numpy as np
import os
from typing import List, Dict, Any


class ResultProcessor:
    """분석 결과 처리기"""
    
    def __init__(self):
        """처리기 초기화"""
        pass
    
    def process_analysis_results(self, image_path: str, image_rgb, 
                               detections: List[Dict[str, Any]], 
                               analyzed_results: List[Dict[str, Any]],
                               verbose: bool = True) -> Dict[str, Any]:
        """
        분석 결과를 종합하여 최종 결과를 생성
        
        Args:
            image_path: 이미지 파일 경로
            image_rgb: RGB 이미지 배열
            detections: 원본 감지 결과
            analyzed_results: 색상 분석 결과
            verbose: 상세 출력 여부
            
        Returns:
            종합된 분석 결과
        """
        # 전체 결과 요약 출력
        if verbose:
            self._print_summary(analyzed_results)
        
        # 결과 딕셔너리 생성
        result = {
            "image_path": image_path,
            "image_rgb": image_rgb,
            "detections": analyzed_results,
            "total_detections": len(detections),
            "analyzed_regions": len(analyzed_results),
            "average_score": self._calculate_average_score(analyzed_results),
            "success": True
        }
        
        return result
    
    def _print_summary(self, analyzed_results: List[Dict[str, Any]]) -> None:
        """분석 결과 요약을 출력"""
        print(f"\n📋 전체 분석 결과 요약:")
        print(f"분석된 영역 수: {len(analyzed_results)}")
        
        if analyzed_results:
            # 전체 평균 점수
            avg_score = self._calculate_average_score(analyzed_results)
            print(f"전체 평균 점수: {avg_score:.3f}")
            
            # 절삭평균 채도 값들 출력
            print(f"\n📊 각 영역별 절삭평균 채도:")
            for result in analyzed_results:
                print(f"  영역 {result['region_id']+1}: {result['max_saturation']:.1f}")
    
    def _calculate_average_score(self, analyzed_results: List[Dict[str, Any]]) -> float:
        """평균 점수를 계산"""
        if not analyzed_results:
            return 0.0
        
        scores = [r['score'] for r in analyzed_results]
        return float(np.mean(scores))
    
    def create_error_result(self, error_message: str, image_path: str = "") -> Dict[str, Any]:
        """
        오류 결과를 생성
        
        Args:
            error_message: 오류 메시지
            image_path: 이미지 파일 경로 (선택적)
            
        Returns:
            오류 결과 딕셔너리
        """
        return {
            "error": error_message,
            "success": False,
            "image_path": image_path,
            "detections": []
        }
    
    def validate_inputs(self, image_path: str) -> Dict[str, Any]:
        """
        입력값을 검증
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            검증 결과 (성공 시 빈 딕셔너리, 실패 시 오류 정보)
        """
        if not os.path.exists(image_path):
            return self.create_error_result(f"이미지를 찾을 수 없습니다: {image_path}")
        
        return {}  # 검증 성공
