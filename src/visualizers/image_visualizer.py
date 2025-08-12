"""
이미지 시각화 모듈
분석 결과를 시각화하는 모듈
"""

from typing import Dict, Any

from ..utils.visualization import visualize_detection_results


class ImageVisualizer:
    """이미지 시각화기"""
    
    def __init__(self):
        """ 이미지 시각화 초기화"""
        pass
    
    def visualize_analysis_results(self, analysis_result: Dict[str, Any], 
                                 verbose: bool = True) -> None:
        """
        분석 결과를 시각화
        
        Args:
            analysis_result: 분석 결과 딕셔너리
            verbose: 상세 출력 여부
        """
        if not analysis_result.get("success", False):
            if verbose:
                print("⚠️ 시각화할 결과가 없습니다.")
            return
        
        if verbose:
            print(f"\n🖼️ 결과 시각화...")
        
        try:
            visualize_detection_results(
                analysis_result["image_rgb"], 
                analysis_result["detections"]
            )
            
            if verbose:
                print("✅ 시각화 완료")
                
        except Exception as e:
            if verbose:
                print(f"❌ 시각화 중 오류 발생: {e}")
    
    def check_visualization_data(self, analysis_result: Dict[str, Any]) -> bool:
        """
        시각화에 필요한 데이터가 있는지 확인
        
        Args:
            analysis_result: 분석 결과 딕셔너리
            
        Returns:
            시각화 가능 여부
        """
        required_keys = ["image_rgb", "detections", "success"]
        
        for key in required_keys:
            if key not in analysis_result:
                print(f"❌ 시각화 데이터 누락: {key}")
                return False
        
        if not analysis_result["success"]:
            print("❌ 분석이 성공하지 않아 시각화할 수 없습니다.")
            return False
        
        return True
