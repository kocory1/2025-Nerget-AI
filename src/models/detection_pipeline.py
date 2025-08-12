"""
통합 감지 파이프라인
YOLO 감지와 색상 분석을 통합하는 모듈
"""

import cv2
import numpy as np
import os
from typing import List, Dict, Any, Optional

from ..core.color_processing import analyze_region_with_clustering
from ..utils.visualization import visualize_detection_results


class DetectionPipeline:
    """통합 감지 및 분석 파이프라인"""
    
    def __init__(self):
        """파이프라인 초기화"""
        self.analyzer = None
        self._init_yolo_analyzer()
    
    def _init_yolo_analyzer(self):
        """YOLO 분석기 초기화"""
        try:
            # yolos_color_clustering을 직접 import
            from yolos_color_clustering import YOLOSColorClustering
            self.analyzer = YOLOSColorClustering()
            print("✅ YOLOS 분석기 초기화 완료 (Fashionpedia 모델 사용)")
        except ImportError as e:
            print(f"❌ YOLO 모듈 import 실패: {e}")
            self.analyzer = None
        except Exception as e:
            print(f"❌ YOLO 분석기 초기화 실패: {e}")
            self.analyzer = None
    
    def detect_and_analyze(self, image_path: str, conf_threshold: float = 0.8, 
                          verbose: bool = True) -> Dict[str, Any]:
        """
        이미지에서 객체를 감지하고 색상을 분석
        
        Args:
            image_path: 이미지 파일 경로
            conf_threshold: 신뢰도 임계값
            verbose: 상세 출력 여부
            
        Returns:
            분석 결과 딕셔너리
        """
        if self.analyzer is None:
            return {"error": "YOLO 분석기가 초기화되지 않았습니다."}
        
        if not os.path.exists(image_path):
            return {"error": f"이미지를 찾을 수 없습니다: {image_path}"}
        
        if verbose:
            print(f"📸 테스트 이미지: {os.path.basename(image_path)}")
        
        try:
            # 1. YOLO로 객체 감지
            if verbose:
                print("\\n2. YOLO 객체 감지 중...")
            
            detected_boxes = self.analyzer.detect_clothing(image_path, conf_threshold=conf_threshold)
            
            if verbose:
                print(f"✅ 감지된 객체: {len(detected_boxes)}개")
            
            if not detected_boxes:
                return {"error": "감지된 객체가 없습니다.", "detections": []}
            
            # 2. 이미지 로드 (색상 분석용)
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 3. 각 감지된 영역에 대해 색상 분석
            if verbose:
                print("\\n3. 각 영역별 색상 분석 (기존 color_clustering.ipynb 기준)...")
            
            all_results = []
            
            for i, detection in enumerate(detected_boxes):
                bbox = detection['bbox']
                confidence = detection['confidence']
                class_id = detection['class_id']
                class_name = detection.get('class_name', f'Class_{class_id}')
                
                if verbose:
                    print(f"\\n📊 영역 {i+1} 분석:")
                    print(f"  클래스: {class_name} (ID: {class_id})")
                    print(f"  신뢰도: {confidence:.3f}")
                    print(f"  바운딩 박스: {bbox}")
                
                # 색상 분석
                color_analysis = analyze_region_with_clustering(
                    image_rgb, bbox, verbose=verbose
                )
                
                if "error" not in color_analysis:
                    result = {
                        'region_id': i,
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': bbox,
                        'score': color_analysis['saturation_score'],
                        'saturation_score': color_analysis['saturation_score'],
                        'max_saturation': color_analysis['largest_cluster_saturation'],
                        'n_clusters': color_analysis['n_clusters'],
                        'n_noise': color_analysis['n_noise'],
                        'labels': color_analysis['labels'],
                        'region_shape': color_analysis['region_shape']
                    }
                    all_results.append(result)
                else:
                    if verbose:
                        print(f"  ⚠️ {color_analysis['error']}")
            
            # 4. 전체 결과 요약
            if verbose:
                print(f"\\n📋 전체 분석 결과 요약 (기존 기준):")
                print(f"분석된 영역 수: {len(all_results)}")
                
                if all_results:
                    # 전체 평균 점수
                    avg_score = np.mean([r['score'] for r in all_results])
                    print(f"전체 평균 점수: {avg_score:.3f}")
                    
                    # 절삭평균 채도 값들 출력
                    print(f"\\n📊 각 영역별 절삭평균 채도:")
                    for result in all_results:
                        print(f"  영역 {result['region_id']+1}: {result['max_saturation']:.1f}")
            
            return {
                "image_path": image_path,
                "image_rgb": image_rgb,
                "detections": all_results,
                "total_detections": len(detected_boxes),
                "analyzed_regions": len(all_results),
                "average_score": np.mean([r['score'] for r in all_results]) if all_results else 0,
                "success": True
            }
            
        except Exception as e:
            error_msg = f"분석 중 오류 발생: {e}"
            if verbose:
                print(f"❌ {error_msg}")
                import traceback
                traceback.print_exc()
            return {"error": error_msg, "success": False}
    
    def visualize_results(self, analysis_result: Dict[str, Any]):
        """분석 결과를 시각화"""
        if not analysis_result.get("success", False):
            print("시각화할 결과가 없습니다.")
            return
        
        print(f"\\n🖼️ 결과 시각화...")
        visualize_detection_results(
            analysis_result["image_rgb"], 
            analysis_result["detections"]
        )
    