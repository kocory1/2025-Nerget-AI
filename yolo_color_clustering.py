import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter
import torch.serialization

class YOLOColorClustering:
    def __init__(self, model_path="runs/detect/train6/weights/best.pt"):
        """
        YOLO 모델과 색상 클러스터링을 결합한 클래스
        
        Args:
            model_path (str): 학습된 YOLO 모델 경로
        """
        # PyTorch 2.6+ 호환성을 위한 안전한 글로벌 설정
        try:
            torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
        except:
            pass
        
        try:
            self.model = YOLO(model_path)
            self.detected_regions = []
        except Exception as e:
            print(f"YOLO 모델 로딩 실패: {e}")
            print("기본 YOLO 모델을 사용합니다.")
            try:
                self.model = YOLO('yolov8n.pt')  # 기본 모델 사용
            except:
                print("YOLO 모델을 사용할 수 없습니다. 색상 분석만 수행합니다.")
                self.model = None
            self.detected_regions = []
        
    def detect_clothing(self, image_path, conf_threshold=0.5):
        """
        이미지에서 의류를 감지합니다.
        
        Args:
            image_path (str): 이미지 파일 경로
            conf_threshold (float): 신뢰도 임계값
            
        Returns:
            list: 감지된 의류 영역들의 바운딩 박스 좌표
        """
        if self.model is None:
            # YOLO 모델이 없으면 전체 이미지를 하나의 영역으로 처리
            image = cv2.imread(image_path)
            if image is not None:
                height, width = image.shape[:2]
                return [{
                    'bbox': [0, 0, width, height],
                    'confidence': 1.0,
                    'class_id': 0
                }]
            return []
        
        # YOLO로 의류 감지
        results = self.model(image_path, conf=conf_threshold)
        
        detected_boxes = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 바운딩 박스 좌표 추출 (x1, y1, x2, y2)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = box.cls[0].cpu().numpy()
                    
                    detected_boxes.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'class_id': int(class_id)
                    })
        
        self.detected_regions = detected_boxes
        return detected_boxes
    
    def extract_region_colors_dbscan(self, image_path, bbox, eps=30, min_samples=10):
        """
        DBSCAN을 사용하여 감지된 영역에서 색상을 추출하고 클러스터링합니다.
        
        Args:
            image_path (str): 이미지 파일 경로
            bbox (list): 바운딩 박스 좌표 [x1, y1, x2, y2]
            eps (float): DBSCAN의 이웃 반경
            min_samples (int): DBSCAN의 최소 샘플 수
            
        Returns:
            tuple: (클러스터별 평균 채도, 클러스터별 색상, 클러스터별 비율, 전체 분석 결과)
        """
        # 이미지 로드
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 바운딩 박스로 영역 추출
        x1, y1, x2, y2 = bbox
        region = image[y1:y2, x1:x2]
        
        if region.size == 0:
            return None, None, None, None
        
        # HSV 변환
        hsv_region = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv_region)
        
        # 픽셀 데이터 준비 (RGB + HSV)
        pixels_rgb = region.reshape(-1, 3)
        pixels_hsv = hsv_region.reshape(-1, 3)
        
        # DBSCAN 클러스터링 (RGB 공간에서)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(pixels_rgb)
        
        # 클러스터별 분석
        unique_labels = np.unique(cluster_labels)
        cluster_analysis = {}
        
        for label in unique_labels:
            if label == -1:  # 노이즈 클러스터
                continue
                
            # 해당 클러스터의 픽셀들
            mask = cluster_labels == label
            cluster_pixels_hsv = pixels_hsv[mask]
            cluster_pixels_rgb = pixels_rgb[mask]
            
            # 클러스터별 평균 HSV 값
            avg_hsv = np.mean(cluster_pixels_hsv, axis=0)
            avg_rgb = np.mean(cluster_pixels_rgb, axis=0)
            
            # 클러스터 크기 (비율)
            cluster_ratio = np.sum(mask) / len(cluster_labels)
            
            cluster_analysis[label] = {
                'avg_hsv': avg_hsv,
                'avg_rgb': avg_rgb,
                'ratio': cluster_ratio,
                'saturation': avg_hsv[1],  # 채도
                'pixel_count': np.sum(mask)
            }
        
        # 클러스터별 평균 채도 계산
        cluster_saturations = []
        cluster_colors = []
        cluster_ratios = []
        
        for label, analysis in cluster_analysis.items():
            cluster_saturations.append(analysis['saturation'])
            cluster_colors.append(analysis['avg_rgb'].astype(int))
            cluster_ratios.append(analysis['ratio'])
        
        # 전체 분석 결과
        total_analysis = {
            'total_clusters': len(cluster_analysis),
            'total_pixels': len(cluster_labels),
            'noise_pixels': np.sum(cluster_labels == -1),
            'noise_ratio': np.sum(cluster_labels == -1) / len(cluster_labels)
        }
        
        return cluster_saturations, cluster_colors, cluster_ratios, total_analysis
    
    def normalize_saturation_score(self, cluster_saturations, cluster_ratios, max_saturation=255):
        """
        클러스터별 평균 채도를 -1~1 범위로 정규화합니다.
        
        Args:
            cluster_saturations (list): 클러스터별 평균 채도
            cluster_ratios (list): 클러스터별 비율
            max_saturation (int): 최대 채도 값 (HSV에서 255)
            
        Returns:
            float: -1~1 범위의 정규화된 채도 점수
        """
        if not cluster_saturations:
            return 0.0
        
        # 가중 평균 채도 계산 (클러스터 비율로 가중)
        weighted_saturation = np.average(cluster_saturations, weights=cluster_ratios)
        
        # -1~1 범위로 정규화
        # 0 (무채색) -> 0
        # max_saturation (최대 채도) -> 1
        normalized_score = (weighted_saturation / max_saturation) * 2 - 1
        
        # -1~1 범위로 클리핑
        normalized_score = np.clip(normalized_score, -1.0, 1.0)
        
        return normalized_score
    
    def classify_style_by_region_dbscan(self, image_path, bbox, eps=30, min_samples=10):
        """
        DBSCAN 기반으로 감지된 영역의 스타일을 분류합니다.
        
        Args:
            image_path (str): 이미지 파일 경로
            bbox (list): 바운딩 박스 좌표
            eps (float): DBSCAN의 이웃 반경
            min_samples (int): DBSCAN의 최소 샘플 수
            
        Returns:
            dict: 분류 결과 (스타일, 정규화된 점수, 클러스터 정보)
        """
        # DBSCAN 색상 분석
        cluster_saturations, cluster_colors, cluster_ratios, total_analysis = self.extract_region_colors_dbscan(
            image_path, bbox, eps, min_samples
        )
        
        if cluster_saturations is None:
            return {
                'style': 'B',
                'normalized_score': -1.0,
                'clusters': [],
                'analysis': None
            }
        
        # 정규화된 채도 점수 계산
        normalized_score = self.normalize_saturation_score(cluster_saturations, cluster_ratios)
        
        # 스타일 분류 (-1~1 범위에서)
        # -1 ~ 0: Basic (B)
        # 0 ~ 1: Showy (S)
        style = 'S' if normalized_score > 0 else 'B'
        
        # 클러스터 정보 정리
        clusters_info = []
        for i, (sat, color, ratio) in enumerate(zip(cluster_saturations, cluster_colors, cluster_ratios)):
            clusters_info.append({
                'cluster_id': i,
                'saturation': sat,
                'color': color.tolist(),
                'ratio': ratio
            })
        
        return {
            'style': style,
            'normalized_score': normalized_score,
            'clusters': clusters_info,
            'analysis': total_analysis
        }
    
    def analyze_image_dbscan(self, image_path, conf_threshold=0.5, eps=30, min_samples=10):
        """
        DBSCAN을 사용하여 이미지를 분석합니다.
        
        Args:
            image_path (str): 이미지 파일 경로
            conf_threshold (float): YOLO 신뢰도 임계값
            eps (float): DBSCAN의 이웃 반경
            min_samples (int): DBSCAN의 최소 샘플 수
            
        Returns:
            dict: 분석 결과
        """
        # 1. 의류 감지
        detected_boxes = self.detect_clothing(image_path, conf_threshold)
        
        if not detected_boxes:
            return {
                'detected_regions': [],
                'analysis_results': [],
                'overall_style': 'No clothing detected',
                'overall_score': 0.0
            }
        
        # 2. 각 감지된 영역 분석
        analysis_results = []
        all_scores = []
        
        for i, detection in enumerate(detected_boxes):
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            # DBSCAN 기반 스타일 분류
            style_result = self.classify_style_by_region_dbscan(
                image_path, bbox, eps, min_samples
            )
            
            analysis_results.append({
                'region_id': i,
                'bbox': bbox,
                'confidence': confidence,
                'style': style_result['style'],
                'normalized_score': style_result['normalized_score'],
                'clusters': style_result['clusters'],
                'analysis': style_result['analysis']
            })
            
            all_scores.append(style_result['normalized_score'])
        
        # 3. 전체 스타일 결정 (가중 평균 점수 기반)
        if all_scores:
            overall_score = np.mean(all_scores)
            overall_style = 'S' if overall_score > 0 else 'B'
        else:
            overall_score = 0.0
            overall_style = 'B'
        
        return {
            'detected_regions': detected_boxes,
            'analysis_results': analysis_results,
            'overall_style': overall_style,
            'overall_score': overall_score
        }
    
    def visualize_results_dbscan(self, image_path, analysis_result):
        """
        DBSCAN 분석 결과를 시각화합니다.
        
        Args:
            image_path (str): 이미지 파일 경로
            analysis_result (dict): analyze_image_dbscan의 결과
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. 원본 이미지에 바운딩 박스 그리기
        axes[0].imshow(image)
        axes[0].set_title('Detected Clothing Regions')
        
        for result in analysis_result['analysis_results']:
            bbox = result['bbox']
            style = result['style']
            confidence = result['confidence']
            score = result['normalized_score']
            
            x1, y1, x2, y2 = bbox
            color = 'red' if style == 'S' else 'blue'
            
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, color=color, linewidth=2)
            axes[0].add_patch(rect)
            
            # 라벨 추가 (스타일, 신뢰도, 점수)
            label_text = f'{style} ({confidence:.2f}, {score:.2f})'
            axes[0].text(x1, y1-10, label_text, fontsize=10, color=color, 
                        weight='bold', bbox=dict(facecolor='white', alpha=0.7))
        
        axes[0].axis('off')
        
        # 2. 클러스터 색상 팔레트
        if analysis_result['analysis_results']:
            all_colors = []
            for result in analysis_result['analysis_results']:
                for cluster in result['clusters']:
                    all_colors.append(cluster['color'])
            
            if all_colors:
                colors_array = np.array(all_colors)
                axes[1].imshow(colors_array.reshape(-1, 1, 3))
                axes[1].set_title('DBSCAN Cluster Colors')
                axes[1].axis('off')
        
        # 3. 채도 점수 분포
        scores = [result['normalized_score'] for result in analysis_result['analysis_results']]
        if scores:
            axes[2].bar(range(len(scores)), scores, 
                       color=['red' if s > 0 else 'blue' for s in scores])
            axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[2].set_title('Normalized Saturation Scores')
            axes[2].set_xlabel('Region ID')
            axes[2].set_ylabel('Score (-1 to 1)')
            axes[2].set_ylim(-1.1, 1.1)
        
        plt.tight_layout()
        plt.show()
        
        # 결과 출력
        print(f"전체 스타일 분류: {analysis_result['overall_style']} (점수: {analysis_result['overall_score']:.3f})")
        print(f"감지된 의류 영역 수: {len(analysis_result['detected_regions'])}")
        
        for i, result in enumerate(analysis_result['analysis_results']):
            print(f"\n영역 {i+1}:")
            print(f"  스타일: {result['style']}")
            print(f"  신뢰도: {result['confidence']:.3f}")
            print(f"  정규화된 점수: {result['normalized_score']:.3f}")
            print(f"  클러스터 수: {len(result['clusters'])}")
            
            for j, cluster in enumerate(result['clusters'][:3]):  # 상위 3개만
                print(f"    클러스터 {j+1}: 채도={cluster['saturation']:.1f}, 비율={cluster['ratio']:.3f}")

    # 기존 메서드들 (하위 호환성을 위해 유지)
    def extract_region_colors(self, image_path, bbox, n_colors=5):
        """기존 K-means 기반 메서드 (하위 호환성)"""
        return self.extract_region_colors_dbscan(image_path, bbox)
    
    def classify_style_by_region(self, image_path, bbox, sat_thresh=40, showy_ratio_thresh=0.15):
        """기존 메서드 (하위 호환성)"""
        result = self.classify_style_by_region_dbscan(image_path, bbox)
        return result['style']
    
    def analyze_image(self, image_path, conf_threshold=0.5, n_colors=5):
        """기존 메서드 (하위 호환성)"""
        return self.analyze_image_dbscan(image_path, conf_threshold)
    
    def visualize_results(self, image_path, analysis_result):
        """기존 메서드 (하위 호환성)"""
        return self.visualize_results_dbscan(image_path, analysis_result)

# 사용 예시
def main():
    # 분석기 초기화
    analyzer = YOLOColorClustering()
    
    # 테스트 이미지 경로 (실제 이미지 경로로 변경하세요)
    test_image_path = "dataset/maximal/00048c3a2fb9c29340473c4cfc06424a.jpg"
    
    try:
        # DBSCAN 기반 이미지 분석
        results = analyzer.analyze_image_dbscan(test_image_path, eps=30, min_samples=10)
        
        # 결과 시각화
        analyzer.visualize_results_dbscan(test_image_path, results)
        
    except Exception as e:
        print(f"분석 중 오류 발생: {e}")

if __name__ == "__main__":
    main() 