#!/usr/bin/env python3
"""
YOLOS + DBSCAN 색상 클러스터링
"""

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoImageProcessor, YolosForObjectDetection
import torch
from PIL import Image

class YOLOSColorClustering:
    def __init__(self):
        """YOLOS 모델 초기화"""
        print("🔄 YOLOS 모델 로딩 중...")
        
        # 1. YOLOS 모델 로딩
        self.processor = AutoImageProcessor.from_pretrained("valentinafeve/yolos-fashionpedia")
        self.model = YolosForObjectDetection.from_pretrained("valentinafeve/yolos-fashionpedia")
        
        print("✅ YOLOS 모델 로딩 완료")
        
        # 클래스 이름 정보 (Fashionpedia 데이터셋 기준)
        self.class_names = {
            0: 'shirt, blouse',
            1: 'top, t-shirt, sweatshirt', 
            2: 'sweater',
            3: 'cardigan',
            4: 'jacket',
            5: 'vest',
            6: 'pants',
            7: 'shorts',
            8: 'skirt',
            9: 'coat',
            10: 'dress',
            11: 'jumpsuit',
            12: 'cape',
            13: 'glasses',
            14: 'hat',
            15: 'headband, head covering, hair accessory',
            16: 'tie',
            17: 'glove',
            18: 'watch',
            19: 'belt',
            20: 'leg warmer',
            21: 'tights, stockings',
            22: 'sock',
            23: 'shoe',
            24: 'bag, wallet',
            25: 'scarf',
            26: 'umbrella',
            27: 'hood',
            28: 'collar',
            29: 'lapel',
            30: 'epaulette',
            31: 'sleeve',
            32: 'pocket',
            33: 'neckline',
            34: 'buckle',
            35: 'zipper',
            36: 'applique',
            37: 'bead',
            38: 'bow',
            39: 'flower',
            40: 'fringe',
            41: 'ribbon',
            42: 'rivet',
            43: 'ruffle',
            44: 'sequin',
            45: 'tassel'
        }
    
    def detect_clothing(self, image_path, conf_threshold=0.3):
        """YOLOS로 의류 객체 감지"""
        
        # 이미지 로드
        image = Image.open(image_path)
        
        # 전처리
        inputs = self.processor(images=image, return_tensors="pt")
        
        # 추론
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 후처리
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=conf_threshold
        )[0]
        
        # 결과 변환
        detected_boxes = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detected_boxes.append({
                'bbox': box.tolist(),  # [x1, y1, x2, y2]
                'confidence': score.item(),
                'class_id': label.item(),
                'class_name': self.class_names.get(label.item(), f'Class_{label.item()}')
            })
        
        return detected_boxes
    
    def analyze_color_clustering(self, image_path, conf_threshold=0.3):
        """YOLOS + DBSCAN 색상 클러스터링 분석"""
        
        print("🎨 YOLOS + DBSCAN 색상 클러스터링 분석")
        print("=" * 60)
        
        # 1. YOLOS 객체 감지
        print("1. YOLOS 객체 감지 중...")
        detected_boxes = self.detect_clothing(image_path, conf_threshold)
        print(f"✅ 감지된 객체: {len(detected_boxes)}개")
        
        if not detected_boxes:
            print("❌ 감지된 객체가 없습니다.")
            return []
        
        # 2. 이미지 로드
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        all_results = []
        
        # 3. 각 감지된 영역에 대해 색상 분석
        print("\n2. 각 영역별 색상 분석...")
        
        for i, detection in enumerate(detected_boxes):
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_id = detection['class_id']
            class_name = detection['class_name']
            
            print(f"\n📊 영역 {i+1} 분석:")
            print(f"  클래스: {class_name} (ID: {class_id})")
            print(f"  신뢰도: {confidence:.3f}")
            print(f"  바운딩 박스: {bbox}")
            
            # 바운딩 박스로 영역 추출
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            region = image_rgb[y1:y2, x1:x2]
            
            if region.size == 0:
                print("  ⚠️ 영역이 비어있습니다.")
                continue
            
            # 배경 제거 시도
            region_hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
            mask = np.ones(region.shape[:2], dtype=bool)
            
            # 명도와 채도 기반 배경 제거
            value_mask = (region_hsv[:, :, 2] > 50) & (region_hsv[:, :, 2] < 220)
            saturation_mask = region_hsv[:, :, 1] > 20
            mask = mask & value_mask & saturation_mask
            
            filtered_region = region.copy()
            filtered_region[~mask] = [0, 0, 0]
            
            valid_pixels = np.sum(mask)
            if valid_pixels < 1000:
                print("  ⚠️ 배경 제거 후 유효한 픽셀이 적어 원본 사용")
                filtered_region = region
            else:
                print(f"  🎯 배경 제거: {region.size//3} -> {valid_pixels} 픽셀")
            
            # 영역 크기 조정
            height, width = filtered_region.shape[:2]
            if height * width > 100000:
                scale_factor = np.sqrt(100000 / (height * width))
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                filtered_region = cv2.resize(filtered_region, (new_width, new_height))
                print(f"  📏 영역 크기 조정: {width}x{height} -> {new_width}x{new_height}")
            
            region = filtered_region
            
            # 4. 색상 클러스터링
            pixel_colors = region.reshape((-1, 3))
            print(f"  🔍 분석할 픽셀 수: {len(pixel_colors)}")
            
            # HSV 변환
            pixel_colors_hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
            pixel_colors_hsv = pixel_colors_hsv.reshape((-1, 3))
            
            # 채도 값 추출
            saturation_values = pixel_colors_hsv[:, 1]
            saturation_value_data = saturation_values.reshape(-1, 1)
            
            # DBSCAN 클러스터링
            dbscan = DBSCAN(eps=0.3, min_samples=1000)
            dbscan.fit(saturation_value_data)
            labels = dbscan.labels_
            
            # 클러스터 분석
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            print(f"  🎯 발견된 클러스터: {n_clusters}개")
            print(f"  📊 노이즈 픽셀: {n_noise}개")
            
            # 클러스터별 평균 채도 계산
            df_clusters = pd.DataFrame({
                'saturation': saturation_value_data.flatten(),
                'cluster': labels
            })
            
            df_filtered = df_clusters[df_clusters['cluster'] != -1].copy()
            
            if len(df_filtered) > 0:
                mean_saturation_per_cluster = df_filtered.groupby('cluster')[['saturation']].mean()
                print(f"  📈 클러스터별 평균 채도:")
                for cluster_id, mean_sat in mean_saturation_per_cluster.iterrows():
                    print(f"    클러스터 {cluster_id}: {mean_sat['saturation']:.1f}")
                
                # 화려함 점수 계산
                filtered_saturation_values = df_filtered['saturation']
                max_overall_saturation = filtered_saturation_values.max()
                vibrancy_score = (max_overall_saturation / 255.0) * 2 - 1
                
                print(f"  📊 영역 분석 결과:")
                print(f"    최대 채도: {max_overall_saturation:.1f}")
                print(f"    화려함 점수: {vibrancy_score:.3f}")
                
                all_results.append({
                    'region_id': i,
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': bbox,
                    'score': vibrancy_score,
                    'max_saturation': max_overall_saturation,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'labels': labels,
                    'region_shape': region.shape
                })
            else:
                print(f"  ⚠️ 유효한 클러스터가 없습니다.")
        
        # 5. 전체 결과 요약
        print(f"\n📋 전체 분석 결과 요약:")
        print(f"분석된 영역 수: {len(all_results)}")
        
        if all_results:
            avg_score = np.mean([r['score'] for r in all_results])
            print(f"전체 평균 점수: {avg_score:.3f}")
            
            print(f"\n📊 각 영역별 최대 채도:")
            for result in all_results:
                print(f"  영역 {result['region_id']+1}: {result['max_saturation']:.1f}")
        
        return all_results

def visualize_results(image, results):
    """결과 시각화 - 클러스터링 결과 포함"""
    
    if not results:
        print("시각화할 결과가 없습니다.")
        return
    
    n_results = len(results)
    fig, axes = plt.subplots(2, n_results, figsize=(5*n_results, 10))
    
    if n_results == 1:
        axes = axes.reshape(2, 1)
    
    for i, result in enumerate(results):
        bbox = result['bbox']
        confidence = result['confidence']
        score = result['score']
        max_sat = result['max_saturation']
        labels = result['labels']
        region_shape = result['region_shape']
        class_name = result['class_name']
        
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # 1. 원본 이미지에 바운딩 박스 그리기
        axes[0, i].imshow(image)
        axes[0, i].set_title(f'{class_name} (Score: {score:.3f})')
        
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           fill=False, color='red', linewidth=2)
        axes[0, i].add_patch(rect)
        
        label_text = f'신뢰도:{confidence:.2f}\n점수:{score:.2f}\n최대채도:{max_sat:.0f}'
        axes[0, i].text(x1, y1-10, label_text, fontsize=8, color='red', 
                    weight='bold', bbox=dict(facecolor='white', alpha=0.7))
        axes[0, i].axis('off')
        
        # 2. 클러스터링 결과 시각화
        if 'labels' in result and labels is not None:
            height, width = region_shape[:2]
            clustered_image = np.zeros((height, width, 3), dtype=np.uint8)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            import random
            cluster_colors = {j: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) 
                            for j in range(n_clusters)}
            noise_color = (0, 0, 0)
            
            for idx, label in enumerate(labels):
                row = idx // width
                col = idx % width
                if row < height and col < width:
                    if label == -1:
                        clustered_image[row, col] = noise_color
                    else:
                        clustered_image[row, col] = cluster_colors[label]
            
            axes[1, i].imshow(clustered_image)
            axes[1, i].set_title(f'DBSCAN Clustering\n({n_clusters} clusters)')
            axes[1, i].axis('off')
        else:
            axes[1, i].text(0.5, 0.5, 'No clustering\nresult', 
                           ha='center', va='center', transform=axes[1, i].transAxes)
            axes[1, i].set_title('No Clustering')
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 테스트
    analyzer = YOLOSColorClustering()
    results = analyzer.analyze_color_clustering("dataset/maximal/0269be0620747ecd46cd060a240b81b0.jpg")
    
    if results:
        image = cv2.imread("dataset/maximal/0269be0620747ecd46cd060a240b81b0.jpg")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        visualize_results(image_rgb, results) 