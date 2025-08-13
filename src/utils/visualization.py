"""
결과 시각화 유틸리티
분석 결과를 그래프와 이미지로 시각화
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List
import os

from ..config.settings import VISUALIZATION_CONFIG


def visualize_analysis_results(analysis_result: Dict):
    """
    전체 분석 결과 시각화
    
    Args:
        analysis_result: analyze_clothing_image() 결과
    """
    if "error" in analysis_result:
        print(f"시각화 불가: {analysis_result['error']}")
        return
    
    # 이미지 로드
    image_path = analysis_result["image_path"]
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지 로드 실패: {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 시각화 구성
    fig, axes = plt.subplots(2, 2, figsize=VISUALIZATION_CONFIG["figure_size"])
    fig.suptitle(f"의류 색상 분석 결과: {os.path.basename(image_path)}", fontsize=16)
    
    # 1. 원본 이미지 + 바운딩 박스
    _plot_image_with_detections(axes[0, 0], image_rgb, analysis_result["region_results"])
    
    # 2. 채도 점수 분포
    _plot_saturation_scores(axes[0, 1], analysis_result["region_results"])
    
    # 3. 클러스터 통계
    _plot_cluster_statistics(axes[1, 0], analysis_result["region_results"])
    
    # 4. 전체 요약
    _plot_overall_summary(axes[1, 1], analysis_result["overall_summary"])
    
    plt.tight_layout()
    plt.show()


def _plot_image_with_detections(ax, image_rgb: np.ndarray, region_results: List[Dict]):
    """원본 이미지에 감지 결과 표시"""
    ax.imshow(image_rgb)
    ax.set_title("감지된 의류 아이템")
    ax.axis('off')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(region_results)))
    
    for i, region_result in enumerate(region_results):
        detection = region_result["detection"]
        bbox = detection["bbox"]
        class_name = detection["class_name"]
        confidence = detection["confidence"]
        saturation_score = region_result["saturation_score"]
        
        # 바운딩 박스
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=VISUALIZATION_CONFIG["bbox_thickness"],
            edgecolor=colors[i],
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # 라벨
        label = f"{class_name}\\n{confidence:.2f}\\n점수: {saturation_score:.2f}"
        ax.text(x1, y1-10, label, fontsize=8, color=colors[i], 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))


def _plot_saturation_scores(ax, region_results: List[Dict]):
    """대표 채도 분포 시각화"""
    if not region_results:
        ax.text(0.5, 0.5, "분석된 영역 없음", transform=ax.transAxes, ha='center')
        ax.set_title("대표 채도 분포")
        return
    
    # 데이터 준비
    class_names = [result["detection"]["class_name"] for result in region_results]
    representative_saturations = []
    for result in region_results:
        color_analysis = result["color_analysis"]
        rep_sat = color_analysis["representative_saturation"]
        representative_saturations.append(rep_sat)
    
    # 색상 설정 (채도에 따라)
    colors = ['blue' if sat < 85 else 'orange' if sat < 170 else 'red' for sat in representative_saturations]
    
    # 막대 그래프
    bars = ax.bar(range(len(representative_saturations)), representative_saturations, color=colors, alpha=0.7)
    
    # 설정
    ax.set_title("대표 채도 분포 (클러스터별 절삭평균 최대값)")
    ax.set_ylabel("대표 채도 (0-255)")
    ax.set_xlabel("의류 아이템")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.axhline(y=85, color='blue', linestyle='--', alpha=0.5, label='낮은 채도')
    ax.axhline(y=170, color='orange', linestyle='--', alpha=0.5, label='중간 채도')
    ax.legend()
    
    # 값 표시
    for bar, sat in zip(bars, representative_saturations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{sat:.0f}', ha='center', va='bottom', fontsize=8)


def _plot_cluster_statistics(ax, region_results: List[Dict]):
    """클러스터 통계 시각화"""
    if not region_results:
        ax.text(0.5, 0.5, "분석된 영역 없음", transform=ax.transAxes, ha='center')
        ax.set_title("클러스터 통계")
        return
    
    # 데이터 수집
    cluster_counts = []
    noise_ratios = []
    class_names = []
    
    for result in region_results:
        color_analysis = result["color_analysis"]
        detection = result["detection"]
        
        cluster_counts.append(color_analysis["n_clusters"])
        noise_ratios.append(color_analysis["noise_ratio"] * 100)
        class_names.append(detection["class_name"])
    
    # 이중 y축 그래프
    ax2 = ax.twinx()
    
    # 클러스터 수 (막대)
    bars = ax.bar(range(len(cluster_counts)), cluster_counts, alpha=0.6, color='skyblue', label='클러스터 수')
    
    # 노이즈 비율 (선)
    line = ax2.plot(range(len(noise_ratios)), noise_ratios, 'ro-', label='노이즈 비율(%)')
    
    # 설정
    ax.set_title("클러스터 분석 통계")
    ax.set_ylabel("클러스터 수", color='blue')
    ax2.set_ylabel("노이즈 비율 (%)", color='red')
    ax.set_xlabel("의류 아이템")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    
    # 범례
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')


def _plot_overall_summary(ax, summary: Dict):
    """전체 요약 정보 시각화"""
    if "error" in summary:
        ax.text(0.5, 0.5, f"요약 불가: {summary['error']}", transform=ax.transAxes, ha='center')
        ax.set_title("전체 요약")
        return
    
    # 텍스트 정보 구성
    rep_sats = summary.get('representative_saturations', [])
    info_text = [
        f"분석된 영역: {summary['total_regions']}개",
        "",
        "각 영역별 대표 채도:",
        "(클러스터별 절삭평균 최대값)",
        "",
    ]
    
    # 대표 채도 리스트 추가
    for i, sat in enumerate(rep_sats, 1):
        info_text.append(f"  영역 {i}: {sat:.1f}")
    
    if rep_sats:
        info_text.extend([
            "",
            f"전체 최대: {max(rep_sats):.1f}",
            f"전체 최소: {min(rep_sats):.1f}",
            f"평균: {np.mean(rep_sats):.1f}"
        ])
    
    # 텍스트 표시
    ax.text(0.1, 0.9, '\\n'.join(info_text), transform=ax.transAxes, 
            fontsize=12, verticalalignment='top', fontfamily='monospace')
    
    # 기본 배경색 (스타일 분류 제거)
    ax.set_facecolor('#f5f5f5')  # 연한 회색
    
    ax.set_title("전체 요약")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def save_visualization(analysis_result: Dict, output_path: str):
    """시각화 결과를 파일로 저장"""
    # 임시로 plt.show() 비활성화하고 저장
    plt.ioff()  # 인터랙티브 모드 비활성화
    
    visualize_analysis_results(analysis_result)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.ion()  # 인터랙티브 모드 재활성화
    print(f"시각화 결과 저장됨: {output_path}")


def visualize_detection_results(image: np.ndarray, results: List[Dict]) -> None:
    """
    YOLO 감지 결과와 색상 분석 결과를 시각화
    
    Args:
        image: RGB 이미지
        results: 분석 결과 리스트
    """
    if not results:
        print("시각화할 결과가 없습니다.")
        return
    
    # 결과 개수에 따라 서브플롯 개수 결정
    n_results = len(results)
    fig, axes = plt.subplots(2, n_results, figsize=(5*n_results, 10))
    
    # 단일 결과인 경우 axes를 2D 배열로 변환
    if n_results == 1:
        axes = axes.reshape(2, 1)
    
    for i, result in enumerate(results):
        bbox = result['bbox']
        confidence = result['confidence']
        score = result['score']
        max_sat = result['max_saturation']
        labels = result.get('labels')
        region_shape = result.get('region_shape', (100, 100, 3))
        
        x1, y1, x2, y2 = bbox
        
        # 1. 원본 이미지에 바운딩 박스 그리기
        axes[0, i].imshow(image)
        axes[0, i].set_title(f'{result["class_name"]} (Score: {score:.3f})')
        
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           fill=False, color='red', linewidth=2)
        axes[0, i].add_patch(rect)
        
        # 라벨 추가
        label_text = f'신뢰도:{confidence:.2f}\\n점수:{score:.2f}\\n절삭평균채도:{max_sat:.0f}'
        axes[0, i].text(x1, y1-10, label_text, fontsize=8, color='red', 
                    weight='bold', bbox=dict(facecolor='white', alpha=0.7))
        axes[0, i].axis('off')
        
        # 2. 클러스터링 결과 시각화
        if labels is not None:
            # 클러스터링 결과를 이미지 형태로 재구성
            height, width = region_shape[:2]
            clustered_image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # 클러스터 개수
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            # 각 클러스터에 랜덤 색상 할당
            import random
            random.seed(42)  # 일관된 색상을 위해 seed 설정
            cluster_colors = {j: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) 
                            for j in range(n_clusters)}
            noise_color = (0, 0, 0)  # 노이즈는 검은색
            
            # 픽셀별로 클러스터 색상 할당
            for idx, label in enumerate(labels):
                row = idx // width
                col = idx % width
                if row < height and col < width:
                    if label == -1:
                        clustered_image[row, col] = noise_color
                    else:
                        clustered_image[row, col] = cluster_colors[label]
            
            axes[1, i].imshow(clustered_image)
            axes[1, i].set_title(f'DBSCAN Clustering\\n({n_clusters} clusters)')
            axes[1, i].axis('off')
        else:
            axes[1, i].text(0.5, 0.5, 'No clustering\\nresult', 
                           ha='center', va='center', transform=axes[1, i].transAxes)
            axes[1, i].set_title('No Clustering')
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()