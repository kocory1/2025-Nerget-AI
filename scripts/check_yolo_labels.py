#!/usr/bin/env python3
"""
YOLO 모델의 라벨 정보 확인
"""

from ultralytics import YOLO
import os

def check_yolo_labels():
    """YOLO 모델의 라벨 정보 확인"""
    
    print("🔍 YOLO 모델 라벨 정보 확인")
    print("=" * 50)
    
    # 1. 훈련된 모델 로드
    model_path = "runs/detect/train6/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return
    
    try:
        model = YOLO(model_path)
        print("✅ 모델 로드 성공")
        
        # 2. 모델 정보 확인
        print(f"\n📊 모델 정보:")
        print(f"  모델 타입: {type(model)}")
        print(f"  모델 경로: {model_path}")
        
        # 3. 클래스 정보 확인
        if hasattr(model, 'names'):
            print(f"\n🏷️ 클래스 라벨:")
            for i, name in model.names.items():
                print(f"  {i}: {name}")
        else:
            print("❌ 클래스 라벨 정보를 찾을 수 없습니다.")
        
        # 4. 모델 설정 확인
        if hasattr(model, 'model'):
            print(f"\n⚙️ 모델 설정:")
            print(f"  모델 아키텍처: {type(model.model)}")
            
            # 클래스 수 확인
            if hasattr(model.model, 'nc'):
                print(f"  클래스 수: {model.model.nc}")
            
            # 클래스 이름 확인
            if hasattr(model.model, 'names'):
                print(f"  클래스 이름: {model.model.names}")
        
        # 5. 데이터셋 정보 확인
        print(f"\n📁 데이터셋 정보:")
        print(f"  데이터 설정: ClothingParts_v1i_yolov11/data.yaml")
        
        # 6. 테스트 이미지로 확인
        print(f"\n🧪 테스트 이미지로 확인:")
        test_image = "dataset/maximal/0269be0620747ecd46cd060a240b81b0.jpg"
        
        if os.path.exists(test_image):
            results = model(test_image)
            
            for result in results:
                if hasattr(result, 'names'):
                    print(f"  감지된 클래스:")
                    for i, name in result.names.items():
                        print(f"    {i}: {name}")
                break
        else:
            print(f"  테스트 이미지를 찾을 수 없습니다: {test_image}")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_yolo_labels() 