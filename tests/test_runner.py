#!/usr/bin/env python3
"""
테스트 실행 도구 - test_yolo_color_simple.py를 편리하게 실행
"""

import os
import sys
import subprocess
from pathlib import Path

def run_main_test():
    """메인 테스트 스크립트 실행"""
    # 프로젝트 루트 디렉토리로 이동
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("🎯 메인 테스트 실행 중...")
    print("=" * 50)
    
    try:
        # test_yolo_color_simple.py 실행
        result = subprocess.run([
            sys.executable, "test_yolo_color_simple.py"
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("⚠️ 경고/오류:")
            print(result.stderr)
            
        if result.returncode == 0:
            print("\n✅ 테스트 성공적으로 완료!")
        else:
            print(f"\n❌ 테스트 실패 (종료 코드: {result.returncode})")
            
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류: {e}")

def check_dependencies():
    """필요한 의존성들이 설치되어 있는지 확인"""
    required_packages = [
        "cv2", "numpy", "sklearn", "matplotlib", 
        "pandas", "scipy", "transformers", "torch"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 누락된 패키지들:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\n설치 명령: pip install -r requirements.txt")
        return False
    
    print("✅ 모든 필요한 패키지가 설치되어 있습니다.")
    return True

if __name__ == "__main__":
    print("🧪 Nerget AI 테스트 러너")
    print("=" * 30)
    
    # 의존성 확인
    if check_dependencies():
        print()
        run_main_test()
    else:
        print("\n❌ 의존성 확인 실패. 패키지를 먼저 설치해주세요.")