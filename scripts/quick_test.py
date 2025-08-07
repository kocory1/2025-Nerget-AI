#!/usr/bin/env python3
"""
빠른 테스트 실행 스크립트
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """빠른 테스트 실행"""
    print("🚀 빠른 테스트 시작")
    print("=" * 30)
    
    # 프로젝트 루트로 이동
    os.chdir(project_root)
    
    # test_yolo_color_simple.py를 subprocess로 실행
    import subprocess
    import sys
    
    try:
        result = subprocess.run([sys.executable, 'test_yolo_color_simple.py'], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("⚠️ 경고/오류:")
            print(result.stderr)
    except Exception as e:
        print(f"❌ 실행 중 오류: {e}")

if __name__ == "__main__":
    main()