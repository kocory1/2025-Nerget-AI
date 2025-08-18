import os
from typing import Dict


def validate_image_path(image_path: str) -> Dict:
	"""간단한 이미지 경로 검증 유틸. 존재하지 않으면 에러 dict 반환, 있으면 빈 dict.

	Note: 확장자 검사 등은 필요 시 확장 가능.
	"""
	if not os.path.exists(image_path):
		return {
			"error": f"이미지를 찾을 수 없습니다: {image_path}"
		}
	return {}


