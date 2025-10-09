from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image
from google import genai

from src.config.settings import GENAI_CONFIG


# 일괄 지시 프롬프트: 항상 1인 전신샷으로 생성(포즈/얼굴 변형 포함)
DEFAULT_PROMPT: str = (
    "Create a photorealistic single‑person full‑body (head‑to‑toe) photo based on the reference image."
    " Keep the outfit’s styling identical — garments, fit, silhouette, materials, layering, and color palette — and preserve the overall"
    " camera angle and lighting. Naturally replace the face with a non‑identifiable new person, and adjust the pose (e.g., arm angle,"
    " head tilt, stance/weight shift) so the subject looks balanced and fully visible from head to shoes. Use a clean, neutral background"
    " that matches the original light direction and perspective. If any watermark, logo, or overlaid text exists, remove it and realistically"
    " inpaint the underlying scene. Do not change the clothes or their fit."
    " Render the output at the highest image resolution the model allows; do not downscale."
    " The result must be unique, natural, and must not resemble any real individual."
)


def edit_and_save_with_nano_banana(
    image_path: str,
    output_dir: Optional[str] = None,
    prompt: Optional[str] = None,
    max_size_px: Optional[int] = None,
) -> str:
    """입력 이미지를 Nano Banana(Gemini)로 편집 후 지정 폴더에 저장합니다.

    Args:
        image_path: 입력 이미지 절대경로
        output_dir: 출력 디렉터리 (기본: 프로젝트 루트의 dataset/after_banana)
        prompt: 사용할 프롬프트 (기본: DEFAULT_PROMPT)
        max_size_px: 긴 변 리사이즈 한계(기본 1024)

    Returns:
        저장된 결과 이미지의 절대경로
    """
    if not image_path:
        raise ValueError("image_path가 비어있습니다.")

    in_path = Path(image_path)
    if not in_path.exists():
        raise FileNotFoundError(f"입력 이미지를 찾을 수 없습니다: {image_path}")

    # 출력 디렉토리 기본 경로: <project_root>/dataset/after_banana
    project_root = Path(__file__).resolve().parents[2]
    default_out_dir = project_root / "dataset" / "after_banana"
    out_dir = Path(output_dir) if output_dir else default_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # API 키 및 모델 설정
    api_key = GENAI_CONFIG.get("api_key")
    model_name = GENAI_CONFIG.get("model") or "gemini-2.5-flash-image-preview"
    if not api_key:
        raise ValueError("GOOGLE_API_KEY가 설정되지 않았습니다. 환경변수 또는 settings.GENAI_CONFIG를 확인하세요.")

    client = genai.Client(api_key=api_key)

    # 입력 이미지는 가능한 원본 해상도로 유지 (max_size_px가 설정된 경우에만 축소)
    image = Image.open(str(in_path)).convert("RGB")
    if isinstance(max_size_px, int) and max_size_px > 0 and max(image.size) > max_size_px:
        image.thumbnail((max_size_px, max_size_px), Image.LANCZOS)

    final_prompt = prompt or DEFAULT_PROMPT

    response = client.models.generate_content(
        model=model_name,
        contents=[final_prompt, image],
    )

    # 응답에서 첫 번째 이미지 파트 추출 후 저장
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        raise RuntimeError("응답에 candidates가 없습니다.")

    parts = getattr(candidates[0], "content", None)
    parts = getattr(parts, "parts", None) or []

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline is not None and getattr(inline, "data", None):
            out_img = Image.open(BytesIO(inline.data))
            out_path = out_dir / f"{in_path.stem}_banana.png"
            out_img.save(str(out_path))
            return str(out_path)

    raise RuntimeError("응답에서 이미지 데이터를 찾지 못했습니다.")
