from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from src.config.settings import GENAI_CONFIG

# 설정에서 API 키/모델을 읽어 클라이언트 초기화
client = genai.Client(api_key=GENAI_CONFIG.get("api_key"))

prompt = (
    "Create a brand-new photorealistic photo of a Korean person, using the reference image only to extract the outfit’s styling. "
    "Match the garments, fit, silhouette, materials, layering, and color palette so the clothing clearly reflects the same style, "
    "but do not reproduce the original person’s identity or face. The new subject must be a different, non‑identifiable individual. "
    "Use a simple, neutral background (e.g., clean indoor wall) that complements the outfit, and keep lighting direction and contrast "
    "coherent with the reference. Avoid recognizable landmarks, logos, or watermarks. Ensure the result is unique, natural, and suitable "
    "for commercial use without copying any protected elements beyond general outfit style."
)

# 사용자가 이 경로만 수정해서 실행
IMAGE_PATH = "/Users/bagminsu/Documents/옷마카세_ai/2025-Nerget-AI/dataset/examples/제출용/10.jpeg"
image = Image.open(IMAGE_PATH)
image = image.convert("RGB")
# 가장 긴 변 1024px로 비율 유지 축소
if max(image.size) > 1024:
    image.thumbnail((1024, 1024), Image.LANCZOS)

response = client.models.generate_content(
    model=GENAI_CONFIG.get("model") or "gemini-2.5-flash-image-preview",
    contents=[prompt, image],
)

for part in response.candidates[0].content.parts:
    if part.text is not None:
        print(part.text)
    elif part.inline_data is not None:
        out = Image.open(BytesIO(part.inline_data.data))
        out.save("edited_image.png")