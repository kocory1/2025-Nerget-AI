import argparse
import os
import cv2


def overlay_vector(image_path: str, vector, output_path: str) -> None:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    h, w = img.shape[:2]
    text = f"vector: [{vector[0]:.4f}, {vector[1]:.4f}, {vector[2]:.4f}]"

    # Background rectangle
    margin = 10
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(img, (margin - 5, margin - 5), (margin + tw + 5, margin + th + 5), (0, 0, 0), -1)

    # Text
    cv2.putText(img, text, (margin, margin + th), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)


def parse_args():
    ap = argparse.ArgumentParser(description="Overlay 3D vector on an image and save.")
    ap.add_argument("image", type=str, help="Path to input image")
    ap.add_argument("--vector", nargs=3, type=float, required=True, help="Vector values: colorful maximal formal")
    ap.add_argument("--out", type=str, required=True, help="Output image path")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    overlay_vector(args.image, args.vector, args.out)
    print({"output": args.out})


