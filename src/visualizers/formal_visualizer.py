"""
Formal results visualizer
- Draw bounding boxes with per-box formal scores (-1/0/1)
- Show overall image-level formal score
"""

from typing import List, Dict, Optional
import os
import cv2
import matplotlib.pyplot as plt


class FormalVisualizer:
    """Visualizer for Formal pipeline outputs."""

    def visualize(
        self,
        image_path: str,
        detections: List[Dict],
        overall_score: Optional[float] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Render detections on the image with formal scores.

        Args:
            image_path: Path to image file
            detections: List of detections with 'bbox', 'class_name', 'confidence', 'formal_score'
            overall_score: Optional overall formal score for the image
            save_path: If provided, save figure to this path instead of showing
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 8))
        plt.imshow(image_rgb)
        plt.axis("off")

        title = f"Formal analysis results: {os.path.basename(image_path)}"
        if overall_score is not None:
            title += f"  |  overall: {overall_score:.3f}"
        plt.title(title)

        for det in detections:
            bbox = det.get("bbox", [0, 0, 0, 0])
            class_name = det.get("class_name", "unknown")
            confidence = det.get("confidence", 0.0)
            formal_score = det.get("formal_score", 0)

            x1, y1, x2, y2 = bbox
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="lime", linewidth=2)
            plt.gca().add_patch(rect)

            label = f"{class_name}\nconf:{confidence:.2f}\nformal:{formal_score:+d}"
            plt.text(
                x1,
                max(0, y1 - 10),
                label,
                fontsize=9,
                color="lime",
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
            )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"Saved visualization to: {save_path}")
        else:
            plt.show()

    def visualize_from_result(self, analysis_result: Dict, save_path: Optional[str] = None) -> None:
        """Convenience wrapper for using pipeline result dict.

        Expects keys: 'image_path', 'detections', 'formal_overall_score'.
        """
        if not analysis_result.get("success", False):
            print("No successful formal result to visualize.")
            return

        image_path = analysis_result.get("image_path")
        detections = analysis_result.get("detections", [])
        overall = analysis_result.get("formal_overall_score")
        self.visualize(image_path, detections, overall_score=overall, save_path=save_path)


