import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.pipelines.maximal_pipeline import MaximalPipeline

image_path = r"dataset\maximal\0049efd51c5af1d0ef72144a2cf611dd.jpg"
pipe = MaximalPipeline(threshold=5)

res = pipe.detect_and_analyze(image_path, conf_threshold=0.4, verbose=True)  # 임계값 낮춤

if not res.get("success"):
    print("Error:", res.get("error"))
    print("total_detections:", res.get("total_detections"))
    sys.exit(1)

print("predicted_style:", res["predicted_style"])
print("maximal_score:", res["maximal_score"])
print("core_item_count:", res["core_item_count"])
print("total_detections:", res.get("total_detections"))