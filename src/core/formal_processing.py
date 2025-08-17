"""
Formal processing core
- Label-based formality scoring for detections (-1/0/1)
- Aggregation helpers (mean over qualified detections)
"""

from typing import List, Dict, Any

from ..config.labels import get_formality_score, get_formality_label


def score_detection_formality(detection: Dict[str, Any]) -> Dict[str, Any]:
    """Attach formality score/label to a single detection.

    Returns a shallow copy with `formal_score` and `formal_label`.
    """
    class_id = detection.get("class_id")
    score = get_formality_score(class_id)
    label = get_formality_label(score)
    enriched = detection.copy()
    enriched["formal_score"] = score
    enriched["formal_label"] = label
    return enriched


def analyze_formality_detections(
    detections: List[Dict[str, Any]],
    conf_threshold: float = 0.8,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Compute per-detection formal scores and aggregate a simple mean
    over detections with confidence >= conf_threshold.
    """
    # Filter by confidence threshold
    qualified = [d for d in detections if d.get("confidence", 0.0) >= conf_threshold]

    if not qualified:
        if verbose:
            print("No qualified detections for formal scoring (conf filter).")
        return {
            "analyzed": [],
            "overall": 0.0,
            "contributing": 0,
            "insufficient_evidence": True,
        }

    analyzed: List[Dict[str, Any]] = []
    for idx, d in enumerate(qualified):
        enriched = score_detection_formality(d)
        enriched.setdefault("region_id", idx)
        analyzed.append(enriched)

    scores = [a["formal_score"] for a in analyzed]
    overall = float(sum(scores) / len(scores)) if scores else 0.0

    if verbose:
        print(f"Formal aggregation (mean over {len(scores)}): {overall:.3f}")

    return {
        "analyzed": analyzed,
        "overall": overall,
        "contributing": len(analyzed),
        "insufficient_evidence": False,
    }


