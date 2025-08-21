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
    Compute formal score with class-level dedup and tanh smoothing.
    Steps:
    - Filter by confidence >= conf_threshold
    - Deduplicate by class_id (like maximal analyzer)
    - Sum per-class formality scores s_i in {-1, 0, +1}
    - formal_score = tanh(sum / tau), tau = max(1, unique_count/2)
    - predicted_label: sum<=0 => Casual, sum>0 => Formal
    """
    # 1) Filter by confidence
    qualified = [d for d in detections if d.get("confidence", 0.0) >= conf_threshold]

    if not qualified:
        if verbose:
            print("No qualified detections for formal scoring (conf filter).")
        return {
            "analyzed": [],
            "formal_score": 0.0,
            "sum_score": 0,
            "unique_classes": 0,
            "predicted_label": "Casual",
            "contributing": 0,
            "insufficient_evidence": True,
        }

    # 2) Per-detection annotation for visibility
    analyzed: List[Dict[str, Any]] = []
    for idx, d in enumerate(qualified):
        enriched = score_detection_formality(d)
        enriched.setdefault("region_id", idx)
        analyzed.append(enriched)

    # 3) Dedup by class_id (only count once per class)
    seen = set()
    unique_scores: List[int] = []
    for a in analyzed:
        cid = a.get("class_id")
        if cid in seen:
            continue
        seen.add(cid)
        unique_scores.append(int(a.get("formal_score", 0)))

    sum_score = int(sum(unique_scores))
    unique_count = len(unique_scores)

    # 4) Compute smoothed score (like maximal tanh smoothing)
    formal_score = compute_formal_score(sum_score, unique_count)

    # 5) Predicted label
    predicted_label = "Formal" if sum_score > 0 else "Casual"

    if verbose:
        print(f"Formal dedup unique_classes={unique_count}, sum={sum_score} -> score={formal_score:.3f} ({predicted_label})")

    return {
        "analyzed": analyzed,
        "formal_score": formal_score,
        "sum_score": sum_score,
        "unique_classes": unique_count,
        "predicted_label": predicted_label,
        "contributing": len(analyzed),
        "insufficient_evidence": False,
    }


def compute_formal_score(sum_score: int, unique_count: int) -> float:
    """
    Tanh-smoothed formal score in [-1, 1].
    - sum_score: integer sum over unique class formality scores in {-1,0,+1}
    - unique_count: number of unique classes contributing
    Smoothing factor tau = max(1, unique_count/2).
    """
    import math
    tau = max(1.0, float(unique_count) / 2.0)
    x = float(sum_score) / tau
    return float(math.tanh(x))


