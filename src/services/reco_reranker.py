"""
간단한 MMR 재랭커
- 목적: 유사도 높은 결과가 서로 너무 비슷하게 몰리지 않도록 다양성을 반영
- 입력: 쿼리 벡터, 후보 (int_id, score_ip, vector)
- 출력: 최종 상위 k 목록 (int_id, score_ip, score_mmr)

핵심 파라미터
- λ(lambda): 유사도 비중. v4가 클수록 다양성 비중↑ → λ↓
- 후보군 크기 C: v4가 클수록 후보 넓힘, 작을수록 좁힘
"""

from __future__ import annotations

import math
from typing import List, Tuple, Dict
import numpy as np


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_lambda_from_v4(v4: float) -> float:
    v4c = _clamp(v4, -1.0, 1.0)
    # λ = clamp(0.6 - 0.3*v4, 0.3, 0.9)
    return _clamp(0.6 - 0.3 * v4c, 0.3, 0.9)


def compute_candidate_count(k: int, v4: float) -> int:
    v4c = _clamp(v4, -1.0, 1.0)
    # C = clamp(ceil(k*(1 + 0.4*v4)), ceil(0.6k), ceil(1.4k))
    c = math.ceil(k * (1.0 + 0.4 * v4c))
    c = max(math.ceil(0.6 * k), min(math.ceil(1.4 * k), c))
    return max(k, c)


def mmr_rerank(
    query_vec: np.ndarray,
    candidates: List[Tuple[int, float, np.ndarray]],  # (int_id, score_ip, vec)
    k: int,
    lam: float,
) -> List[Tuple[int, float, float]]:
    # 간단 구현: 반복적으로 가장 높은 MMR 점수의 항목을 하나씩 선택
    selected: List[Tuple[int, float, float]] = []
    selected_vecs: List[np.ndarray] = []

    # 정규화된 벡터를 전제(FAISS 입력과 동일한 전처리)
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)

    pool = candidates[:]
    for _ in range(k):
        best_idx = -1
        best_score = -1e9
        for i, (iid, ip, v) in enumerate(pool):
            v_norm = v / (np.linalg.norm(v) + 1e-12)
            sim_q = float(np.dot(q, v_norm))
            if selected_vecs:
                sims = [float(np.dot(v_norm / (np.linalg.norm(v_norm) + 1e-12), sv)) for sv in selected_vecs]
                sim_s = max(sims)
            else:
                sim_s = 0.0
            mmr = lam * sim_q - (1.0 - lam) * sim_s
            if mmr > best_score:
                best_score = mmr
                best_idx = i
        if best_idx == -1:
            break
        iid, ip, v = pool.pop(best_idx)
        selected.append((iid, ip, float(best_score)))
        selected_vecs.append(v / (np.linalg.norm(v) + 1e-12))
        if not pool:
            break
    return selected


