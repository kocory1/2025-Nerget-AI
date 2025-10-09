from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, conlist, Field
from typing import List
import numpy as np

from ...services.faiss_index_store import FaissIndexStore
from ...services.reco_reranker import (
    compute_lambda_from_v4,
    compute_candidate_count,
    mmr_rerank,
)


router = APIRouter(prefix="/reco", tags=["reco"])


class UserVectorRequest(BaseModel):
    # 4차원: v1,v2,v3(검색), v4(다양성)
    vector: conlist(float, min_length=4, max_length=4)
    k: int = Field(default=30, ge=1, le=200)


@router.get("/health")
def health():
    store = FaissIndexStore()
    store.load()
    ready = store.index is not None
    return {"ready": ready, "count": len(store.id_to_meta)}


@router.post("/reload")
def reload_index():
    store = FaissIndexStore()
    summary = store.reload_from_runs()
    return {"reloaded": True, **summary}


@router.post("/by-user-vector")
def recommend_by_user_vector(payload: UserVectorRequest):
    # 1) 인덱스 로딩
    store = FaissIndexStore()
    store.load()
    if store.index is None or store.dim is None:
        raise HTTPException(status_code=503, detail="FAISS index not ready. Call /reco/reload first.")

    v1, v2, v3, v4 = [float(x) for x in payload.vector]
    # 검색은 3D만 사용, 4번째는 0.0으로 맞춤(인덱스 차원 맞추기)
    query_vec = [v1, v2, v3, 0.0] if store.dim >= 4 else [v1, v2, v3]

    # 2) v4를 다양성 파라미터로 변환
    lam = compute_lambda_from_v4(v4)
    cand_count = compute_candidate_count(payload.k, v4)

    # 3) 1차 후보 검색(IndexFlatIP: 정규화 내적=코사인)
    top = store.search_by_vector(query_vec, topk=cand_count)
    if not top:
        return {"k": payload.k, "lambda": lam, "candidateCount": 0, "items": []}

    # 후보 벡터 모으기(간단 구현: 저장된 id_to_vector 사용, 없으면 스킵)
    cands = []
    for iid, ip in top:
        vec = store.id_to_vector.get(iid)
        if vec is None:
            # 로드 경로에서는 벡터가 비어 있을 수 있음 → 생략
            continue
        cands.append((iid, ip, vec))

    if not cands:
        # 로드만 한 경우에는 벡터가 없어 재랭킹이 어려움 → IP 점수 기준 상위 k 반환
        picked = top[: payload.k]
        items = []
        for rank, (iid, ip) in enumerate(picked, start=1):
            meta = store.id_to_meta.get(iid)
            items.append({
                "id": meta.string_id if meta else str(iid),
                "localPath": meta.local_path if meta else None,
                "score_ip": float(ip),
                "score_mmr": float(ip),
                "rank": rank,
            })
        return {"k": payload.k, "lambda": lam, "candidateCount": len(top), "items": items}

    # 4) MMR 재랭킹
    q = np.asarray(query_vec, dtype=np.float32)
    reranked = mmr_rerank(q, cands, k=payload.k, lam=lam)

    # 5) 응답 구성
    items = []
    for rank, (iid, ip, mmr_score) in enumerate(reranked, start=1):
        meta = store.id_to_meta.get(iid)
        items.append({
            "id": meta.string_id if meta else str(iid),
            "localPath": meta.local_path if meta else None,
            "score_ip": float(ip),
            "score_mmr": float(mmr_score),
            "rank": rank,
        })
    return {"k": payload.k, "lambda": lam, "candidateCount": len(top), "items": items}


