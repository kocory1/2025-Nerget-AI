"""
간단한 FAISS 인덱스 스토어
- 역할: 인덱스 빌드/로드/저장/검색
- 엔진: IndexFlatIP (L2 정규화 → 코사인 유사도와 동일)

초심자용 요약
- CSV에서 벡터(v1,v2,v3,(v4=옵션))를 읽어 모읍니다.
- 벡터는 길이를 1로 맞추는(L2 정규화) 과정을 거칩니다.
- FAISS(IndexFlatIP)에 적재하면 빠르게 비슷한 벡터를 찾을 수 있습니다.
- 검색은 3D만 사용합니다. v4는 추천 다양성 제어(재랭킹)에서만 사용합니다.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import faiss  # type: ignore


@dataclass
class VectorMeta:
    string_id: str
    local_path: str
    version: str
    status: str
    created_at: str
    updated_at: str
    int_id: int


def _uuid_to_int64(u: str) -> int:
    """UUID 문자열을 int64로 바꿉니다(음수 회피 위해 마스크)."""
    try:
        v = uuid.UUID(u)
        return (v.int & ((1 << 63) - 1))
    except Exception:
        return (hash(u) & ((1 << 63) - 1))


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """행 단위 L2 정규화(0으로 나누기 방지)."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class FaissIndexStore:
    """
    인덱스 스토어(단일 책임: 빌드/저장/로드/검색)
    - build_from_csvs(csv_paths): CSV들로부터 인덱스를 만듭니다.
    - save()/load(): 디스크에 저장/로딩합니다.
    - search_by_vector(): 3D 질의로 최근접 이웃을 찾습니다.
    """

    def __init__(self) -> None:
        self.index: Optional[faiss.Index] = None
        self.dim: Optional[int] = None
        self.id_to_meta: Dict[int, VectorMeta] = {}
        self.id_to_vector: Dict[int, np.ndarray] = {}
        self.index_path: Path = Path("runs/faiss/index_ip.faiss")
        self.meta_path: Path = Path("runs/faiss/idmap.json")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

    # -------- 파일 탐색 --------
    def find_csvs_under(self, base_dir: Path) -> List[Path]:
        out: List[Path] = []
        if not base_dir.exists():
            return out
        for p in base_dir.rglob("vectors.csv"):
            out.append(p)
        return sorted(out)

    # -------- 빌드 --------
    def build_from_csvs(self, csv_paths: List[Path]) -> None:
        rows: List[Tuple[int, VectorMeta, np.ndarray]] = []

        for path in csv_paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                header = f.readline()
                for line in f:
                    parts = [s.strip() for s in line.strip().split(",")]
                    if len(parts) < 10:
                        continue
                    s_id, local_path = parts[0], parts[1]
                    try:
                        v1 = float(parts[2])
                        v2 = float(parts[3])
                        v3 = float(parts[4])
                        vec = [v1, v2, v3]
                        # v4는 인덱스 차원 정합을 위해 0.0으로 패드할 수 있음
                        if len(parts) >= 6:
                            try:
                                v4 = float(parts[5])
                                vec.append(v4)
                            except Exception:
                                pass
                    except Exception:
                        continue
                    version, status, created_at, updated_at = parts[6], parts[7], parts[8], parts[9]
                    int_id = _uuid_to_int64(s_id)
                    meta = VectorMeta(
                        string_id=s_id,
                        local_path=local_path,
                        version=version,
                        status=status,
                        created_at=created_at,
                        updated_at=updated_at,
                        int_id=int_id,
                    )
                    rows.append((int_id, meta, np.asarray(vec, dtype=np.float32)))

        if not rows:
            self.index = None
            self.dim = None
            self.id_to_meta.clear()
            self.id_to_vector.clear()
            return

        # 벡터 차원 맞추기(최대 차원으로 0 패딩)
        max_dim = max(v.shape[0] for _, _, v in rows)
        ids: List[int] = []
        vecs: List[np.ndarray] = []
        self.id_to_meta.clear()
        self.id_to_vector.clear()
        for int_id, meta, vec in rows:
            if vec.shape[0] < max_dim:
                vec = np.pad(vec, (0, max_dim - vec.shape[0]))
            ids.append(int_id)
            vecs.append(vec.astype(np.float32))
            self.id_to_meta[int_id] = meta
            self.id_to_vector[int_id] = vec.astype(np.float32)

        mat = np.stack(vecs).astype(np.float32)
        mat = _l2_normalize(mat)

        d = mat.shape[1]
        self.dim = d
        base = faiss.IndexFlatIP(d)
        idmap = faiss.IndexIDMap2(base)
        idmap.add_with_ids(mat, np.asarray(ids, dtype=np.int64))
        self.index = idmap

    # -------- 저장/로딩 --------
    def save(self) -> None:
        if self.index is None:
            return
        faiss.write_index(self.index, str(self.index_path))
        payload = {
            "dim": self.dim,
            "items": {
                str(i): {
                    "string_id": m.string_id,
                    "local_path": m.local_path,
                    "version": m.version,
                    "status": m.status,
                    "created_at": m.created_at,
                    "updated_at": m.updated_at,
                }
                for i, m in self.id_to_meta.items()
            },
        }
        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def load(self) -> None:
        if not self.index_path.exists() or not self.meta_path.exists():
            self.index = None
            self.dim = None
            self.id_to_meta.clear()
            self.id_to_vector.clear()
            return
        self.index = faiss.read_index(str(self.index_path))
        with self.meta_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        self.dim = int(payload.get("dim") or 3)
        self.id_to_meta.clear()
        for k, v in (payload.get("items") or {}).items():
            int_id = int(k)
            self.id_to_meta[int_id] = VectorMeta(
                string_id=v["string_id"],
                local_path=v["local_path"],
                version=v.get("version", ""),
                status=v.get("status", ""),
                created_at=v.get("created_at", ""),
                updated_at=v.get("updated_at", ""),
                int_id=int_id,
            )
        # id_to_vector는 로딩 시 복구하지 않습니다(간단 구현). 필요 시 재빌드 사용.
        self.id_to_vector.clear()

    # -------- 검색 --------
    def search_by_vector(self, vector3_or4: List[float], topk: int) -> List[Tuple[int, float]]:
        """
        벡터로 검색합니다.
        - 입력: [v1,v2,v3,(v4)] 3~4차원
        - 출력: [(int_id, ip_score)] 내적 점수는 코사인 유사도와 동일(정규화 기준)
        """
        if self.index is None or self.dim is None:
            return []
        q = np.asarray(vector3_or4, dtype=np.float32)
        if q.shape[0] < self.dim:
            q = np.pad(q, (0, self.dim - q.shape[0]))
        elif q.shape[0] > self.dim:
            q = q[: self.dim]
        q = q[None, :]
        q = _l2_normalize(q)
        scores, ids = self.index.search(q, topk)
        result: List[Tuple[int, float]] = []
        for _id, sc in zip(ids[0].tolist(), scores[0].tolist()):
            if _id == -1:
                continue
            result.append((int(_id), float(sc)))
        return result

    # -------- 유틸 --------
    def reload_from_runs(self) -> Dict:
        """runs/unified_csv/**/vectors.csv를 스캔해 인덱스를 재구성합니다."""
        csvs = self.find_csvs_under(Path("runs/unified_csv"))
        self.build_from_csvs(csvs)
        self.save()
        return {"csv_files": [str(p) for p in csvs], "count": len(self.id_to_meta)}


