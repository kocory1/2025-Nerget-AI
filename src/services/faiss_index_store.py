"""
간단한 FAISS 인덱스 스토어
- 역할: 인덱스 빌드/로드/저장/검색
- 엔진: IndexFlatIP (L2 정규화 → 코사인 유사도와 동일)

요약
- RDS MySQL(IMAGE_VECTORS)에서 v1..v4를 읽어 인덱스 빌드(RDS 우선)
- 실패/0건 시 파일 폴백: src/faiss/{index_ip.faiss,idmap.json}
- 검색은 3D만 사용, v4는 다양성(MMR) 재랭킹에만 사용
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os

import numpy as np
import faiss  # type: ignore
import aiomysql  # type: ignore

try:
    # 상대 임포트: services -> config
    from ..config.settings import DATABASE_CONFIG
except Exception:
    # 테스트/독립 실행 시 폴백(환경변수 직접 사용)
    DATABASE_CONFIG = {
        "host": os.getenv("DB_HOST", "127.0.0.1"),
        "port": int(os.getenv("DB_PORT", "3306")),
        "user": os.getenv("DB_USER", "root"),
        "password": os.getenv("DB_PASSWORD", ""),
        "database": os.getenv("DB_NAME", "nerget_ai"),
    }


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
    - save()/load(): 디스크에 저장/로딩합니다.
    - search_by_vector(): 3D 질의로 최근접 이웃을 찾습니다.
    """

    def __init__(self) -> None:
        self.index: Optional[faiss.Index] = None
        self.dim: Optional[int] = None
        self.id_to_meta: Dict[int, VectorMeta] = {}
        self.id_to_vector: Dict[int, np.ndarray] = {}
        # 고정 경로: src/faiss (환경변수 오버라이드 제거)
        faiss_dir = Path("src/faiss")
        self.index_path: Path = faiss_dir / "index_ip.faiss"
        self.meta_path: Path = faiss_dir / "idmap.json"
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

    # CSV 기반 경로 제거됨 (RDS 우선 + 파일 폴백만 유지)

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
    # CSV 리로드 유틸 제거됨



    # -------- RDS(MySQL)에서 빌드 --------
    async def build_from_mysql(self, status: str = "DONE") -> Dict:
        """
        RDS MySQL의 IMAGE_VECTORS에서 벡터(id, s3Key, v1..v4 등)를 읽어 인덱스를 구성합니다.
        - meta.local_path에는 s3Key를 저장합니다.
        - 벡터는 L2 정규화하여 IndexFlatIP로 색인합니다.
        반환: {"count": N, "dim": d}
        """
        rows: List[Tuple[int, VectorMeta, np.ndarray]] = []

        conn: Optional[aiomysql.Connection] = None
        cur: Optional[aiomysql.Cursor] = None
        try:
            conn = await aiomysql.connect(
                host=DATABASE_CONFIG["host"],
                port=int(DATABASE_CONFIG["port"]),
                user=DATABASE_CONFIG["user"],
                password=DATABASE_CONFIG["password"],
                db=DATABASE_CONFIG["database"],
                autocommit=True,
                charset="utf8mb4",
            )
            cur = await conn.cursor()
            await cur.execute(
                """
                SELECT id, s3Key, v1, v2, v3, v4, status, createdAt, updatedAt
                FROM IMAGE_VECTORS
                WHERE status=%s
                """,
                (status,),
            )
            async for s_id, s3key, v1, v2, v3, v4, st, created_at, updated_at in cur:
                try:
                    vec: List[float] = [float(v1), float(v2), float(v3)]
                    if v4 is not None:
                        try:
                            vec.append(float(v4))
                        except Exception:
                            pass
                except Exception:
                    continue

                int_id = _uuid_to_int64(str(s_id))
                meta = VectorMeta(
                    string_id=str(s_id),
                    local_path=str(s3key or ""),  # s3Key 저장
                    version="",  # 버전 컬럼이 없으므로 빈값
                    status=str(st or ""),
                    created_at=str(created_at or ""),
                    updated_at=str(updated_at or ""),
                    int_id=int_id,
                )
                rows.append((int_id, meta, np.asarray(vec, dtype=np.float32)))
        finally:
            try:
                if cur is not None:
                    await cur.close()
            except Exception:
                pass
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass

        # 로우가 없으면 비우고 종료
        if not rows:
            self.index = None
            self.dim = None
            self.id_to_meta.clear()
            self.id_to_vector.clear()
            return {"count": 0, "dim": 0}

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
            arr = vec.astype(np.float32)
            vecs.append(arr)
            self.id_to_meta[int_id] = meta
            self.id_to_vector[int_id] = arr

        mat = np.stack(vecs).astype(np.float32)
        mat = _l2_normalize(mat)

        d = mat.shape[1]
        self.dim = d
        base = faiss.IndexFlatIP(d)
        idmap = faiss.IndexIDMap2(base)
        idmap.add_with_ids(mat, np.asarray(ids, dtype=np.int64))
        self.index = idmap

        return {"count": len(self.id_to_meta), "dim": int(self.dim or 0)}


    async def rebuild_from_db_with_fallback(self) -> Dict:
        """
        1) 우선 DB에서 빌드 시도 후 저장
        2) 실패/0건이면 디스크에서 로드하여 폴백
        반환: {"source": "db"|"file", "count": N, "dim": d, ...}
        """
        try:
            summary = await self.build_from_mysql(status="DONE")
            if int(summary.get("count", 0)) > 0:
                self.save()
                return {"source": "db", **summary}
        except Exception:
            pass

        self.load()
        return {
            "source": "file",
            "count": len(self.id_to_meta),
            "dim": int(self.dim or 0),
        }

