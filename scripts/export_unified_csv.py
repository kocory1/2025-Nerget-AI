from __future__ import annotations

"""
UnifiedPipeline 결과를 CSV로 내보내는 스크립트.

- 입력: --dir (이미지 디렉토리)
- 출력: --out (CSV 경로)
  - 헤더: id,localPath,v1,v2,v3,v4,version,status,createdAt,updatedAt
  - v1..v3: colorful, maximal, formal
  - v4: 예약(0.0)
 - 실패 로그: <out>.parent / logs / failed.txt
"""

import argparse
import csv
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable, List


# 프로젝트 루트 경로 추가 (src import를 위해)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def iter_image_files(target_dir: Path) -> Iterable[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    for p in sorted(target_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def ensure_header(csv_path: Path, overwrite: bool) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "id",
        "localPath",
        "v1",
        "v2",
        "v3",
        "v4",
        "version",
        "status",
        "createdAt",
        "updatedAt",
    ]
    if overwrite or not csv_path.exists() or csv_path.stat().st_size == 0:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)


def build_row(local_path: str, v1: float, v2: float, v3: float) -> List[str]:
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    # 경로 기반 안정적 UUID (재실행 시 동일 경로 → 동일 id)
    row_id = str(uuid.uuid5(uuid.NAMESPACE_URL, local_path))
    return [
        row_id,
        local_path,
        f"{v1:.6f}",
        f"{v2:.6f}",
        f"{v3:.6f}",
        f"{0.0:.6f}",  # v4 reserved
        "unified-v1",
        "DONE",
        now,
        now,
    ]


def export_dir_to_csv(
    images_dir: Path,
    out_csv: Path,
    conf_threshold: float,
    limit: int,
    overwrite: bool,
) -> None:
    from src.pipelines.unified_pipeline import UnifiedPipeline

    ensure_header(out_csv, overwrite)

    # 실패 로그 경로 구성
    logs_dir = out_csv.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    failed_path = logs_dir / "failed.txt"

    # 이미지 수집 및 제한
    files = list(iter_image_files(images_dir))
    if limit and limit > 0:
        files = files[:limit]

    if not files:
        return

    pipeline = UnifiedPipeline()

    # CSV append 모드로 기록 (헤더는 ensure_header에서 처리)
    with out_csv.open("a", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        for idx, img_path in enumerate(files, start=1):
            try:
                result = pipeline.detect_and_analyze(
                    str(img_path), conf_threshold=conf_threshold, verbose=False
                )
                scores = result.get("image_level_scores") or {}
                colorful = float(scores.get("colorful") or 0.0)
                maximal = float(scores.get("maximal") or 0.0)
                formal = float(scores.get("formal") or 0.0)

                # localPath는 프로젝트 루트 기준 상대경로 중 'dataset/...' 형태로 맞춤
                try:
                    rel = img_path.relative_to(PROJECT_ROOT)
                except Exception:
                    rel = img_path
                # 앞쪽에 'dataset/'부터 시작하도록 보정
                rel_parts = list(rel.parts)
                if "dataset" in rel_parts:
                    rel = Path(*rel_parts[rel_parts.index("dataset"):])
                local_path = str(rel).replace("\\", "/")

                row = build_row(local_path, colorful, maximal, formal)
                writer.writerow(row)
            except Exception as e:
                with failed_path.open("a", encoding="utf-8") as f_fail:
                    f_fail.write(f"{img_path}\t{e}\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="이미지 디렉토리 경로")
    ap.add_argument("--out", required=True, help="출력 CSV 경로")
    ap.add_argument("--limit", type=int, default=0, help="처리 이미지 수 제한(0=무제한)")
    ap.add_argument("--overwrite", action="store_true", help="기존 CSV 덮어쓰기")
    ap.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    images_dir = Path(args.dir)
    out_csv = Path(args.out)
    if not images_dir.exists() or not images_dir.is_dir():
        print(f"[error] 이미지 디렉토리를 찾을 수 없습니다: {images_dir}")
        return

    export_dir_to_csv(
        images_dir=images_dir,
        out_csv=out_csv,
        conf_threshold=args.conf,
        limit=args.limit,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()


