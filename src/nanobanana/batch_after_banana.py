from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List
from datetime import datetime

from src.nanobanana.processor import edit_and_save_with_nano_banana


def iter_images(root: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _format_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main(limit: int | None = None, *, overwrite: bool = False, src_dir: Path | None = None) -> None:
    project_root = Path(__file__).resolve().parents[2]
    before = src_dir if src_dir else project_root / "dataset" / "before_banana"
    after = project_root / "dataset" / "after_banana"
    after.mkdir(parents=True, exist_ok=True)

    # 대상 파일 선집계 및 계획 출력 (src_dir 지정 시 가독성 향상)
    images: List[Path] = list(iter_images(before))
    total = len(images)
    exists = 0
    to_process: List[Path] = []
    for img_path in images:
        expected_out = after / f"{img_path.stem}_banana.png"
        if expected_out.exists() and not overwrite:
            exists += 1
        else:
            to_process.append(img_path)
    plan_name = (before.name if before.is_dir() else str(before))
    print(f"[plan] {_format_ts()} dir={plan_name} total={total} already_done={exists} to_process={len(to_process)} overwrite={overwrite}")

    processed = 0
    failed = 0
    skipped = exists if not overwrite else 0

    for idx, img_path in enumerate(to_process, start=1):
        expected_out = after / f"{img_path.stem}_banana.png"
        try:
            out = edit_and_save_with_nano_banana(str(img_path), output_dir=str(after))
            processed += 1
            print(f"[{_format_ts()}] ok({idx}/{len(to_process)}): {img_path.name} -> {Path(out).name}")
            if limit and processed >= limit:
                print(f"[stop] limit reached: {limit}")
                break
        except Exception as e:
            failed += 1
            print(f"[{_format_ts()}] fail({idx}/{len(to_process)}): {img_path.name} ({e})")

    print(f"[done] {_format_ts()} dir={plan_name} ok={processed} skip={skipped} fail={failed}")


if __name__ == "__main__":
    # 사용: python -m src.nanobanana.batch_after_banana [limit] [--overwrite] [--dir <path>]
    arg_limit = None
    overwrite = False
    dir_arg = None
    # 간단 파서: 순서 무관, --dir 다음 인자 경로 처리
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--overwrite":
            overwrite = True
            i += 1
        elif a == "--dir":
            if i + 1 < len(args):
                try:
                    dir_arg = Path(args[i + 1])
                except Exception:
                    dir_arg = None
                i += 2
            else:
                i += 1
        else:
            try:
                arg_limit = int(a)
            except ValueError:
                pass
            i += 1
    main(limit=arg_limit, overwrite=overwrite, src_dir=dir_arg)
