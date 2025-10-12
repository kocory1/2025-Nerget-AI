"""
Playwright 기반 Pinterest 이미지 크롤러 (학습용 주석 버전)

목표:
- 'korean outfit' 검색 결과에서 약 100장 이미지 수집
- 저장 위치: dataset/before_banana/
- 파일명: korean_output_<YYYYMMDD_HHMMSS>_<####>.jpg

설명:
- Playwright로 실제 브라우저(Chromium)를 띄워 페이지를 렌더링합니다.
- 무한 스크롤 페이지에서 이미지 <img> 태그를 계속 수집합니다.
- srcset에 여러 해상도의 이미지가 있을 수 있으므로 가장 큰 해상도 후보를 선택합니다.
- 중복 URL을 Set으로 제거한 뒤 aiohttp로 병렬 다운로드합니다.

사전 준비:
- pip install playwright aiohttp
- python -m playwright install chromium
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus, urlparse, quote
import re
import io

import aiohttp
from playwright.sync_api import sync_playwright
from PIL import Image


# 출력 베이스 디렉터리: 프로젝트 루트 기준으로 dataset/before_banana
BASE_OUT_DIR = Path(__file__).resolve().parents[2] / "dataset" / "before_banana"
BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)
URL_INDEX_FILE = BASE_OUT_DIR / "_dedup_urls.jsonl"  # 전역 URL/베이스네임 인덱스


def _best_src_from_img(img) -> str | None:
    """하나의 <img> 요소에서 가장 고해상도에 해당하는 이미지 URL을 선택합니다.

    - img.get_attribute("src"): 기본 소스
    - img.get_attribute("srcset"): "<url> <width>w, <url> <width>w, ..." 형태
    - width 가 가장 큰(=가장 고해상도) 항목을 선택합니다.
    """
    src = img.get_attribute("src") or ""
    srcset = img.get_attribute("srcset") or ""
    data_src = img.get_attribute("data-src") or ""
    data_srcset = img.get_attribute("data-srcset") or ""

    candidates: list[tuple[int, str]] = []
    if src:
        candidates.append((0, src))  # width 알 수 없으면 0으로 등록

    # srcset 파싱: 콤마로 분리 → 각 파트에서 (url, width) 추출
    for part in srcset.split(","):
        part = part.strip()
        if not part:
            continue
        pieces = part.split(" ")
        if len(pieces) >= 2:
            url, sz = pieces[0], pieces[1]
            try:
                w = int(sz.rstrip("w"))
            except Exception:
                w = 0
            candidates.append((w, url))

    # data-src 후보 추가
    if data_src:
        candidates.append((0, data_src))

    # data-srcset 파싱
    for part in data_srcset.split(","):
        part = part.strip()
        if not part:
            continue
        pieces = part.split(" ")
        if len(pieces) >= 2:
            url, sz = pieces[0], pieces[1]
            try:
                w = int(sz.rstrip("w"))
            except Exception:
                w = 0
            candidates.append((w, url))

    # currentSrc (브라우저가 선택한 실제 소스)
    try:
        curr = img.evaluate("el => el.currentSrc") or ""
        if curr:
            candidates.append((0, curr))
    except Exception:
        pass

    if not candidates:
        return None

    # width 기준 내림차순 정렬 후 최댓값 선택
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _collect_image_urls(
    page,
    *,
    limit: int = 100,
    step_px: int = 1200,
    idle_rounds_max: int = 30,
    max_rounds: int | None = None,
    debug: bool = False,
) -> list[str]:
    """페이지에서 이미지 URL을 수집합니다.

    - limit: 목표 수집 개수
    - step_px: 스크롤 한 번에 내릴 픽셀 수
    - idle_rounds_max: 새 URL 증가가 없었던 라운드가 연속으로 이 값에 도달하면 중단
    """
    urls: set[str] = set()
    idle_rounds = 0
    round_idx = 0

    while (
        len(urls) < limit
        and idle_rounds < idle_rounds_max
        and (max_rounds is None or round_idx < max_rounds)
    ):
        round_idx += 1
        # 현재 보이는 모든 핀 이미지 선택
        # Pinterest는 lazy-load를 위해 data-* 속성을 쓰는 경우가 많음. 전 img를 보고 후보를 추출한다.
        imgs = page.query_selector_all('img')

        before = len(urls)
        pinimg_found = 0
        for img in imgs:
            u = _best_src_from_img(img)
            if u and u.startswith("https://i.pinimg.com"):
                pinimg_found += 1
                urls.add(u)

        # 새롭게 추가된 URL 수 계산
        gained = len(urls) - before
        if gained == 0:
            idle_rounds += 1  # 증가분 없으면 idle로 카운트
        else:
            idle_rounds = 0

        # 아래로 스크롤해서 더 많은 핀 로드 유도
        # Playwright에서는 evaluate의 인자를 화살표 함수 매개변수로 받아야 합니다.
        try:
            page.evaluate("(s) => window.scrollBy(0, s)", step_px)
        except Exception:
            try:
                page.mouse.wheel(0, step_px)
            except Exception:
                pass

        # 주기적으로 페이지 끝으로 이동해 추가 로드를 강제
        if round_idx % 5 == 0:
            try:
                page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
            except Exception:
                try:
                    page.keyboard.press("End")
                except Exception:
                    pass

        time.sleep(1.2)  # 충분한 로딩 시간을 부여

        if debug:
            print(f"[collect] round={round_idx} imgs={len(imgs)} pinimg_in_round={pinimg_found} gained={gained} total={len(urls)} idle_rounds={idle_rounds}/{idle_rounds_max}")

    return list(urls)[:limit]


def _slugify(value: str) -> str:
    """검색어를 폴더명에 안전한 슬러그로 변환합니다.

    - 한글/영문/숫자/언더스코어/하이픈을 허용
    - 공백은 언더스코어로 치환
    - 선두/후미 구분자 정리
    """
    v = value.strip()
    # 공백 → 언더스코어
    v = re.sub(r"\s+", "_", v)
    # 허용 문자만 남김: 한글(가-힣), 영문, 숫자, 언더스코어, 하이픈
    v = re.sub(r"[^가-힣a-zA-Z0-9_\-]", "", v)
    # 연속 구분자 축약
    v = re.sub(r"[_\-]+", lambda m: m.group(0)[0], v)
    # 앞뒤 구분자 제거
    v = v.strip("_- ")
    return v or "query"


def _safe_filename_from_url(url: str) -> str:
    """URL 기반 파일명 생성.

    - 기본: URL의 path 마지막 세그먼트(대개 *.jpg)
    - 예외: 세그먼트가 없거나 너무 짧으면 URL 전체를 퍼센트 인코딩해 사용
    - 길이 제한: 180자로 절단 후 .jpg 보장
    """
    parsed = urlparse(url)
    from pathlib import Path as _P
    last = (_P(parsed.path).name or '').strip()
    if not last or '.' not in last:
        # 세그먼트가 비정상이면 URL 전체를 안전하게 인코딩
        last = quote(url, safe='')
    # 너무 길면 절단
    if len(last) > 180:
        last = last[:180]
    # 확장자 보장(없으면 .jpg)
    lower = last.lower()
    if not (lower.endswith('.jpg') or lower.endswith('.jpeg') or lower.endswith('.png') or lower.endswith('.webp')):
        last = f"{last}.jpg"
    return last


def _load_url_index_sets() -> tuple[set[str], set[str]]:
    """전역 URL 인덱스(JSONL)를 읽어 URL/베이스네임 집합을 반환."""
    urls: set[str] = set()
    basenames: set[str] = set()
    if URL_INDEX_FILE.exists():
        try:
            for line in URL_INDEX_FILE.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                u = str(rec.get("url") or "").strip()
                b = str(rec.get("url_basename") or "").strip()
                if u:
                    urls.add(u)
                if b:
                    basenames.add(b)
        except Exception:
            pass
    return urls, basenames


def _append_url_index(record: dict) -> None:
    try:
        with URL_INDEX_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _backfill_index_from_existing() -> None:
    """기존 파일들로부터 URL 베이스네임 인덱스 백필(최초 1회용)."""
    if URL_INDEX_FILE.exists():
        return
    try:
        for p in BASE_OUT_DIR.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            rec = {
                "url": None,
                "url_basename": p.name,  # 파일명이 URL의 마지막 세그먼트였던 전제
                "filename": str(p.relative_to(BASE_OUT_DIR)),
                "saved_at": datetime.now().isoformat(timespec="seconds"),
            }
            _append_url_index(rec)
    except Exception:
        pass


async def _download_all(
    urls: list[str],
    base_name: str,
    *,
    out_dir: Path,
    concurrency: int = 8,
    min_width: int | None = None,
    min_height: int | None = None,
    referer: str | None = None,
    user_agent: str | None = None,
    debug: bool = False,
) -> int:
    """수집한 이미지 URL을 병렬로 다운로드합니다.

    - concurrency: 동시에 다운로드할 작업 수(서버 과부하·차단 방지를 위해 5~10 권장)
    - 파일명 규칙: 이미지 URL에서 유도한 파일명 사용(폴더 내 동일명 존재 시 스킵)
    """
    sem = asyncio.Semaphore(concurrency)

    saved_count = 0
    skipped_exists = 0
    skipped_dup_url = 0
    skipped_small = 0
    http_non200 = 0
    open_errors = 0
    default_headers = {
        "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }
    if referer:
        default_headers["Referer"] = referer
    if user_agent:
        default_headers["User-Agent"] = user_agent

    # 전역 URL 인덱스 로드 (URL만 기준으로 중복 제거)
    url_set, basename_set = _load_url_index_sets()
    lock = asyncio.Lock()

    async with aiohttp.ClientSession(headers=default_headers) as session:
        tasks: list[asyncio.Task] = []

        for url in urls:
            filename = _safe_filename_from_url(url)
            out_path = out_dir / filename

            async def runner(u=url, p=out_path):
                async with sem:
                    try:
                        # 동일 이름 파일이 이미 있으면 스킵하여 중복 회피
                        if p.exists():
                            nonlocal skipped_exists
                            skipped_exists += 1
                            if debug:
                                print(f"[skip] exists {p.name}")
                            return
                        async with session.get(u, timeout=20) as resp:
                            if resp.status == 200:
                                data = await resp.read()
                                # URL/베이스네임 중복 검사 후 저장
                                url_basename = _safe_filename_from_url(u)
                                async with lock:
                                    # URL 동일 여부만으로 전역 중복 제거
                                    if u in url_set:
                                        nonlocal skipped_dup_url
                                        skipped_dup_url += 1
                                        if debug:
                                            print(f"[skip] dup-url {url_basename}")
                                        return
                                    # 최소 해상도 검증
                                    try:
                                        with Image.open(io.BytesIO(data)) as im:
                                            w, h = im.size
                                        if (min_width and w < min_width) or (min_height and h < min_height):
                                            skipped_small += 1
                                            if debug:
                                                print(f"[skip] too-small {url_basename} ({w}x{h})")
                                            return
                                    except Exception:
                                        open_errors += 1
                                        if debug:
                                            print(f"[skip] open-error {url_basename}")
                                        return
                                    # 저장 및 인덱스 업데이트
                                    p.write_bytes(data)
                                    url_set.add(u)
                                    basename_set.add(url_basename)
                                    _append_url_index({
                                        "url": u,
                                        "url_basename": url_basename,
                                        "filename": str(p.relative_to(BASE_OUT_DIR)),
                                        "saved_at": datetime.now().isoformat(timespec="seconds"),
                                    })
                                    nonlocal saved_count
                                    saved_count += 1
                                    if debug:
                                        print(f"[save] {p.name}")
                            else:
                                http_non200 += 1
                                if debug:
                                    print(f"[skip] http{resp.status} {_safe_filename_from_url(u)}")
                    except Exception:
                        # 네트워크 오류 등은 조용히 스킵 (운영 시엔 로깅 권장)
                        if debug:
                            print(f"[skip] exception {_safe_filename_from_url(u)}")
                        pass

            tasks.append(asyncio.create_task(runner()))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    if debug:
        print(f"[summary] saved={saved_count} exists={skipped_exists} dup_url={skipped_dup_url} small={skipped_small} http!=200={http_non200} open_err={open_errors}")
    return saved_count

def _width_hint_from_url(url: str) -> int | None:
    """Pinterest 이미지 URL 경로에 포함된 폭 힌트(예: /736x/)를 추출합니다.
    찾지 못하면 None.
    """
    m = re.search(r"/(\d+)x/", url)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def main(
    query: str = "korean outfit",
    *,
    limit: int = 100,
    headless: bool = True,
    min_width: int = 0,
    min_height: int = 0,
    step_px: int = 1600,
    idle_rounds_max: int = 80,
    max_rounds: int | None = None,
    allow_only_jpeg: bool = False,
    debug: bool = False,
) -> None:
    """엔드투엔드 실행 진입점.

    - query: Pinterest 검색어 (예: 'korean outfit')
    - limit: 목표 수집 개수 (기본 100)
    - headless: 브라우저 창 표시 여부(True=숨김)
    """
    # 기존 파일 인덱스 백필(최초 1회)
    _backfill_index_from_existing()

    with sync_playwright() as pw:
        # Chromium 브라우저 실행
        if debug:
            print(f"[init] launch browser headless={headless}")
        browser = pw.chromium.launch(headless=headless)

        # 컨텍스트 설정: 로캘/UA를 지정하면 결과가 더 안정적일 수 있습니다.
        ua = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
        context = browser.new_context(
            locale="ko-KR",
            user_agent=ua,
            viewport={"width": 1366, "height": 900},
        )

        page = context.new_page()
        if debug:
            try:
                vp = page.viewport_size
            except Exception:
                vp = None
            print(f"[init] new page opened, viewport={vp}")

        # Pinterest 검색 URL로 이동
        search_url = f"https://www.pinterest.com/search/pins/?q={quote_plus(query)}"
        if debug:
            print(f"[nav] goto {search_url}")
        page.goto(search_url, wait_until="networkidle")
        try:
            page.wait_for_load_state("networkidle", timeout=5000)
        except Exception:
            pass

        # 쿠키/가입 유도 모달 닫기 시도
        try:
            for sel in [
                'button:has-text("Accept")',
                'button:has-text("모두 허용")',
                'button:has-text("동의")',
                'button[aria-label="Close"]',
                'div[role="dialog"] button',
            ]:
                if page.locator(sel).first.is_visible(timeout=500):
                    page.locator(sel).first.click(timeout=1500)
                    if debug:
                        print(f"[ui] clicked modal/button selector: {sel}")
        except Exception:
            pass

        # 이미지 URL 수집 루프
        urls = _collect_image_urls(
            page,
            limit=limit,
            step_px=step_px,
            idle_rounds_max=idle_rounds_max,
            max_rounds=max_rounds,
            debug=debug,
        )

        browser.close()

    # 힌트 필터 제거: 수집한 URL 전체를 그대로 사용
    filtered_urls = urls
    if debug:
        print(f"[filter] collected={len(urls)} after_hint_filter={len(filtered_urls)} (disabled)")

    # 옵션: 확장자가 .jpg/.jpeg 인 URL만 허용
    if allow_only_jpeg:
        before_cnt = len(filtered_urls)
        kept: list[str] = []
        for u in filtered_urls:
            try:
                path = urlparse(u).path.lower()
            except Exception:
                path = ""
            if path.endswith(".jpg") or path.endswith(".jpeg"):
                kept.append(u)
        filtered_urls = kept
        if debug:
            print(f"[filter-jpeg] before={before_cnt} after={len(filtered_urls)} (.jpg/.jpeg only)")

    # 쿼리 기반 하위 디렉터리: dataset/before_banana/<slug>
    slug = _slugify(query)
    out_dir = BASE_OUT_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    # 파일명 접두어(요구사항 유지): korean_output (현재는 URL 기반 파일명 사용)
    base_name = "korean_output"
    if debug:
        print(f"[dl] start downloads: concurrency=6 min={min_width}x{min_height} out_dir={out_dir}")
    saved_n = asyncio.run(_download_all(
        filtered_urls,
        base_name,
        out_dir=out_dir,
        concurrency=6,
        # 실제 해상도 필터는 비활성화(폭 힌트 필터만 사용)
        min_width=0,
        min_height=0,
        referer=search_url,
        user_agent=ua,
        debug=debug,
    ))
    print(f"saved {saved_n} images to {out_dir.resolve()}")


if __name__ == "__main__":
    # 기본값: 'korean outfit' 검색으로 100장 수집
    main(query="korean outfit", limit=100, headless=True)


