"""
collect_financials.py
DART 사업보고서 → Supabase kr_financials 테이블 적재

사전 준비:
  pip install requests supabase

환경변수:
  SUPABASE_URL   Supabase 프로젝트 URL
  SUPABASE_KEY   Supabase service_role 키 (또는 anon 키)

Supabase 테이블 DDL (한 번만 실행):
  CREATE TABLE IF NOT EXISTS kr_financials (
      id              BIGSERIAL PRIMARY KEY,
      ticker          TEXT NOT NULL,          -- "005930.KS"
      stock_code      TEXT NOT NULL,          -- "005930"
      corp_code       TEXT NOT NULL,          -- DART corp_code
      company_name    TEXT,
      year            INTEGER NOT NULL,
      fs_div          TEXT,                   -- "CFS" | "OFS"
      revenue         BIGINT,                 -- 매출액 (원)
      operating_profit BIGINT,               -- 영업이익 (원)
      net_income      BIGINT,                 -- 당기순이익 (원)
      total_equity    BIGINT,                 -- 자본총계 (원)
      total_assets    BIGINT,                 -- 자산총계 (원)
      total_debt      BIGINT,                 -- 부채총계 (원)
      gross_profit    BIGINT,                 -- 매출총이익 (원)
      created_at      TIMESTAMPTZ DEFAULT NOW(),
      updated_at      TIMESTAMPTZ DEFAULT NOW(),
      UNIQUE (stock_code, year)
  );
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import io
import logging
import os
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from typing import Optional

import requests
from supabase import create_client, Client

# ── 설정 ──────────────────────────────────────────────────────────────
DART_API_KEY  = "50b787d351a2bdb6e499293a663069be9047d462"
DART_BASE_URL = "https://opendart.fss.or.kr/api"

START_YEAR = 2015
END_YEAR   = 2024

# DART API rate limit 대비 딜레이
DELAY_API   = 0.35   # 재무 API 요청 간 (초) — 약 170 req/min 이내
DELAY_CORP  = 0.15   # 종목 전환 추가 딜레이 (초)
RETRY_WAIT  = 5.0    # 429 / 서버 오류 시 재시도 대기 (초)
MAX_RETRIES = 3

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# ── 로거 ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("collect_financials.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── KR 종목 목록 (main.py 의 KR_TICKER_NAMES 동일) ────────────────────
KR_TICKER_NAMES: dict[str, str] = {
    "005930.KS": "삼성전자",
    "000660.KS": "SK하이닉스",
    "009150.KS": "삼성전기",
    "011070.KS": "LG이노텍",
    "042700.KS": "한미반도체",
    "000990.KS": "DB하이텍",
    "007660.KS": "이수페타시스",
    "066570.KS": "LG전자",
    "034220.KS": "LG디스플레이",
    "006400.KS": "삼성SDI",
    "373220.KS": "LG에너지솔루션",
    "018260.KS": "삼성에스디에스",
    "402340.KS": "SK스퀘어",
    "003670.KS": "포스코퓨처엠",
    "005380.KS": "현대차",
    "000270.KS": "기아",
    "012330.KS": "현대모비스",
    "011210.KS": "현대위아",
    "204320.KS": "만도",
    "018880.KS": "한온시스템",
    "086280.KS": "현대글로비스",
    "161390.KS": "한국타이어앤테크놀로지",
    "002350.KS": "넥센타이어",
    "073240.KS": "금호타이어",
    "064350.KS": "현대로템",
    "105560.KS": "KB금융",
    "055550.KS": "신한지주",
    "086790.KS": "하나금융지주",
    "316140.KS": "우리금융지주",
    "032830.KS": "삼성생명",
    "000810.KS": "삼성화재",
    "006800.KS": "미래에셋증권",
    "005940.KS": "NH투자증권",
    "016360.KS": "삼성증권",
    "138930.KS": "BNK금융지주",
    "139130.KS": "DGB금융지주",
    "175330.KS": "JB금융지주",
    "088350.KS": "한화생명",
    "204210.KS": "메리츠금융지주",
    "071050.KS": "한국금융지주",
    "207940.KS": "삼성바이오로직스",
    "068270.KS": "셀트리온",
    "006280.KS": "녹십자",
    "128940.KS": "한미약품",
    "000100.KS": "유한양행",
    "185750.KS": "종근당",
    "069620.KS": "대웅제약",
    "009420.KS": "한미사이언스",
    "302440.KS": "SK바이오사이언스",
    "326030.KS": "SK바이오팜",
    "003090.KS": "대웅",
    "017900.KS": "광동제약",
    "001060.KS": "JW중외제약",
    "035420.KS": "NAVER",
    "035720.KS": "카카오",
    "259960.KS": "크래프톤",
    "036570.KS": "엔씨소프트",
    "251270.KS": "넷마블",
    "377300.KS": "카카오페이",
    "017670.KS": "SK텔레콤",
    "030200.KS": "KT",
    "032640.KS": "LG유플러스",
    "012510.KS": "더존비즈온",
    "030000.KS": "제일기획",
    "051910.KS": "LG화학",
    "096770.KS": "SK이노베이션",
    "011170.KS": "롯데케미칼",
    "009830.KS": "한화솔루션",
    "011780.KS": "금호석유",
    "010950.KS": "S-Oil",
    "036460.KS": "한국가스공사",
    "015760.KS": "한국전력",
    "010060.KS": "OCI",
    "000880.KS": "한화",
    "000720.KS": "현대건설",
    "006360.KS": "GS건설",
    "047040.KS": "대우건설",
    "375500.KS": "DL이앤씨",
    "294870.KS": "HDC현대산업개발",
    "028260.KS": "삼성물산",
    "034020.KS": "두산에너빌리티",
    "009410.KS": "태영건설",
    "003070.KS": "코오롱글로벌",
    "139480.KS": "이마트",
    "023530.KS": "롯데쇼핑",
    "097950.KS": "CJ제일제당",
    "004170.KS": "신세계",
    "008770.KS": "호텔신라",
    "282330.KS": "BGF리테일",
    "271560.KS": "오리온",
    "001040.KS": "CJ",
    "004370.KS": "농심",
    "090430.KS": "아모레퍼시픽",
    "051900.KS": "LG생활건강",
    "002790.KS": "아모레G",
    "005300.KS": "롯데칠성",
    "000080.KS": "하이트진로",
    "003230.KS": "삼양식품",
    "034730.KS": "SK",
    "011200.KS": "HMM",
    "003490.KS": "대한항공",
    "004020.KS": "현대제철",
    "005490.KS": "POSCO홀딩스",
    "009540.KS": "HD현대중공업",
    "010140.KS": "삼성중공업",
    "042660.KS": "한화오션",
    "047810.KS": "한국항공우주",
    "012450.KS": "한화에어로스페이스",
}

# ── DART 계정명 → 필드 매핑 ───────────────────────────────────────────
# 같은 필드에 여러 계정명이 매핑될 경우 첫 번째로 발견된 값을 사용
ACCOUNT_MAP: dict[str, str] = {
    # 매출
    "매출액":                   "revenue",
    "영업수익":                  "revenue",
    "수익(매출액)":              "revenue",
    "매출":                     "revenue",
    "영업수익(매출액)":           "revenue",
    # 영업이익
    "영업이익":                  "operating_profit",
    "영업이익(손실)":             "operating_profit",
    # 당기순이익
    "당기순이익":                 "net_income",
    "당기순이익(손실)":            "net_income",
    "분기순이익":                 "net_income",
    "반기순이익":                 "net_income",
    "당기순이익(손실)(지배기업소유주지분)": "net_income",
    # 자기자본
    "자본총계":                   "total_equity",
    "지배기업소유주지분":           "total_equity",
    "지배기업주주지분":             "total_equity",
    "자본":                      "total_equity",
    # 총자산
    "자산총계":                   "total_assets",
    "자산":                      "total_assets",
    # 부채
    "부채총계":                   "total_debt",
    "부채":                      "total_debt",
    # 매출총이익
    "매출총이익":                  "gross_profit",
    "매출이익":                   "gross_profit",
    "매출총손익":                  "gross_profit",
}


# ── 유틸 ──────────────────────────────────────────────────────────────
def _parse_number(raw: str) -> Optional[int]:
    """DART 재무 숫자 파싱. 괄호=음수, 쉼표 제거. 원 단위 정수 반환."""
    s = raw.strip().replace(",", "")
    if not s:
        return None
    try:
        if s.startswith("(") and s.endswith(")"):
            return -int(float(s[1:-1]))
        return int(float(s))
    except ValueError:
        return None


def _dart_get(url: str, params: dict) -> Optional[dict]:
    """DART API GET 요청 (재시도 포함)."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=20)
            if resp.status_code == 429:
                logger.warning("Rate limit (429) — %d초 대기 후 재시도 (%d/%d)", RETRY_WAIT, attempt, MAX_RETRIES)
                time.sleep(RETRY_WAIT * attempt)
                continue
            if resp.status_code >= 500:
                logger.warning("서버 오류 (%d) — %d초 대기 후 재시도", resp.status_code, RETRY_WAIT)
                time.sleep(RETRY_WAIT)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.warning("요청 실패 (%d/%d): %s", attempt, MAX_RETRIES, e)
            time.sleep(RETRY_WAIT)
    return None


# ── DART corpCode 로딩 ────────────────────────────────────────────────
def load_corp_codes() -> dict[str, str]:
    """DART corpCode.xml → {6자리 종목코드: corp_code}"""
    logger.info("DART corpCode.xml 다운로드 중...")
    url = f"{DART_BASE_URL}/corpCode.xml?crtfc_key={DART_API_KEY}"
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise SystemExit(f"DART corpCode 다운로드 실패: {e}")

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        xml_bytes = z.read("CORPCODE.xml")

    root = ET.fromstring(xml_bytes)
    codes: dict[str, str] = {}
    for item in root.findall(".//list"):
        sc = (item.findtext("stock_code") or "").strip()
        cc = (item.findtext("corp_code")  or "").strip()
        if sc and cc:
            codes[sc] = cc

    logger.info("corp_code 로드 완료: %d개", len(codes))
    return codes


# ── DART 재무 조회 ────────────────────────────────────────────────────
FinRow = dict[str, Optional[int]]

def fetch_financials(corp_code: str, year: int) -> tuple[Optional[FinRow], Optional[str]]:
    """
    단일 corp_code × 연도 재무 조회.
    CFS(연결) 먼저 시도 → 없으면 OFS(별도) fallback.
    반환: (재무 데이터 dict | None, 사용된 fs_div | None)
    """
    empty: FinRow = {
        "revenue": None,
        "operating_profit": None,
        "net_income": None,
        "total_equity": None,
        "total_assets": None,
        "total_debt": None,
        "gross_profit": None,
    }

    for fs_div in ("CFS", "OFS"):
        time.sleep(DELAY_API)

        data = _dart_get(
            f"{DART_BASE_URL}/fnlttSinglAcntAll.json",
            {
                "crtfc_key":  DART_API_KEY,
                "corp_code":  corp_code,
                "bsns_year":  str(year),
                "reprt_code": "11011",   # 사업보고서
                "fs_div":     fs_div,
            },
        )
        if data is None or data.get("status") != "000":
            continue   # OFS 로 재시도

        row: FinRow = dict(empty)
        for item in data.get("list", []):
            nm    = item.get("account_nm", "").strip()
            field = ACCOUNT_MAP.get(nm)
            if field and row.get(field) is None:
                val = _parse_number(item.get("thstrm_amount", ""))
                if val is not None:
                    row[field] = val

        # 유효 데이터가 하나라도 있으면 반환
        if any(v is not None for v in row.values()):
            return row, fs_div

    return None, None


# ── Supabase upsert ───────────────────────────────────────────────────
def upsert_record(supabase: Client, record: dict) -> bool:
    try:
        supabase.table("kr_financials").upsert(
            record,
            on_conflict="stock_code,year",
        ).execute()
        return True
    except Exception as e:
        logger.error(
            "Supabase 저장 실패 (%s %s년): %s",
            record.get("stock_code"), record.get("year"), e,
        )
        return False


# ── 메인 ──────────────────────────────────────────────────────────────
def main() -> None:
    # 환경변수 체크
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise SystemExit(
            "환경변수 SUPABASE_URL / SUPABASE_KEY 를 설정해 주세요.\n"
            "  Windows: set SUPABASE_URL=https://...\n"
            "  Linux:   export SUPABASE_URL=https://..."
        )

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("Supabase 연결 완료: %s", SUPABASE_URL[:40] + "...")

    corp_code_map = load_corp_codes()

    tickers = list(KR_TICKER_NAMES.items())
    years   = list(range(START_YEAR, END_YEAR + 1))
    total   = len(tickers) * len(years)

    succeeded = skipped = failed = done = 0
    start_ts = time.time()

    print()
    print("=" * 70)
    print(f"  수집 시작: {len(tickers)}개 종목 × {len(years)}년 = 총 {total}건")
    print(f"  기간: {START_YEAR} ~ {END_YEAR}  |  DART 사업보고서 (CFS→OFS fallback)")
    print("=" * 70)
    print()

    for corp_idx, (ticker, name) in enumerate(tickers, 1):
        stock_code = ticker.split(".")[0]   # "005930.KS" → "005930"
        corp_code  = corp_code_map.get(stock_code)

        print(f"[{corp_idx:3d}/{len(tickers)}] {name} ({stock_code})")

        if not corp_code:
            logger.warning("  → corp_code 없음, 건너뜀")
            skipped += len(years)
            done    += len(years)
            print(f"  → corp_code 없음 — {len(years)}건 전체 스킵\n")
            continue

        corp_ok = corp_skip = corp_fail = 0

        for year in years:
            done += 1
            pct   = done / total * 100
            elapsed = time.time() - start_ts
            remaining = elapsed / done * (total - done) if done else 0

            fin, fs_div = fetch_financials(corp_code, year)

            if fin is None:
                skipped  += 1
                corp_skip += 1
                print(f"  {year}년  [{pct:5.1f}%]  데이터 없음")
                continue

            record = {
                "ticker":            ticker,
                "stock_code":        stock_code,
                "corp_code":         corp_code,
                "company_name":      name,
                "year":              year,
                "fs_div":            fs_div,
                "revenue":           fin["revenue"],
                "operating_profit":  fin["operating_profit"],
                "net_income":        fin["net_income"],
                "total_equity":      fin["total_equity"],
                "total_assets":      fin["total_assets"],
                "total_debt":        fin["total_debt"],
                "gross_profit":      fin["gross_profit"],
            }

            ok = upsert_record(supabase, record)

            if ok:
                succeeded  += 1
                corp_ok    += 1
                filled = sum(1 for v in fin.values() if v is not None)
                eta_str = f"{int(remaining // 60)}분 {int(remaining % 60)}초"
                print(
                    f"  {year}년  [{pct:5.1f}%]  {fs_div}  "
                    f"항목 {filled}/7  (남은 시간 약 {eta_str})"
                )
            else:
                failed    += 1
                corp_fail += 1
                print(f"  {year}년  [{pct:5.1f}%]  Supabase 저장 실패")

        print(
            f"  → {name} 완료: 저장 {corp_ok}건 / 스킵 {corp_skip}건 / 실패 {corp_fail}건\n"
        )
        time.sleep(DELAY_CORP)

    elapsed_total = time.time() - start_ts
    print()
    print("=" * 70)
    print(f"  완료!")
    print(f"  저장 성공 : {succeeded:4d}건")
    print(f"  데이터 없음: {skipped:4d}건  (미상장 연도 / 미제출 등)")
    print(f"  저장 실패 : {failed:4d}건")
    print(f"  전체 소요  : {int(elapsed_total // 60)}분 {int(elapsed_total % 60)}초")
    print("=" * 70)
    logger.info(
        "수집 완료 — 성공: %d / 스킵: %d / 실패: %d / 소요: %.0f초",
        succeeded, skipped, failed, elapsed_total,
    )


if __name__ == "__main__":
    main()
