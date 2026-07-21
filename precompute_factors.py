"""
precompute_factors.py

KOSPI + KOSDAQ 전체 종목 팩터를 계산하여 Supabase factor_scores 테이블에 저장.
Railway Cron Job으로 매일 새벽 3시 KST 자동 실행.

데이터 소스:
  · 종목 리스트  : DART corpCode.xml  (stock_code 있는 전체 상장 종목)
  · 유효 종목 필터: yfinance bulk download (Yahoo Finance에 데이터 있는 종목만)
  · 시가총액     : 최신 종가(yfinance) × 발행주식수(DART stockTotqySttus API)
  · 모멘텀 1M/3M: yfinance bulk download (유효 종목 필터와 동일 데이터 재사용)
  · PER, PBR    : 시가총액(yfinance) / 재무(Supabase kr_financials)
  · ROE, GPA, 부채비율, PSR : Supabase kr_financials (전년도 재무)

── Supabase factor_scores 테이블 DDL ──────────────────────────────────────
CREATE TABLE factor_scores (
    ticker       TEXT    NOT NULL,
    date         DATE    NOT NULL,
    per          FLOAT8,
    pbr          FLOAT8,
    roe          FLOAT8,
    gpa          FLOAT8,
    momentum_1m  FLOAT8,
    momentum_3m  FLOAT8,
    psr          FLOAT8,
    debt_ratio   FLOAT8,
    market_cap   FLOAT8,
    PRIMARY KEY (ticker, date)
);
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import io
import logging
import os
import time
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# ── 로깅 ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("precompute_factors.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── 설정 ──────────────────────────────────────────────────────────────────
SUPABASE_URL  = os.environ["SUPABASE_URL"]
SUPABASE_KEY  = os.environ["SUPABASE_KEY"]
DART_API_KEY  = os.environ.get("DART_API_KEY", "50b787d351a2bdb6e499293a663069be9047d462")

KST = timezone(timedelta(hours=9))

MIN_MARKET_CAP   = 10_000_000_000   # 100억 원 미만 제외
BATCH_SIZE       = 300              # Supabase upsert 배치 크기
YF_PRICE_CHUNK   = 100              # yfinance bulk download 청크 크기 (소형 유지)
YF_CHUNK_SLEEP   = 3               # 청크 간 대기 시간(초) — rate limit 방지
MOMENTUM_DAYS    = 100              # 다운로드할 과거 거래일 (3M + 여유)
DART_WORKERS     = 8               # DART API 병렬 스레드 수
DART_SHARE_CHUNK = 500             # 발행주식수 조회 청크 크기
DART_CHUNK_SLEEP = 1               # 청크 간 대기 시간(초)


# ── 유틸 ──────────────────────────────────────────────────────────────────

def last_trading_day(base: date) -> date:
    """base 직전 영업일 (주말 건너뜀)."""
    d = base - timedelta(days=1)
    while d.weekday() >= 5:   # 5=토, 6=일
        d -= timedelta(days=1)
    return d


# ── 1. 종목 리스트: DART corpCode.xml ─────────────────────────────────────

def fetch_tickers_from_dart() -> tuple[list[str], dict[str, str]]:
    """
    DART corpCode.xml 파싱 → 6자리 stock_code 보유한 전체 상장 종목 코드 반환.
    KOSPI + KOSDAQ + KONEX 포함. 이후 yfinance 유효성 검증으로 필터링.

    Returns
    -------
    tickers      : list[str]        6자리 종목 코드 목록
    corp_code_map: dict[str, str]   {stock_code: corp_code} (발행주식수 조회에 사용)
    """
    logger.info("DART corpCode.xml 에서 종목 리스트 조회 중...")
    url = f"https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={DART_API_KEY}"
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except Exception as exc:
        logger.error("DART corpCode.xml 다운로드 실패: %s", exc)
        return [], {}

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            xml_bytes = z.read("CORPCODE.xml")
        root = ET.fromstring(xml_bytes)
    except Exception as exc:
        logger.error("corpCode.xml 파싱 실패: %s", exc)
        return [], {}

    tickers: list[str] = []
    corp_code_map: dict[str, str] = {}
    for item in root.findall("list"):
        code      = (item.findtext("stock_code") or "").strip()
        corp_code = (item.findtext("corp_code")  or "").strip()
        if len(code) == 6 and code.isdigit():
            tickers.append(code)
            corp_code_map[code] = corp_code

    logger.info("DART 상장 종목: %d개", len(tickers))
    return tickers, corp_code_map


# ── 2. 유효 종목 확인 + 가격 데이터 (yfinance bulk download) ──────────────

def screen_valid_tickers(
    stock_codes: list[str], target: date
) -> tuple[list[str], pd.DataFrame]:
    """
    yfinance bulk download로 Yahoo Finance에 데이터가 있는 유효 종목 확인.
    동시에 MOMENTUM_DAYS 기간의 종가 데이터를 수집(모멘텀 계산에 재사용).

    DART 3924 종목 중 KONEX/상폐 등 Yahoo Finance 미등록 종목은 자동 제거됨.

    Returns
    -------
    valid_codes : list[str]  Yahoo Finance에서 최근 종가가 있는 종목 코드
    close_df    : pd.DataFrame  (날짜 × "코드.KS") 종가 DataFrame
    """
    logger.info("유효 종목 확인 + 가격 데이터 수집 (yfinance bulk, %d종목)...", len(stock_codes))
    start_str = (target - timedelta(days=MOMENTUM_DAYS + 30)).strftime("%Y-%m-%d")
    end_str   = (target + timedelta(days=1)).strftime("%Y-%m-%d")

    tickers_ks = [f"{c}.KS" for c in stock_codes]
    all_close: list[pd.DataFrame] = []

    n_chunks = (len(tickers_ks) + YF_PRICE_CHUNK - 1) // YF_PRICE_CHUNK
    for i in range(0, len(tickers_ks), YF_PRICE_CHUNK):
        chunk_idx = i // YF_PRICE_CHUNK
        chunk = tickers_ks[i : i + YF_PRICE_CHUNK]
        if chunk_idx > 0:
            time.sleep(YF_CHUNK_SLEEP)   # rate limit 방지
        try:
            hist = yf.download(
                chunk,
                start=start_str,
                end=end_str,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if hist.empty:
                continue
            if isinstance(hist.columns, pd.MultiIndex):
                close = hist["Close"]
                if isinstance(close, pd.Series):
                    close = close.to_frame()
            else:
                # 청크 내 단일 종목만 유효한 경우 → flat 컬럼 (Open/High/Close 등)
                # 종목 식별 불가로 건너뜀
                logger.debug(
                    "yfinance flat DataFrame (chunk %d, %d종목) — 스킵",
                    chunk_idx, len(chunk),
                )
                continue
            close_valid = close.loc[:, close.iloc[-1].notna()]
            if not close_valid.empty:
                all_close.append(close_valid.dropna(how="all"))
            if chunk_idx % 10 == 9:
                logger.info(
                    "  가격 다운로드: %d/%d 청크 완료, 현재까지 유효 종목 %d개",
                    chunk_idx + 1, n_chunks,
                    sum(df.shape[1] for df in all_close),
                )
        except Exception as exc:
            logger.warning("yfinance 다운로드 실패 (chunk %d): %s", chunk_idx, exc)

    if not all_close:
        logger.error("가격 데이터를 전혀 수집하지 못함")
        return [], pd.DataFrame()

    close_all = pd.concat(all_close, axis=1)
    close_all = close_all.loc[:, ~close_all.columns.duplicated()]

    # 마지막 거래일 기준 종가가 있는 종목만 유효
    last_row = close_all.iloc[-1]
    valid_cols = last_row.index[last_row.notna()].tolist()
    close_all = close_all[valid_cols].dropna(how="all")

    valid_codes = [str(c).replace(".KS", "") for c in valid_cols]
    logger.info(
        "유효 종목: %d개 (DART %d개 중 Yahoo Finance 데이터 보유)",
        len(valid_codes), len(stock_codes),
    )
    return valid_codes, close_all


# ── 3-a. 발행주식수: DART stockTotqySttus API ─────────────────────────────

def fetch_shares_from_dart(
    corp_code_map: dict[str, str],
    valid_codes: list[str],
    bsns_year: int,
) -> pd.Series:
    """
    DART 주식총수 현황 API → 종목별 보통주 발행주식총수 반환.

    bsns_year 연도 사업보고서(reprt_code=11011) 우선, 없으면 전년도로 fallback.
    DART_WORKERS 스레드로 병렬 호출, DART_SHARE_CHUNK 단위로 끊어서 처리.
    """
    logger.info("DART 발행주식수 조회 중 (%d종목, %d년 기준)...", len(valid_codes), bsns_year)
    base_url = "https://opendart.fss.or.kr/api/stockTotqySttus.json"

    def _fetch_one(stock_code: str) -> tuple[str, int]:
        corp_code = corp_code_map.get(stock_code)
        if not corp_code:
            return stock_code, 0
        for year in (bsns_year, bsns_year - 1):
            try:
                resp = requests.get(
                    base_url,
                    params={
                        "crtfc_key": DART_API_KEY,
                        "corp_code": corp_code,
                        "bsns_year": str(year),
                        "reprt_code": "11011",
                    },
                    timeout=10,
                )
                data = resp.json()
                if data.get("status") != "000":
                    continue
                items = data.get("list", [])
                # 보통주만 추출 (se 필드), 없으면 우선주·기타 제외하고 합산
                def _parse(v):
                    try:
                        return int(str(v).replace(",", ""))
                    except Exception:
                        return 0
                common = [it for it in items if (it.get("se") or "") == "보통주"]
                targets = common if common else [
                    it for it in items
                    if (it.get("se") or "") not in ("합계", "기타", "")
                ]
                total = sum(_parse(it.get("istc_totqy")) for it in targets)
                if total > 0:
                    return stock_code, total
            except Exception:
                pass
        return stock_code, 0

    shares: dict[str, int] = {}
    for i in range(0, len(valid_codes), DART_SHARE_CHUNK):
        chunk = valid_codes[i : i + DART_SHARE_CHUNK]
        if i > 0:
            time.sleep(DART_CHUNK_SLEEP)
        with ThreadPoolExecutor(max_workers=DART_WORKERS) as ex:
            futures = {ex.submit(_fetch_one, c): c for c in chunk}
            for f in as_completed(futures):
                code, count = f.result()
                if count > 0:
                    shares[code] = count
        logger.info(
            "  발행주식수 진행: %d/%d종목 완료, 수집 성공 %d종목",
            min(i + DART_SHARE_CHUNK, len(valid_codes)),
            len(valid_codes),
            len(shares),
        )

    logger.info("발행주식수 수집 완료: %d종목", len(shares))
    return pd.Series(shares, dtype=float)


# ── 3-b. 시가총액 계산: 종가 × 발행주식수 ────────────────────────────────

def compute_market_caps(
    valid_codes: list[str],
    close_df: pd.DataFrame,
    shares: pd.Series,
) -> pd.Series:
    """최신 종가(yfinance close_df) × 발행주식수(DART) = 시가총액(KRW)."""
    caps: dict[str, float] = {}
    for code in valid_codes:
        col = f"{code}.KS"
        if col not in close_df.columns:
            continue
        price_series = close_df[col].dropna()
        if price_series.empty:
            continue
        price = float(price_series.iloc[-1])
        s = shares.get(code, 0)
        if price > 0 and s > 0:
            caps[code] = price * float(s)
    logger.info("시가총액 계산 완료: %d종목", len(caps))
    return pd.Series(caps, dtype=float)


# ── 4. 모멘텀: screen_valid_tickers 에서 받은 가격 데이터 재사용 ──────────

def compute_momentum(close_df: pd.DataFrame, stock_codes: list[str]) -> pd.DataFrame:
    """
    이미 다운로드한 종가 DataFrame에서 1M·3M 모멘텀 계산.
    close_df 컬럼은 "코드.KS" 형식.
    """
    if close_df.empty:
        return pd.DataFrame()

    close_ks = [f"{c}.KS" for c in stock_codes]
    available = [c for c in close_ks if c in close_df.columns]
    if not available:
        return pd.DataFrame()

    close = close_df[available].dropna(how="all")
    if len(close) < 2:
        return pd.DataFrame()

    today_px = close.iloc[-1]
    idx_1m   = max(0, len(close) - 1 - 21)
    idx_3m   = max(0, len(close) - 1 - 63)
    px_1m    = close.iloc[idx_1m]
    px_3m    = close.iloc[idx_3m]

    with np.errstate(divide="ignore", invalid="ignore"):
        mom_1m = (today_px - px_1m) / px_1m.replace(0, np.nan)
        mom_3m = (today_px - px_3m) / px_3m.replace(0, np.nan)

    mom = pd.DataFrame({"momentum_1m": mom_1m, "momentum_3m": mom_3m})
    mom = mom.replace([np.inf, -np.inf], np.nan)
    mom.index = [str(t).replace(".KS", "") for t in mom.index]
    logger.info("모멘텀 계산 완료: %d종목", mom.notna().any(axis=1).sum())
    return mom


# ── 5. Supabase 재무 데이터 ───────────────────────────────────────────────

def fetch_financials(supabase, stock_codes: list[str], fin_year: int) -> pd.DataFrame:
    """
    kr_financials 에서 PER·PBR·ROE·GPA·부채비율·PSR 계산에 필요한 항목 조회.
    500개씩 배치 처리.
    """
    logger.info("Supabase 재무 데이터 조회 (year=%d, %d종목)...", fin_year, len(stock_codes))
    SEL = "stock_code,net_income,total_equity,total_assets,total_debt,gross_profit,revenue"
    NUM = ("net_income", "total_equity", "total_assets", "total_debt", "gross_profit", "revenue")
    all_rows: list[dict] = []

    for i in range(0, len(stock_codes), 500):
        chunk = stock_codes[i : i + 500]
        try:
            resp = (
                supabase.table("kr_financials")
                .select(SEL)
                .eq("year", fin_year)
                .in_("stock_code", chunk)
                .execute()
            )
            if resp.data:
                all_rows.extend(resp.data)
        except Exception as exc:
            logger.warning("Supabase 조회 실패 (batch %d): %s", i // 500, exc)

    if not all_rows:
        return pd.DataFrame()

    rows = [{"ticker": r["stock_code"], **{c: r.get(c) for c in NUM}} for r in all_rows]
    df = pd.DataFrame(rows).set_index("ticker")
    df = df.apply(pd.to_numeric, errors="coerce")
    logger.info("재무 데이터 로드 완료: %d종목", len(df))
    return df


# ── 6. 팩터 합치기 ────────────────────────────────────────────────────────

def build_factors(
    stock_codes: list[str],
    market_caps: pd.Series,
    momentum: pd.DataFrame,
    fin_df: pd.DataFrame,
) -> pd.DataFrame:
    df = pd.DataFrame(index=stock_codes)

    df["market_cap"] = market_caps.reindex(df.index)

    for col in ("momentum_1m", "momentum_3m"):
        if not momentum.empty and col in momentum.columns:
            df[col] = momentum[col].reindex(df.index)

    if not fin_df.empty:
        fin    = fin_df.reindex(df.index)
        ni     = fin["net_income"]
        eq     = fin["total_equity"]
        assets = fin["total_assets"]
        debt   = fin["total_debt"]
        gp     = fin["gross_profit"]
        rev    = fin["revenue"]
        mc     = df["market_cap"]

        with np.errstate(divide="ignore", invalid="ignore"):
            df["per"]        = np.where((ni > 0) & mc.notna(), mc / ni, np.nan)
            df["pbr"]        = np.where((eq > 0) & mc.notna(), mc / eq, np.nan)
            df["roe"]        = np.where(eq > 0, ni / eq, np.nan)
            df["gpa"]        = np.where(assets > 0, gp / assets, np.nan)
            df["debt_ratio"] = np.where(eq > 0, debt / eq, np.nan)
            df["psr"]        = np.where((rev > 0) & mc.notna(), mc / rev, np.nan)

    # 최소 시가총액 필터 (100억 미만 제외)
    df = df[df["market_cap"].fillna(0) >= MIN_MARKET_CAP]

    return df.replace([np.inf, -np.inf], np.nan)


# ── 7. Supabase 저장 ──────────────────────────────────────────────────────

def upsert_factors(supabase, df: pd.DataFrame, target_date: date) -> None:
    date_iso    = target_date.isoformat()
    FACTOR_COLS = ("per", "pbr", "roe", "gpa", "momentum_1m",
                   "momentum_3m", "psr", "debt_ratio", "market_cap")

    rows: list[dict] = []
    for ticker, row in df.iterrows():
        record: dict = {"ticker": str(ticker), "date": date_iso}
        for col in FACTOR_COLS:
            val = row.get(col) if col in row.index else None
            if val is None or not pd.notna(val):
                record[col] = None
            elif col == "market_cap":
                record[col] = int(float(val))
            else:
                record[col] = round(float(val), 6)
        rows.append(record)

    logger.info("Supabase upsert 시작: %d종목 / %s", len(rows), date_iso)
    ok = fail = 0
    for i in range(0, len(rows), BATCH_SIZE):
        chunk = rows[i : i + BATCH_SIZE]
        try:
            supabase.table("factor_scores").upsert(
                chunk, on_conflict="ticker,date"
            ).execute()
            ok += len(chunk)
        except Exception as exc:
            logger.error("upsert 실패 (batch %d): %s", i // BATCH_SIZE, exc)
            fail += len(chunk)

    logger.info("upsert 완료: 성공 %d / 실패 %d", ok, fail)


# ── 메인 ──────────────────────────────────────────────────────────────────

def main() -> None:
    now_kst  = datetime.now(KST)
    target   = last_trading_day(now_kst.date())
    fin_year = target.year - 1   # 전년도 재무 (룩어헤드 바이어스 방지)

    logger.info("=== precompute_factors 시작 ===")
    logger.info("실행 시각 (KST) : %s", now_kst.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("데이터 기준일   : %s", target.isoformat())
    logger.info("재무 기준 연도  : %d년", fin_year)

    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    # 1. 종목 리스트 (DART) + corp_code 맵
    stock_codes, corp_code_map = fetch_tickers_from_dart()
    if not stock_codes:
        logger.error("종목 리스트 없음 — 종료")
        return
    logger.info("[단계 1] DART 종목 수: %d", len(stock_codes))

    # 2. 유효 종목 확인 + 가격 데이터 (yfinance bulk download)
    #    DART 3924개 중 Yahoo Finance에 데이터 있는 종목만 추림 (KONEX/상폐 자동 제거)
    #    → 동시에 130일 종가 수집 (모멘텀 계산 + 시가총액 계산에 재사용)
    valid_codes, close_df = screen_valid_tickers(stock_codes, target)
    if not valid_codes:
        logger.error("유효 종목 없음 — 종료")
        return
    logger.info("[단계 2] Yahoo Finance 유효 종목: %d개", len(valid_codes))

    # 3. 시가총액 = 최신 종가(yfinance) × 발행주식수(DART)
    shares = fetch_shares_from_dart(corp_code_map, valid_codes, fin_year)
    logger.info("[단계 3-a] 발행주식수 수집: %d종목", len(shares))
    market_caps = compute_market_caps(valid_codes, close_df, shares)
    logger.info("[단계 3-b] 시가총액 계산: %d종목", len(market_caps))

    # 4. 시가총액 필터 (100억 이상)
    active_codes = [
        c for c in valid_codes
        if c in market_caps.index and market_caps[c] >= MIN_MARKET_CAP
    ]
    logger.info(
        "[단계 4] 시가총액 필터 후 활성 종목: %d개 (기준: %d억 이상)",
        len(active_codes), MIN_MARKET_CAP // 100_000_000,
    )
    if not active_codes:
        logger.error("활성 종목 없음 — 종료")
        return

    # 5. 모멘텀 (이미 다운로드한 가격 데이터 재사용)
    momentum = compute_momentum(close_df, active_codes)
    logger.info("[단계 5] 모멘텀 계산: %d종목", len(momentum))

    # 6. 재무 데이터 (Supabase kr_financials)
    fin_df = fetch_financials(sb, active_codes, fin_year)
    if fin_df.empty:
        logger.warning("%d년 재무 데이터 없음 → %d년으로 재시도", fin_year, fin_year - 1)
        fin_df = fetch_financials(sb, active_codes, fin_year - 1)
    logger.info("[단계 6] 재무 데이터: %d종목", len(fin_df))

    # 7. 팩터 합치기
    factor_df = build_factors(active_codes, market_caps, momentum, fin_df)
    logger.info("[단계 7] 팩터 계산 완료: %d종목", len(factor_df))

    # 8. Supabase 저장
    upsert_factors(sb, factor_df, target)

    logger.info("=== precompute_factors 완료 ===")


if __name__ == "__main__":
    main()
