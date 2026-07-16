"""
precompute_factors.py

KOSPI + KOSDAQ 전체 종목 팩터를 계산하여 Supabase factor_scores 테이블에 저장.
Railway Cron Job으로 매일 새벽 3시 KST 자동 실행.

데이터 소스:
  · 종목 리스트  : DART corpCode.xml  (stock_code 있는 전체 상장 종목)
  · 시가총액     : yfinance fast_info (ThreadPoolExecutor 병렬)
  · 모멘텀 1M/3M: yfinance bulk download
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
YF_WORKERS       = 20               # yfinance fast_info 병렬 스레드 수
YF_PRICE_CHUNK   = 500              # yfinance bulk download 청크 크기
MOMENTUM_DAYS    = 100              # 다운로드할 과거 거래일 (3M + 여유)


# ── 유틸 ──────────────────────────────────────────────────────────────────

def last_trading_day(base: date) -> date:
    """base 직전 영업일 (주말 건너뜀). 공휴일은 yfinance alternative 데이터로 처리."""
    d = base - timedelta(days=1)
    while d.weekday() >= 5:   # 5=토, 6=일
        d -= timedelta(days=1)
    return d


# ── 1. 종목 리스트: DART corpCode.xml ─────────────────────────────────────

def fetch_tickers_from_dart() -> list[str]:
    """
    DART corpCode.xml 파싱 → 6자리 stock_code 보유한 전체 상장 종목 코드 반환.
    KOSPI + KOSDAQ + KONEX 포함. 시가총액 필터로 KONEX 자연 제거.
    """
    logger.info("DART corpCode.xml 에서 종목 리스트 조회 중...")
    url = f"https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={DART_API_KEY}"
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except Exception as exc:
        logger.error("DART corpCode.xml 다운로드 실패: %s", exc)
        return []

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            xml_bytes = z.read("CORPCODE.xml")
        root = ET.fromstring(xml_bytes)
    except Exception as exc:
        logger.error("corpCode.xml 파싱 실패: %s", exc)
        return []

    tickers: list[str] = []
    for item in root.findall("list"):
        code = (item.findtext("stock_code") or "").strip()
        if len(code) == 6 and code.isdigit():
            tickers.append(code)

    logger.info("DART 상장 종목: %d개", len(tickers))
    return tickers


# ── 2. 시가총액: yfinance fast_info 병렬 조회 ─────────────────────────────

def fetch_market_caps(stock_codes: list[str]) -> pd.Series:
    """
    yfinance Ticker.fast_info.market_cap 을 ThreadPoolExecutor로 병렬 조회.
    KRW 단위. 조회 실패 종목은 NaN.
    """
    logger.info("시가총액 조회 중 (yfinance, %d종목)...", len(stock_codes))
    caps: dict[str, float] = {}

    def _get(code: str) -> tuple[str, float]:
        try:
            mc = yf.Ticker(f"{code}.KS").fast_info.market_cap
            return code, float(mc) if mc and mc > 0 else 0.0
        except Exception:
            return code, 0.0

    with ThreadPoolExecutor(max_workers=YF_WORKERS) as ex:
        futures = {ex.submit(_get, c): c for c in stock_codes}
        done = 0
        for f in as_completed(futures):
            code, mc = f.result()
            if mc > 0:
                caps[code] = mc
            done += 1
            if done % 200 == 0:
                logger.info("  시가총액 진행: %d/%d", done, len(stock_codes))

    logger.info("시가총액 수집 완료: %d종목", len(caps))
    return pd.Series(caps, dtype=float)


# ── 3. 모멘텀: yfinance bulk download ────────────────────────────────────

def fetch_momentum(stock_codes: list[str], target: date) -> pd.DataFrame:
    """
    yfinance bulk download로 1M(21거래일) / 3M(63거래일) 모멘텀 계산.
    YF_PRICE_CHUNK 단위로 나눠 다운로드 후 합산.
    """
    logger.info("모멘텀 계산 중 (yfinance, %d종목)...", len(stock_codes))
    start_str = (target - timedelta(days=MOMENTUM_DAYS + 30)).strftime("%Y-%m-%d")
    end_str   = (target + timedelta(days=1)).strftime("%Y-%m-%d")

    tickers_ks = [f"{c}.KS" for c in stock_codes]

    all_close: list[pd.DataFrame] = []
    for i in range(0, len(tickers_ks), YF_PRICE_CHUNK):
        chunk = tickers_ks[i : i + YF_PRICE_CHUNK]
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
            close = hist["Close"] if isinstance(hist.columns, pd.MultiIndex) else hist
            if isinstance(close, pd.Series):
                close = close.to_frame()
            all_close.append(close.dropna(how="all"))
        except Exception as exc:
            logger.warning("yfinance 다운로드 실패 (chunk %d): %s", i // YF_PRICE_CHUNK, exc)

    if not all_close:
        logger.warning("모멘텀: 가격 데이터 없음")
        return pd.DataFrame()

    close_all = pd.concat(all_close, axis=1)
    close_all = close_all.loc[:, ~close_all.columns.duplicated()]
    close_all = close_all.dropna(how="all")

    if len(close_all) < 2:
        return pd.DataFrame()

    # 기준: 마지막 거래일 (target 이전)
    today_px = close_all.iloc[-1]
    idx_1m   = max(0, len(close_all) - 1 - 21)
    idx_3m   = max(0, len(close_all) - 1 - 63)
    px_1m    = close_all.iloc[idx_1m]
    px_3m    = close_all.iloc[idx_3m]

    with np.errstate(divide="ignore", invalid="ignore"):
        mom_1m = (today_px - px_1m) / px_1m.replace(0, np.nan)
        mom_3m = (today_px - px_3m) / px_3m.replace(0, np.nan)

    mom = pd.DataFrame({"momentum_1m": mom_1m, "momentum_3m": mom_3m})
    mom = mom.replace([np.inf, -np.inf], np.nan)
    # 인덱스: "005930.KS" → "005930"
    mom.index = [str(t).replace(".KS", "") for t in mom.index]
    logger.info("모멘텀 계산 완료: %d종목", mom.notna().any(axis=1).sum())
    return mom


# ── 4. Supabase 재무 데이터 ───────────────────────────────────────────────

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


# ── 5. 팩터 합치기 ────────────────────────────────────────────────────────

def build_factors(
    stock_codes: list[str],
    market_caps: pd.Series,
    momentum: pd.DataFrame,
    fin_df: pd.DataFrame,
) -> pd.DataFrame:
    df = pd.DataFrame(index=stock_codes)

    # 시가총액
    df["market_cap"] = market_caps.reindex(df.index)

    # 모멘텀
    for col in ("momentum_1m", "momentum_3m"):
        if col in momentum.columns:
            df[col] = momentum[col].reindex(df.index)

    # 재무 기반 팩터 (Supabase kr_financials)
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
            # PER = 시가총액 / 당기순이익
            df["per"] = np.where((ni > 0) & mc.notna(), mc / ni, np.nan)
            # PBR = 시가총액 / 자기자본
            df["pbr"] = np.where((eq > 0) & mc.notna(), mc / eq, np.nan)
            # ROE = 당기순이익 / 자기자본
            df["roe"] = np.where(eq > 0, ni / eq, np.nan)
            # GPA = 매출총이익 / 총자산
            df["gpa"] = np.where(assets > 0, gp / assets, np.nan)
            # 부채비율 = 부채총계 / 자기자본
            df["debt_ratio"] = np.where(eq > 0, debt / eq, np.nan)
            # PSR = 시가총액 / 매출액
            df["psr"] = np.where((rev > 0) & mc.notna(), mc / rev, np.nan)

    # 최소 시가총액 필터 (100억 미만 제외)
    df = df[df["market_cap"].fillna(0) >= MIN_MARKET_CAP]

    return df.replace([np.inf, -np.inf], np.nan)


# ── 6. Supabase 저장 ──────────────────────────────────────────────────────

def upsert_factors(supabase, df: pd.DataFrame, target_date: date) -> None:
    date_iso    = target_date.isoformat()
    FACTOR_COLS = ("per", "pbr", "roe", "gpa", "momentum_1m",
                   "momentum_3m", "psr", "debt_ratio", "market_cap")

    rows: list[dict] = []
    for ticker, row in df.iterrows():
        record: dict = {"ticker": str(ticker), "date": date_iso}
        for col in FACTOR_COLS:
            val = row.get(col) if col in row.index else None
            record[col] = round(float(val), 6) if (val is not None and pd.notna(val)) else None
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
    target   = last_trading_day(now_kst.date())   # 직전 영업일
    fin_year = target.year - 1                    # 전년도 재무 (룩어헤드 바이어스 방지)

    logger.info("=== precompute_factors 시작 ===")
    logger.info("실행 시각 (KST) : %s", now_kst.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("데이터 기준일   : %s", target.isoformat())
    logger.info("재무 기준 연도  : %d년", fin_year)

    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    # 1. 종목 리스트 (DART)
    stock_codes = fetch_tickers_from_dart()
    if not stock_codes:
        logger.error("종목 리스트 없음 — 종료")
        return

    # 2. 시가총액 (yfinance)
    market_caps = fetch_market_caps(stock_codes)

    # 3. 모멘텀 (yfinance)
    # 시가총액 있는 종목만 대상으로 (존재하지 않는 티커 yfinance 호출 절약)
    active_codes = [c for c in stock_codes if c in market_caps.index and market_caps[c] >= MIN_MARKET_CAP]
    logger.info("시가총액 필터 후 활성 종목: %d개", len(active_codes))

    momentum = fetch_momentum(active_codes, target)

    # 4. 재무 데이터 (Supabase)
    fin_df = fetch_financials(sb, active_codes, fin_year)
    if fin_df.empty:
        logger.warning("%d년 재무 데이터 없음 → %d년으로 재시도", fin_year, fin_year - 1)
        fin_df = fetch_financials(sb, active_codes, fin_year - 1)

    # 5. 팩터 합치기
    factor_df = build_factors(active_codes, market_caps, momentum, fin_df)
    logger.info("팩터 계산 완료: %d종목 (필터 후)", len(factor_df))

    # 6. Supabase 저장
    upsert_factors(sb, factor_df, target)

    logger.info("=== precompute_factors 완료 ===")


if __name__ == "__main__":
    main()
