"""
precompute_factors.py

KOSPI + KOSDAQ 전체 종목 팩터를 계산하여 Supabase factor_scores 테이블에 저장.
Railway Cron Job으로 매일 새벽 3시 KST 자동 실행.

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

import logging
import os
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pykrx import stock as pykrx
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
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

KST = timezone(timedelta(hours=9))

# 최소 시가총액 100억 원 미만 제외
MIN_MARKET_CAP = 10_000_000_000

# Supabase upsert 배치 크기
BATCH_SIZE = 300

# pykrx 호출 간격 (초)
PYKRX_SLEEP = 0.8


# ── 유틸 ──────────────────────────────────────────────────────────────────

def last_trading_day(base: date) -> date:
    """base 기준 직전 영업일 (주말 건너뜀). 공휴일은 pykrx alternative=True 로 처리."""
    d = base - timedelta(days=1)
    while d.weekday() >= 5:   # 5=토, 6=일
        d -= timedelta(days=1)
    return d


def to_pykrx_date(d: date) -> str:
    return d.strftime("%Y%m%d")


# ── pykrx 데이터 수집 ─────────────────────────────────────────────────────

def fetch_tickers(date_str: str) -> list[str]:
    logger.info("종목 리스트 조회 중...")
    try:
        kospi  = pykrx.get_market_ticker_list(date_str, market="KOSPI")
        time.sleep(PYKRX_SLEEP)
        kosdaq = pykrx.get_market_ticker_list(date_str, market="KOSDAQ")
        tickers = list(dict.fromkeys(list(kospi) + list(kosdaq)))  # 순서 유지 + 중복 제거
        logger.info("총 %d종목 (KOSPI %d + KOSDAQ %d)", len(tickers), len(kospi), len(kosdaq))
        return tickers
    except Exception as exc:
        logger.error("종목 리스트 조회 실패: %s", exc)
        return []


def fetch_fundamentals(date_str: str) -> pd.DataFrame:
    """pykrx → PER, PBR (KRX 공식 데이터)."""
    logger.info("PER/PBR 조회 중 (%s)...", date_str)
    frames: list[pd.DataFrame] = []
    for market in ("KOSPI", "KOSDAQ"):
        try:
            df = pykrx.get_market_fundamental_by_ticker(date_str, market=market, alternative=True)
            if df is not None and not df.empty and "PER" in df.columns and "PBR" in df.columns:
                frames.append(df[["PER", "PBR"]])
            time.sleep(PYKRX_SLEEP)
        except Exception as exc:
            logger.warning("pykrx fundamental 실패 (%s): %s", market, exc)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="first")]
    # 0·음수 → NaN
    out = out.where(out > 0)
    return out


def fetch_market_caps(date_str: str) -> pd.Series:
    """pykrx → 시가총액 (원)."""
    logger.info("시가총액 조회 중 (%s)...", date_str)
    frames: list[pd.Series] = []
    for market in ("KOSPI", "KOSDAQ"):
        try:
            df = pykrx.get_market_cap_by_ticker(date_str, market=market, alternative=True)
            if df is not None and not df.empty and "시가총액" in df.columns:
                frames.append(df["시가총액"])
            time.sleep(PYKRX_SLEEP)
        except Exception as exc:
            logger.warning("pykrx 시가총액 실패 (%s): %s", market, exc)

    if not frames:
        return pd.Series(dtype=float)
    out = pd.concat(frames)
    return out[~out.index.duplicated(keep="first")]


def fetch_close_prices(date_str: str) -> pd.Series:
    """pykrx → 특정 날짜 종가 (전 시장)."""
    frames: list[pd.Series] = []
    for market in ("KOSPI", "KOSDAQ"):
        try:
            df = pykrx.get_market_ohlcv_by_ticker(date_str, market=market, alternative=True)
            if df is not None and not df.empty and "종가" in df.columns:
                frames.append(df["종가"])
            time.sleep(PYKRX_SLEEP)
        except Exception as exc:
            logger.warning("pykrx OHLCV 실패 (%s %s): %s", date_str, market, exc)

    if not frames:
        return pd.Series(dtype=float)
    out = pd.concat(frames)
    return out[~out.index.duplicated(keep="first")]


def fetch_momentum(target: date) -> pd.DataFrame:
    """1M / 3M 모멘텀 = (현재 종가 - N개월 전 종가) / N개월 전 종가."""
    logger.info("모멘텀 계산 중...")

    today_str = to_pykrx_date(target)
    date_1m   = to_pykrx_date(last_trading_day(
        date(target.year, target.month, target.day) - timedelta(days=28)
    ))
    date_3m   = to_pykrx_date(last_trading_day(
        date(target.year, target.month, target.day) - timedelta(days=84)
    ))

    price_now = fetch_close_prices(today_str)
    price_1m  = fetch_close_prices(date_1m)
    price_3m  = fetch_close_prices(date_3m)

    mom = pd.DataFrame(index=price_now.index)

    common_1m = price_now.index.intersection(price_1m.index)
    if len(common_1m):
        with np.errstate(divide="ignore", invalid="ignore"):
            mom.loc[common_1m, "momentum_1m"] = (
                (price_now.loc[common_1m] - price_1m.loc[common_1m])
                / price_1m.loc[common_1m].replace(0, np.nan)
            )

    common_3m = price_now.index.intersection(price_3m.index)
    if len(common_3m):
        with np.errstate(divide="ignore", invalid="ignore"):
            mom.loc[common_3m, "momentum_3m"] = (
                (price_now.loc[common_3m] - price_3m.loc[common_3m])
                / price_3m.loc[common_3m].replace(0, np.nan)
            )

    return mom.replace([np.inf, -np.inf], np.nan)


# ── Supabase 재무 데이터 ───────────────────────────────────────────────────

def fetch_financials(supabase, tickers: list[str], fin_year: int) -> pd.DataFrame:
    """
    kr_financials 테이블에서 ROE, GPA, debt_ratio, PSR 계산에 필요한
    재무 항목 조회. 500개씩 배치 처리.
    """
    logger.info("Supabase 재무 데이터 조회 (year=%d, %d종목)...", fin_year, len(tickers))
    COLS = "stock_code, net_income, total_equity, total_assets, total_debt, gross_profit, revenue"
    all_rows: list[dict] = []

    for i in range(0, len(tickers), 500):
        chunk = tickers[i : i + 500]
        try:
            resp = (
                supabase.table("kr_financials")
                .select(COLS)
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

    NUM_COLS = ("net_income", "total_equity", "total_assets", "total_debt", "gross_profit", "revenue")
    rows = [
        {"ticker": row["stock_code"], **{c: row.get(c) for c in NUM_COLS}}
        for row in all_rows
    ]
    df = pd.DataFrame(rows).set_index("ticker")
    df = df.apply(pd.to_numeric, errors="coerce")
    logger.info("재무 데이터 로드 완료: %d종목", len(df))
    return df


# ── 팩터 합치기 ───────────────────────────────────────────────────────────

def build_factors(
    tickers: list[str],
    fundamentals: pd.DataFrame,
    market_caps: pd.Series,
    momentum: pd.DataFrame,
    fin_df: pd.DataFrame,
) -> pd.DataFrame:
    df = pd.DataFrame(index=tickers)

    # PER, PBR (KRX 공식)
    if not fundamentals.empty:
        df["per"] = fundamentals["PER"].reindex(df.index)
        df["pbr"] = fundamentals["PBR"].reindex(df.index)

    # 시가총액
    if not market_caps.empty:
        df["market_cap"] = market_caps.reindex(df.index)

    # 모멘텀
    for col in ("momentum_1m", "momentum_3m"):
        if col in momentum.columns:
            df[col] = momentum[col].reindex(df.index)

    # 순수 재무 비율 + PSR (Supabase kr_financials)
    if not fin_df.empty:
        fin = fin_df.reindex(df.index)
        ni     = fin["net_income"]
        eq     = fin["total_equity"]
        assets = fin["total_assets"]
        debt   = fin["total_debt"]
        gp     = fin["gross_profit"]
        rev    = fin["revenue"]

        with np.errstate(divide="ignore", invalid="ignore"):
            df["roe"]        = np.where(eq > 0, ni / eq,     np.nan)
            df["gpa"]        = np.where(assets > 0, gp / assets, np.nan)
            df["debt_ratio"] = np.where(eq > 0, debt / eq,   np.nan)

        if "market_cap" in df.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                df["psr"] = np.where(rev > 0, df["market_cap"] / rev, np.nan)

    # 최소 시가총액 필터 (100억 미만 제외)
    if "market_cap" in df.columns:
        df = df[df["market_cap"].fillna(0) >= MIN_MARKET_CAP]

    return df.replace([np.inf, -np.inf], np.nan)


# ── Supabase 저장 ─────────────────────────────────────────────────────────

def upsert_factors(supabase, df: pd.DataFrame, target_date: date) -> None:
    date_iso = target_date.isoformat()
    FACTOR_COLS = ("per", "pbr", "roe", "gpa", "momentum_1m", "momentum_3m",
                   "psr", "debt_ratio", "market_cap")

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
    now_kst    = datetime.now(KST)
    target     = last_trading_day(now_kst.date())   # 직전 영업일 데이터 사용
    date_str   = to_pykrx_date(target)
    fin_year   = target.year - 1                    # 전년도 재무 (룩어헤드 바이어스 방지)

    logger.info("=== precompute_factors 시작 ===")
    logger.info("실행 시각 (KST): %s", now_kst.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("데이터 기준일: %s  /  재무 기준 연도: %d년", date_str, fin_year)

    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    # 1. 종목 리스트
    tickers = fetch_tickers(date_str)
    if not tickers:
        logger.error("종목 리스트 없음 — 종료")
        return

    # 2. PER, PBR (pykrx)
    fundamentals = fetch_fundamentals(date_str)

    # 3. 시가총액 (pykrx)
    market_caps = fetch_market_caps(date_str)

    # 4. 모멘텀 (pykrx)
    momentum = fetch_momentum(target)

    # 5. 재무 데이터 (Supabase kr_financials) — 없으면 전전년도 시도
    fin_df = fetch_financials(sb, tickers, fin_year)
    if fin_df.empty:
        logger.warning("%d년 재무 데이터 없음 → %d년으로 재시도", fin_year, fin_year - 1)
        fin_df = fetch_financials(sb, tickers, fin_year - 1)

    # 6. 팩터 계산
    factor_df = build_factors(tickers, fundamentals, market_caps, momentum, fin_df)
    logger.info("팩터 계산 완료: %d종목 (필터 후)", len(factor_df))

    # 7. Supabase 저장
    upsert_factors(sb, factor_df, target)

    logger.info("=== precompute_factors 완료 ===")


if __name__ == "__main__":
    main()
