from __future__ import annotations

import io
import logging
import os
import random
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Literal

import requests

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

random.seed(42)
np.random.seed(42)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Quant Backtest API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 팩터 메타 ─────────────────────────────────────────────────────
INFO_FACTOR_KEYS: dict[str, str] = {
    "per": "trailingPE",
    "pbr": "priceToBook",
    "roe": "returnOnEquity",
    "ev_ebitda": "enterpriseToEbitda",
    "psr": "priceToSalesTrailing12Months",
    "debt_ratio": "debtToEquity",
}

HIGHER_IS_BETTER: dict[str, bool] = {
    "per": False,
    "pbr": False,
    "roe": True,
    "ev_ebitda": False,
    "gpa": True,
    "momentum_1m": True,
    "momentum_3m": True,
    "psr": False,
    "debt_ratio": False,
}

SUPPORTED_FACTORS = frozenset(HIGHER_IS_BETTER)
TOP_N = 20
MAX_WORKERS = 16

UNIVERSE_TICKERS: list[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B",
    "JPM", "JNJ", "V", "UNH", "XOM", "PG", "MA", "HD", "CVX", "MRK",
    "ABBV", "PEP", "KO", "AVGO", "COST", "WMT", "BAC", "TMO", "ACN",
    "LLY", "CSCO", "DHR", "ABT", "TXN", "NEE", "QCOM", "PM", "RTX",
    "HON", "UPS", "AMGN", "SBUX",
]

# ── 테마별 유니버스 ────────────────────────────────────────────────
THEME_TICKERS: dict[str, list[str]] = {
    "all": UNIVERSE_TICKERS,
    "semiconductor": [
        "NVDA", "AVGO", "QCOM", "AMD", "INTC", "MU",
        "AMAT", "LRCX", "KLAC", "TXN", "ON", "MCHP",
        "ADI", "MRVL", "NXPI", "SWKS", "MPWR", "TSM",
        "ASML", "WOLF",
    ],
    "tech": [
        "AAPL", "MSFT", "GOOGL", "META", "CRM", "ORCL",
        "ADBE", "NOW", "INTU", "IBM", "CSCO", "DELL",
        "HPQ", "ACN", "PLTR", "SNOW", "FTNT", "PANW",
        "ZS", "OKTA",
    ],
    "pharma_bio": [
        "JNJ", "MRK", "ABBV", "LLY", "PFE", "BMY",
        "AMGN", "GILD", "REGN", "BIIB", "VRTX", "MRNA",
        "ILMN", "ISRG", "ABT", "DHR", "TMO", "BSX",
        "MDT", "ZBH",
    ],
    "finance": [
        "JPM", "BAC", "V", "MA", "GS", "MS",
        "BLK", "WFC", "C", "AXP", "COF", "PGR",
        "MET", "PRU", "TFC", "USB", "FITB", "BK",
        "KEY", "RF",
    ],
    "energy": [
        "XOM", "CVX", "COP", "EOG", "SLB", "PSX",
        "VLO", "MPC", "HAL", "DVN", "OKE", "WMB",
        "KMI", "LNG", "HES", "MRO", "APA", "BKR",
        "PXD", "FANG",
    ],
    "consumer": [
        "AMZN", "COST", "WMT", "TGT", "HD", "LOW",
        "NKE", "SBUX", "MCD", "CMG", "DG", "DLTR",
        "ROST", "TJX", "LULU", "PG", "KO", "PEP",
        "PM", "CL",
    ],
    "auto": [
        "TSLA", "GM", "F", "TM", "HMC", "STLA",
        "RIVN", "NIO", "APTV", "BWA", "LEA", "MGA",
        "ALV", "VC", "GNTX", "LKQ", "ADNT", "SMP",
        "THRM", "MOD",
    ],
}

# ── 스트레스 테스트 구간 ────────────────────────────────────────────
STRESS_PERIODS: list[dict[str, str]] = [
    {"name": "코로나 폭락",    "start": "2020-02-19", "end": "2020-03-23"},
    {"name": "금리인상 충격",  "start": "2022-01-03", "end": "2022-10-12"},
    {"name": "금융위기",       "start": "2008-09-15", "end": "2009-03-09"},
    {"name": "닷컴버블",       "start": "2000-03-10", "end": "2002-10-09"},
]

# 테마별 캐시 (키: theme id)
_factor_cache: dict[str, pd.DataFrame] = {}

# ── KR 시장 상수 ──────────────────────────────────────────────────
# KOSPI 200 대표 50종목 (yfinance .KS 티커 → 한글 종목명)
KR_TICKER_NAMES: dict[str, str] = {
    "005930.KS": "삼성전자",
    "000660.KS": "SK하이닉스",
    "035420.KS": "NAVER",
    "005380.KS": "현대차",
    "051910.KS": "LG화학",
    "006400.KS": "삼성SDI",
    "035720.KS": "카카오",
    "068270.KS": "셀트리온",
    "105560.KS": "KB금융",
    "055550.KS": "신한지주",
    "012330.KS": "현대모비스",
    "028260.KS": "삼성물산",
    "066570.KS": "LG전자",
    "003670.KS": "포스코홀딩스",
    "096770.KS": "SK이노베이션",
    "017670.KS": "SK텔레콤",
    "030200.KS": "KT",
    "032830.KS": "삼성생명",
    "086790.KS": "하나금융지주",
    "316140.KS": "우리금융지주",
    "018260.KS": "삼성에스디에스",
    "011200.KS": "HMM",
    "010950.KS": "S-Oil",
    "009150.KS": "삼성전기",
    "042700.KS": "한미반도체",
    "000270.KS": "기아",
    "207940.KS": "삼성바이오로직스",
    "091990.KS": "셀트리온헬스케어",
    "036570.KS": "엔씨소프트",
    "251270.KS": "넷마블",
    "011070.KS": "LG이노텍",
    "009540.KS": "한국조선해양",
    "010140.KS": "삼성중공업",
    "042660.KS": "한화오션",
    "047810.KS": "한국항공우주",
    "012450.KS": "한화에어로스페이스",
    "015760.KS": "한국전력",
    "036460.KS": "한국가스공사",
    "003490.KS": "대한항공",
    "020560.KS": "아시아나항공",
    "004020.KS": "현대제철",
    "005490.KS": "POSCO홀딩스",
    "000720.KS": "현대건설",
    "006360.KS": "GS건설",
    "034730.KS": "SK",
    "034220.KS": "LG디스플레이",
    "008770.KS": "호텔신라",
    "139480.KS": "이마트",
    "004170.KS": "신세계",
    "023530.KS": "롯데쇼핑",
}

KOSPI200_TICKERS: list[str] = list(KR_TICKER_NAMES.keys())

KR_THEME_TICKERS: dict[str, list[str]] = {
    "all": KOSPI200_TICKERS,
    "semiconductor": ["005930.KS", "000660.KS", "009150.KS", "011070.KS", "042700.KS", "096770.KS"],
    "battery":       ["006400.KS", "051910.KS", "003670.KS"],
    "finance":       ["105560.KS", "055550.KS", "086790.KS", "316140.KS", "032830.KS"],
    "auto":          ["005380.KS", "000270.KS", "012330.KS"],
    "pharma_bio":    ["207940.KS", "068270.KS", "091990.KS"],
    "tech":          ["035420.KS", "035720.KS", "036570.KS", "018260.KS", "251270.KS"],
    "energy":        ["096770.KS", "010950.KS", "015760.KS", "036460.KS"],
    "defense":       ["047810.KS", "012450.KS"],
    "shipbuilding":  ["009540.KS", "010140.KS", "042660.KS"],
    "consumer":      ["139480.KS", "023530.KS", "004170.KS", "008770.KS"],
}

KR_STRESS_PERIODS: list[dict[str, str]] = [
    {"name": "코로나 폭락",   "start": "2020-01-20", "end": "2020-03-19"},
    {"name": "금리인상 충격", "start": "2022-01-03", "end": "2022-10-12"},
    {"name": "금융위기",      "start": "2008-09-15", "end": "2009-03-09"},
    {"name": "IT버블",        "start": "2000-03-10", "end": "2001-09-30"},
]

DART_API_KEY: str = os.environ.get(
    "DART_API_KEY", "50b787d351a2bdb6e499293a663069be9047d462"
)

# KR 팩터 캐시 / DART 캐시 / 유니버스 캐시
_kr_factor_cache: dict[str, pd.DataFrame] = {}
_kr_names_cache: dict[str, str] = {}   # "005930.KS" → "삼성전자"
_kr_corp_codes: dict[str, str] = {}    # "005930" → DART corp_code
_kospi_universe_cache: list[str] = []  # 시가총액 상위 200 .KS 티커 목록


def _load_dart_kr_names() -> dict[str, str]:
    """DART corpCode.xml 1회 파싱 → 종목명 + corp_code 동시 캐시"""
    global _kr_names_cache, _kr_corp_codes
    if _kr_names_cache:
        return _kr_names_cache

    url = f"https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={DART_API_KEY}"
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code != 200:
            logger.warning("DART corpCode API 오류: %d", resp.status_code)
            _kr_names_cache = dict(KR_TICKER_NAMES)
            return _kr_names_cache

        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            xml_bytes = z.read("CORPCODE.xml")

        root = ET.fromstring(xml_bytes)
        for item in root.findall(".//list"):
            sc   = (item.findtext("stock_code") or "").strip()
            name = (item.findtext("corp_name")  or "").strip()
            cc   = (item.findtext("corp_code")  or "").strip()
            if sc and name:
                _kr_names_cache[f"{sc}.KS"] = name
                _kr_names_cache[sc] = name
            if sc and cc:
                _kr_corp_codes[sc] = cc         # 재무 API 호출용

        logger.info("DART 데이터 로드 완료: 종목명 %d개 / corp_code %d개",
                    len(_kr_names_cache) // 2, len(_kr_corp_codes))
    except Exception as e:
        logger.warning("DART 로드 실패 (%s), 하드코딩 목록으로 대체", e)
        _kr_names_cache = dict(KR_TICKER_NAMES)

    return _kr_names_cache


def _parse_kr_number(raw: str) -> float | None:
    """DART 재무제표 숫자 파싱 (쉼표·괄호 처리)"""
    s = raw.strip().replace(",", "")
    if not s:
        return None
    try:
        return -float(s[1:-1]) if s.startswith("(") and s.endswith(")") else float(s)
    except ValueError:
        return None


def _fetch_dart_financials(corp_code: str) -> dict[str, float]:
    """DART 단일 회사 재무제표 조회 → 당기순이익·자기자본 반환 (단위: 백만원).
    1월 초에 전년도 사업보고서가 미제출일 수 있으므로 전전년도까지 fallback.
    """
    current_year = datetime.now().year
    # 사업보고서(11011)는 통상 3~4월에 제출되므로 전년도 → 전전년도 순서로 시도
    for year in (current_year - 1, current_year - 2):
        for fs_div in ("CFS", "OFS"):
            params = {
                "crtfc_key": DART_API_KEY,
                "corp_code": corp_code,
                "bsns_year": str(year),
                "reprt_code": "11011",   # 사업보고서
                "fs_div": fs_div,
            }
            try:
                resp = requests.get(
                    "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json",
                    params=params, timeout=15,
                )
                data = resp.json()
                if data.get("status") != "000":
                    continue

                result: dict[str, float] = {}
                for item in data.get("list", []):
                    nm  = item.get("account_nm", "")
                    val = _parse_kr_number(item.get("thstrm_amount", ""))
                    if val is None:
                        continue
                    if nm in ("당기순이익", "당기순이익(손실)", "분기순이익"):
                        result.setdefault("net_income", val)
                    elif nm in ("자본총계", "지배기업주주지분"):
                        result.setdefault("total_equity", val)

                if result:
                    return result
            except Exception as e:
                logger.debug("DART 재무 조회 실패 (corp=%s year=%d fs=%s): %s", corp_code, year, fs_div, e)

    return {}


def _load_kr_dart_factors(
    tickers: list[str],
    market_caps: dict[str, float] | None = None,
) -> pd.DataFrame:
    """DART 재무 + 시가총액으로 PER/PBR/ROE 계산 (병렬)

    market_caps: {ticker: KRW} — 사전 로드된 시가총액 (없으면 yfinance에서 재조회)
    DART 금액 단위: 백만원 → KRW = val × 1_000_000
    """
    _load_dart_kr_names()   # _kr_corp_codes 보장

    def _fetch_one(ticker: str) -> dict | None:
        base = ticker.split(".")[0]          # "005930.KS" → "005930"
        cc = _kr_corp_codes.get(base)
        if not cc:
            return None

        fin = _fetch_dart_financials(cc)
        if not fin:
            return None

        # 시가총액: 사전 제공값 우선, 없으면 yfinance 재조회
        if market_caps and ticker in market_caps:
            mc = market_caps[ticker]
        else:
            try:
                mc = float(yf.Ticker(ticker).info.get("marketCap") or 0)
            except Exception:
                mc = 0.0
        if not (np.isfinite(mc) and mc > 0):
            return None

        ni = fin.get("net_income")    # 백만원
        eq = fin.get("total_equity")  # 백만원
        row: dict = {"ticker": ticker}

        # PER: 당기순이익이 양수인 종목만 (적자 종목 제외)
        if ni is not None and ni > 0:
            per = mc / (ni * 1_000_000)
            if np.isfinite(per) and 0 < per < 1000:
                row["per"] = round(per, 2)

        # PBR/ROE: 자기자본 양수인 종목만 (자본잠식 제외)
        if eq is not None and eq > 0:
            pbr = mc / (eq * 1_000_000)
            if np.isfinite(pbr) and 0 < pbr < 100:
                row["pbr"] = round(pbr, 2)

            if ni is not None:
                roe = ni / eq           # 동일 단위 → 변환 불필요
                if np.isfinite(roe):
                    row["roe"] = round(roe, 4)

        return row if len(row) > 1 else None

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_fetch_one, t): t for t in tickers}
        for future in as_completed(futures):
            try:
                r = future.result()
                if r:
                    rows.append(r)
            except Exception as e:
                logger.debug("DART factor fetch 예외: %s", e)

    if not rows:
        return pd.DataFrame()

    rows.sort(key=lambda r: r["ticker"])
    df = pd.DataFrame(rows).set_index("ticker")
    logger.info("DART PER/PBR/ROE 로드 완료: %d종목", len(df))
    return df


def _load_kospi_universe(n: int = 200) -> list[str]:
    """pykrx로 KOSPI 시가총액 상위 N종목을 .KS 형식으로 반환 (1회 캐시).
    실패 시 하드코딩된 KOSPI200_TICKERS(50종목) fallback.
    """
    global _kospi_universe_cache
    if _kospi_universe_cache:
        return _kospi_universe_cache

    try:
        from pykrx import stock as krx_stock  # type: ignore[import-not-found]

        # 어제부터 최대 10일 소급해 가장 최근 평일 탐색
        # (오늘 오전에는 당일 데이터가 아직 없을 수 있으므로 어제 기준 시작)
        d = datetime.now() - timedelta(days=1)
        cap_df = None
        for _ in range(10):
            if d.weekday() < 5:
                date_str = d.strftime("%Y%m%d")
                cap_df = krx_stock.get_market_cap_by_ticker(date_str, market="KOSPI")
                if cap_df is not None and not cap_df.empty:
                    break
            d -= timedelta(days=1)
        if cap_df is None or cap_df.empty:
            raise ValueError("pykrx 시가총액 데이터 없음 (최근 10일 시도 실패)")

        cap_col = "시가총액" if "시가총액" in cap_df.columns else cap_df.columns[0]
        cap_df = cap_df[cap_df[cap_col] > 0].sort_values(cap_col, ascending=False, kind="stable")

        top_codes = cap_df.head(n).index.astype(str).str.zfill(6).tolist()
        _kospi_universe_cache = [f"{c}.KS" for c in top_codes]
        logger.info("KOSPI 시가총액 상위 %d종목 로드 완료 (기준일: %s)", len(_kospi_universe_cache), date_str)

    except Exception as e:
        logger.warning("KOSPI 유니버스 로드 실패 (%s), 하드코딩 50종목 사용", e)
        _kospi_universe_cache = list(KOSPI200_TICKERS)

    return _kospi_universe_cache


# ── Pydantic 모델 ─────────────────────────────────────────────────
class FactorInput(BaseModel):
    id: str
    weight: float = Field(ge=0)


class BacktestRequest(BaseModel):
    factors: list[FactorInput]
    start_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="YYYY-MM-DD")
    end_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="YYYY-MM-DD")
    interval: Literal["1d", "1wk", "1mo"] = "1d"
    theme: str = "all"  # "all" | "semiconductor" | "tech" | ... (THEME_TICKERS / KR_THEME_TICKERS 키)
    market: Literal["US", "KR"] = "US"


class MonthlyReturn(BaseModel):
    month: str
    return_: float = Field(serialization_alias="return")
    model_config = {"populate_by_name": True, "serialize_by_alias": True}


class HeatmapPoint(BaseModel):
    """연도×월 히트맵 데이터 (항상 월별 집계)."""
    year: int
    month: int
    return_: float = Field(serialization_alias="return")
    model_config = {"populate_by_name": True, "serialize_by_alias": True}


class StressTestResult(BaseModel):
    name: str
    start: str
    end: str
    portfolio_return: float
    sp500_return: float


class SectorWeight(BaseModel):
    sector: str
    count: int
    weight: float  # 0~1 fraction


class FactorCorr(BaseModel):
    factor_a: str
    factor_b: str
    correlation: float


class BacktestResponse(BaseModel):
    cagr: float
    mdd: float
    sharpe: float
    monthly_returns: list[MonthlyReturn]
    top_stocks: list[str]
    top_tickers: list[str]  # 표시명과 별개로 실제 티커 심볼 목록
    # ── 분석 섹션 ──
    heatmap_returns: list[HeatmapPoint]
    stress_tests: list[StressTestResult]
    sector_weights: list[SectorWeight]
    factor_correlations: list[FactorCorr]


# ── 기업 상세 모델 ─────────────────────────────────────────────────
class PricePoint(BaseModel):
    date: str
    close: float


class StockDetailResponse(BaseModel):
    ticker: str
    name: str
    sector: str | None = None
    industry: str | None = None
    price_history: list[PricePoint]
    per: float | None = None
    pbr: float | None = None
    roe: float | None = None
    ev_ebitda: float | None = None
    psr: float | None = None
    debt_ratio: float | None = None
    operating_margin: float | None = None
    market_cap: int | None = None
    week_52_high: float | None = None
    week_52_low: float | None = None
    current_price: float | None = None


# ── 팩터 데이터 로딩 ──────────────────────────────────────────────
def get_theme_tickers(theme: str = "all") -> list[str]:
    return list(THEME_TICKERS.get(theme, THEME_TICKERS["all"]))


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if np.isfinite(num) else None


_GP_LABELS = ("Gross Profit", "GrossProfit", "gross_profit")
_ASSET_LABELS = ("Total Assets", "TotalAssets", "total_assets")


def _extract_gpa(t: yf.Ticker, info: dict) -> float | None:
    gross = _finite_float(info.get("grossProfits"))
    assets = _finite_float(info.get("totalAssets"))

    if gross is None:
        try:
            for stmt in (t.financials, t.income_stmt):
                if stmt is None or stmt.empty:
                    continue
                for label in _GP_LABELS:
                    if label in stmt.index:
                        gross = _finite_float(stmt.loc[label].dropna().iloc[0])
                        break
                if gross is not None:
                    break
        except Exception:
            pass

    if assets is None:
        try:
            bs = t.balance_sheet
            if bs is not None and not bs.empty:
                for label in _ASSET_LABELS:
                    if label in bs.index:
                        assets = _finite_float(bs.loc[label].dropna().iloc[0])
                        break
        except Exception:
            pass

    if gross is None or assets is None or assets <= 0:
        return None
    return gross / assets


def _fetch_info_factors(ticker: str) -> dict[str, float | str]:
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
    except Exception:
        return {}

    row: dict[str, float | str] = {}

    long_name = info.get("longName") or info.get("shortName")
    if long_name:
        row["longName"] = str(long_name)

    # 업종 정보 (섹터 분산 차트용)
    sector = info.get("sector")
    if sector:
        row["sector"] = str(sector)

    for factor_id, info_key in INFO_FACTOR_KEYS.items():
        value = _finite_float(info.get(info_key))
        # 0은 yfinance가 데이터 없을 때 반환하는 무효값 — None과 동일하게 제외
        if value is not None and value != 0.0:
            row[factor_id] = value

    gpa = _extract_gpa(t, info)
    if gpa is not None:
        row["gpa"] = gpa

    # DART PER/PBR/ROE 계산에 재사용 (KR 이중 yfinance 호출 방지)
    raw_cap = _finite_float(info.get("marketCap"))
    if raw_cap and raw_cap > 0:
        row["market_cap"] = raw_cap

    return row


def _load_momentum(tickers: list[str], period: str, name: str) -> pd.Series:
    """period 기간의 모멘텀(총수익률)을 계산해 Series로 반환."""
    if not tickers:
        return pd.Series(dtype=float)

    hist = yf.download(
        tickers, period=period, interval="1d",
        auto_adjust=True, progress=False, threads=True,
    )
    if hist.empty:
        return pd.Series(index=tickers, dtype=float)

    # MultiIndex(price_type, ticker) 또는 flat 모두 처리
    if isinstance(hist.columns, pd.MultiIndex):
        close = hist["Close"]
    else:
        close = hist[["Close"]] if "Close" in hist.columns else hist

    if isinstance(close, pd.Series):
        close = close.to_frame(str(tickers[0]))

    # newer yfinance 가 Ticker 객체를 컬럼으로 반환하는 경우 대비
    close.columns = close.columns.astype(str)

    momentum: dict[str, float] = {}
    for ticker in close.columns:
        prices = close[ticker].dropna()
        if len(prices) < 2:
            continue
        ret = float(prices.iloc[-1] / prices.iloc[0] - 1)
        if np.isfinite(ret):
            momentum[ticker] = ret

    return pd.Series(momentum, name=name)


def _load_momentum_1m(tickers: list[str]) -> pd.Series:
    return _load_momentum(tickers, period="1mo", name="momentum_1m")


def _load_momentum_3m(tickers: list[str]) -> pd.Series:
    return _load_momentum(tickers, period="3mo", name="momentum_3m")


def load_factor_universe(theme: str = "all") -> pd.DataFrame:
    global _factor_cache
    if theme in _factor_cache:
        return _factor_cache[theme]

    tickers = get_theme_tickers(theme)
    rows: list[dict[str, float | str]] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_fetch_info_factors, t): t for t in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                factors = future.result()
                if factors:
                    rows.append({"ticker": ticker, **factors})
            except Exception as e:
                logger.debug("팩터 조회 예외 (%s): %s", ticker, e)

    if not rows:
        raise HTTPException(
            status_code=503,
            detail="팩터 데이터를 가져오지 못했습니다. 잠시 후 다시 시도해 주세요.",
        )

    rows.sort(key=lambda r: r["ticker"])
    df = pd.DataFrame(rows).set_index("ticker")
    mom_1m = _load_momentum_1m(tickers)
    if not mom_1m.empty:
        df["momentum_1m"] = mom_1m.reindex(df.index)
    mom_3m = _load_momentum_3m(tickers)
    if not mom_3m.empty:
        df["momentum_3m"] = mom_3m.reindex(df.index)

    _factor_cache[theme] = df
    logger.info("Loaded factor data for theme '%s': %d tickers", theme, len(df))
    return df


def _fill_kr_sector_median(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """NaN 팩터값을 섹터 중위값 → 전체 중위값 순서로 채운다.

    yfinance sector 컬럼이 있으면 섹터별로 그룹화하고,
    섹터 정보 없거나 섹터 내에도 모두 NaN이면 전체 중위값 사용.
    """
    for col in cols:
        if col not in df.columns:
            continue
        if "sector" in df.columns:
            sector_med = df.groupby("sector", observed=True)[col].transform("median")
            df[col] = df[col].fillna(sector_med)
        overall_med = df[col].median()
        if pd.notna(overall_med):
            df[col] = df[col].fillna(overall_med)
    return df


# ── KR 팩터 데이터 로딩 ───────────────────────────────────────────
def load_kr_factor_universe(theme: str = "all") -> pd.DataFrame:
    global _kr_factor_cache
    cache_key = f"kr_{theme}"
    if cache_key in _kr_factor_cache:
        return _kr_factor_cache[cache_key]

    # "all" 테마: pykrx 시가총액 상위 200종목 동적 로드
    if theme == "all":
        tickers = _load_kospi_universe(n=200)
    else:
        tickers = list(KR_THEME_TICKERS.get(theme, _load_kospi_universe(n=200)))

    rows: list[dict[str, float | str]] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_fetch_info_factors, t): t for t in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                factors = future.result()
                if factors:
                    rows.append({"ticker": ticker, **factors})
            except Exception as e:
                logger.debug("KR 팩터 조회 예외 (%s): %s", ticker, e)

    rows.sort(key=lambda r: r["ticker"])
    idx = pd.Index(tickers, name="ticker")
    df = pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame(index=idx)

    # DART 종목명 (실패 시 KR_TICKER_NAMES fallback → 티커 코드)
    dart_names = _load_dart_kr_names()
    df["longName"] = df.index.map(lambda t: dart_names.get(t) or KR_TICKER_NAMES.get(t, t))

    # _fetch_info_factors에서 수집한 시가총액 재활용 → DART 계산 시 yfinance 재호출 방지
    market_caps: dict[str, float] = {}
    if "market_cap" in df.columns:
        market_caps = {
            t: float(df.at[t, "market_cap"])
            for t in df.index
            if pd.notna(df.at[t, "market_cap"])
        }
        df.drop(columns=["market_cap"], inplace=True)

    # DART PER/PBR/ROE (우선) → yfinance fallback 보존 → 섹터 중위값 imputation
    dart_fund = _load_kr_dart_factors(tickers, market_caps=market_caps)
    for col in ("per", "pbr", "roe"):
        dart_col = (
            dart_fund[col].reindex(df.index)
            if col in dart_fund.columns
            else pd.Series(dtype=float, index=df.index)
        )
        yf_col = df[col] if col in df.columns else pd.Series(dtype=float, index=df.index)
        # DART 값 우선 적용, DART 없는 종목은 yfinance 값 유지
        df[col] = dart_col.combine_first(yf_col)

    dart_count = {
        c: int(dart_fund[c].notna().sum())
        for c in ("per", "pbr", "roe") if c in dart_fund.columns
    }
    after_merge = {c: int(df[c].notna().sum()) for c in ("per", "pbr", "roe") if c in df.columns}
    logger.info("PER/PBR/ROE — DART: %s / DART+yfinance 합산: %s", dart_count, after_merge)

    # 여전히 NaN인 종목: 섹터 중위값 → 전체 중위값 순서로 보완
    _fill_kr_sector_median(df, ["per", "pbr", "roe"])

    after_impute = {c: int(df[c].notna().sum()) for c in ("per", "pbr", "roe") if c in df.columns}
    logger.info("섹터 imputation 후 유효 종목 수: %s / 전체 %d종목", after_impute, len(df))

    # 모멘텀 (yfinance .KS) — reindex로 인덱스 불일치 시 NaN 처리
    mom_1m = _load_momentum(tickers, period="1mo", name="momentum_1m")
    if not mom_1m.empty:
        df["momentum_1m"] = mom_1m.reindex(df.index)

    mom_3m = _load_momentum(tickers, period="3mo", name="momentum_3m")
    if not mom_3m.empty:
        df["momentum_3m"] = mom_3m.reindex(df.index)

    _kr_factor_cache[cache_key] = df
    logger.info("KR 팩터 유니버스 로드 완료 (theme='%s'): %d종목", theme, len(df))
    return df


def run_kr_equal_weight_backtest(
    tickers: list[str], start: datetime, end: datetime, interval: str = "1d"
) -> tuple[pd.Series, pd.Series]:
    """KR 종목 백테스트 (yfinance .KS)"""
    return run_equal_weight_backtest(tickers, start, end, interval)


def _calc_kr_single_stress(
    tickers: list[str], period: dict[str, str]
) -> StressTestResult | None:
    try:
        all_tickers = sorted(set(tickers + ["^KS11"]))
        hist = yf.download(
            all_tickers,
            start=period["start"],
            end=period["end"],
            interval="1d",
            auto_adjust=True, progress=False, threads=True,
        )
        if hist.empty:
            return None

        close = hist["Close"] if isinstance(hist.columns, pd.MultiIndex) else hist
        if isinstance(close, pd.Series):
            close = close.to_frame(all_tickers[0])

        port_cols = [t for t in tickers if t in close.columns]
        portfolio_return: float = 0.0
        if port_cols:
            port_close = close[port_cols].dropna(how="all")
            if len(port_close) >= 2:
                ret_df = port_close.pct_change().dropna(how="all")
                portfolio_return = float((1 + ret_df.mean(axis=1)).prod() - 1)

        kospi_return: float = 0.0
        if "^KS11" in close.columns:
            kp = close["^KS11"].dropna()
            if len(kp) >= 2:
                kospi_return = float(kp.iloc[-1] / kp.iloc[0] - 1)

        return StressTestResult(
            name=period["name"],
            start=period["start"],
            end=period["end"],
            portfolio_return=portfolio_return if np.isfinite(portfolio_return) else 0.0,
            sp500_return=kospi_return if np.isfinite(kospi_return) else 0.0,
        )
    except Exception as exc:
        logger.warning("KR 스트레스 테스트 실패 (%s): %s", period["name"], exc)
        return None


def calc_kr_stress_tests(tickers: list[str]) -> list[StressTestResult]:
    results: list[StressTestResult | None] = [None] * len(KR_STRESS_PERIODS)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_calc_kr_single_stress, tickers, period): i
            for i, period in enumerate(KR_STRESS_PERIODS)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return [r for r in results if r is not None]


# ── 팩터 스코어링 ─────────────────────────────────────────────────
def percentile_score(series: pd.Series, higher_is_better: bool) -> pd.Series:
    rank = series.rank(pct=True, method="average")
    return rank if higher_is_better else 1.0 - rank


def compute_composite_scores(
    universe: pd.DataFrame, factors: list[FactorInput]
) -> pd.Series:
    factor_ids = [f.id.lower() for f in factors]
    unknown = set(factor_ids) - SUPPORTED_FACTORS
    if unknown:
        supported = ", ".join(sorted(SUPPORTED_FACTORS))
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 팩터: {', '.join(sorted(unknown))}. 사용 가능: {supported}",
        )

    if not factors:
        raise HTTPException(status_code=400, detail="최소 1개의 팩터가 필요합니다.")

    missing_columns = [fid for fid in factor_ids if fid not in universe.columns]
    if missing_columns:
        raise HTTPException(
            status_code=503,
            detail=f"팩터 데이터 없음: {', '.join(missing_columns)}",
        )

    mask = pd.Series(True, index=universe.index)
    for fid in factor_ids:
        mask &= universe[fid].notna()

    eligible = universe.loc[mask]
    if eligible.empty:
        raise HTTPException(
            status_code=503,
            detail="선택한 팩터 조합에 대해 유효한 종목이 없습니다.",
        )

    weight_sum = sum(f.weight for f in factors)
    if weight_sum <= 0:
        raise HTTPException(status_code=400, detail="팩터 가중치 합은 0보다 커야 합니다.")

    composite = pd.Series(0.0, index=eligible.index)
    for factor in factors:
        fid = factor.id.lower()
        w = factor.weight / weight_sum
        scores = percentile_score(eligible[fid], HIGHER_IS_BETTER[fid])
        composite = composite + w * scores

    # 동점 시 ticker 알파벳 오름차순으로 2차 정렬 → 항상 동일한 top_tickers 보장
    return (
        composite
        .rename_axis("ticker")
        .to_frame("score")
        .sort_values(["score", "ticker"], ascending=[False, True], kind="stable")
        ["score"]
    )


# ── 백테스트 계산 ─────────────────────────────────────────────────
_PERIODS_PER_YEAR: dict[str, float] = {"1d": 252.0, "1wk": 52.0, "1mo": 12.0}
_RESAMPLE_RULES: dict[str, str]     = {"1d": "W-FRI", "1wk": "W-FRI", "1mo": "ME"}
_DATE_FMTS: dict[str, str]          = {"1d": "%Y-%m-%d", "1wk": "%Y-%m-%d", "1mo": "%Y-%m"}


def run_equal_weight_backtest(
    tickers: list[str], start: datetime, end: datetime, interval: str = "1d"
) -> tuple[pd.Series, pd.Series]:
    if not tickers:
        raise HTTPException(status_code=400, detail="백테스트할 종목이 없습니다.")

    prices = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=True, progress=False, threads=True,
    )

    if prices.empty:
        raise HTTPException(status_code=503, detail="주가 데이터를 가져오지 못했습니다.")

    close = prices["Close"] if isinstance(prices.columns, pd.MultiIndex) else prices
    if isinstance(close, pd.Series):
        close = close.to_frame()

    close = close.dropna(axis=1, how="all").dropna(how="all")
    if close.shape[1] == 0:
        raise HTTPException(status_code=503, detail="유효한 주가 시계열이 없습니다.")

    daily_returns = close.pct_change().dropna(how="all")
    portfolio_returns = daily_returns.mean(axis=1).dropna()
    if portfolio_returns.empty:
        raise HTTPException(status_code=503, detail="포트폴리오 수익률을 계산할 수 없습니다.")

    equity = (1 + portfolio_returns).cumprod()
    return portfolio_returns, equity


def calc_cagr(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    days = (equity.index[-1] - equity.index[0]).days
    years = days / 365.0
    if years <= 0:
        return 0.0
    return float((1 + total_return) ** (1 / years) - 1)


def calc_mdd(equity: pd.Series) -> float:
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    return float(drawdown.min())


def calc_sharpe(
    returns: pd.Series, interval: str = "1d", risk_free_annual: float = 0.02
) -> float:
    if returns.std() == 0 or returns.empty:
        return 0.0
    n = _PERIODS_PER_YEAR.get(interval, 252.0)
    rf_per_period = risk_free_annual / n
    excess = returns - rf_per_period
    return float(excess.mean() / excess.std() * np.sqrt(n))


def calc_period_returns(returns: pd.Series, interval: str) -> list[MonthlyReturn]:
    rule = _RESAMPLE_RULES.get(interval, "ME")
    date_fmt = _DATE_FMTS.get(interval, "%Y-%m")
    resampled = (1 + returns).resample(rule).prod() - 1
    return [
        MonthlyReturn(month=idx.strftime(date_fmt), return_=float(ret))
        for idx, ret in resampled.items()
        if np.isfinite(ret)
    ]


# ── 분석 1: 연도×월 히트맵 ─────────────────────────────────────────
def calc_heatmap_returns(returns: pd.Series) -> list[HeatmapPoint]:
    """인터벌과 관계없이 월별로 집계해 히트맵 데이터를 반환."""
    monthly = (1 + returns).resample("ME").prod() - 1
    result: list[HeatmapPoint] = []
    for idx, ret in monthly.items():
        if np.isfinite(ret):
            result.append(
                HeatmapPoint(year=int(idx.year), month=int(idx.month), return_=float(ret))
            )
    return result


# ── 분석 2: 스트레스 테스트 ───────────────────────────────────────
def _calc_single_stress(
    tickers: list[str], period: dict[str, str]
) -> StressTestResult | None:
    try:
        all_tickers = sorted(set(tickers + ["^GSPC"]))
        hist = yf.download(
            all_tickers,
            start=period["start"],
            end=period["end"],
            interval="1d",
            auto_adjust=True, progress=False, threads=True,
        )
        if hist.empty:
            return None

        close = hist["Close"] if isinstance(hist.columns, pd.MultiIndex) else hist
        if isinstance(close, pd.Series):
            close = close.to_frame(all_tickers[0])

        # 포트폴리오 수익률 (데이터 있는 종목만 사용)
        port_cols = [t for t in tickers if t in close.columns]
        portfolio_return: float = 0.0
        if port_cols:
            port_close = close[port_cols].dropna(how="all")
            if len(port_close) >= 2:
                ret_df = port_close.pct_change().dropna(how="all")
                portfolio_return = float((1 + ret_df.mean(axis=1)).prod() - 1)

        # S&P 500 수익률
        sp500_return: float = 0.0
        if "^GSPC" in close.columns:
            sp_prices = close["^GSPC"].dropna()
            if len(sp_prices) >= 2:
                sp500_return = float(sp_prices.iloc[-1] / sp_prices.iloc[0] - 1)

        return StressTestResult(
            name=period["name"],
            start=period["start"],
            end=period["end"],
            portfolio_return=portfolio_return if np.isfinite(portfolio_return) else 0.0,
            sp500_return=sp500_return if np.isfinite(sp500_return) else 0.0,
        )
    except Exception as exc:
        logger.warning("Stress test failed for %s: %s", period["name"], exc)
        return None


def calc_stress_tests(tickers: list[str]) -> list[StressTestResult]:
    """4개 하락장 구간 병렬 계산."""
    results: list[StressTestResult | None] = [None] * len(STRESS_PERIODS)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_calc_single_stress, tickers, period): i
            for i, period in enumerate(STRESS_PERIODS)
        }
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
    return [r for r in results if r is not None]


# ── 분석 3: 업종별 분산 ───────────────────────────────────────────
def calc_sector_weights(
    universe: pd.DataFrame, top_tickers: list[str]
) -> list[SectorWeight]:
    if "sector" not in universe.columns:
        return []

    sector_counts: dict[str, int] = {}
    for ticker in top_tickers:
        if ticker in universe.index:
            val = universe.at[ticker, "sector"]
            if pd.notna(val) and val:
                sector_counts[str(val)] = sector_counts.get(str(val), 0) + 1

    total = sum(sector_counts.values())
    if total == 0:
        return []

    return [
        SectorWeight(sector=s, count=c, weight=round(c / total, 4))
        for s, c in sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)
    ]


# ── 분석 4: 팩터 상관관계 ─────────────────────────────────────────
def calc_factor_correlations(
    universe: pd.DataFrame, factors: list[FactorInput]
) -> list[FactorCorr]:
    factor_ids = [f.id.lower() for f in factors]
    available = [fid for fid in factor_ids if fid in universe.columns]

    if len(available) < 2:
        return []

    corr_df = universe[available].dropna()
    if len(corr_df) < 3:
        return []

    corr_matrix = corr_df.corr()
    results: list[FactorCorr] = []
    for fa in available:
        for fb in available:
            val = corr_matrix.at[fa, fb]
            if np.isfinite(val):
                results.append(
                    FactorCorr(factor_a=fa, factor_b=fb, correlation=round(float(val), 3))
                )
    return results


# ── API 엔드포인트 ────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/backtest", response_model=BacktestResponse)
def backtest(request: BacktestRequest):
    try:
        start = datetime.strptime(request.start_date, "%Y-%m-%d")
        end   = datetime.strptime(request.end_date,   "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="날짜 형식은 YYYY-MM-DD여야 합니다.")

    if start >= end:
        raise HTTPException(
            status_code=400, detail="시작날짜는 종료날짜보다 이전이어야 합니다."
        )

    if request.market == "KR":
        # ── 한국 시장 ────────────────────────────────────────────
        kr_theme  = request.theme if request.theme in KR_THEME_TICKERS else "all"
        universe  = load_kr_factor_universe(kr_theme)
        scores    = compute_composite_scores(universe, request.factors)
        top_tickers = scores.head(TOP_N).index.tolist()

        period_returns, equity = run_kr_equal_weight_backtest(
            top_tickers, start, end, request.interval
        )

        # longName 은 항상 값 있음 (fallback = 종목코드)
        top_stocks = [
            str(universe.at[t, "longName"]) if t in universe.index else t
            for t in top_tickers
        ]

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_stress  = executor.submit(calc_kr_stress_tests, top_tickers)
            future_heatmap = executor.submit(calc_heatmap_returns, period_returns)
            stress_tests   = future_stress.result()
            heatmap_data   = future_heatmap.result()

        return BacktestResponse(
            cagr=calc_cagr(equity),
            mdd=calc_mdd(equity),
            sharpe=calc_sharpe(period_returns, request.interval),
            monthly_returns=calc_period_returns(period_returns, request.interval),
            top_stocks=top_stocks,
            top_tickers=top_tickers,
            heatmap_returns=heatmap_data,
            stress_tests=stress_tests,
            sector_weights=[],
            factor_correlations=calc_factor_correlations(universe, request.factors),
        )

    # ── 미국 시장 (기존 로직) ────────────────────────────────────
    theme       = request.theme if request.theme in THEME_TICKERS else "all"
    universe    = load_factor_universe(theme)
    scores      = compute_composite_scores(universe, request.factors)
    top_tickers = scores.head(TOP_N).index.tolist()

    period_returns, equity = run_equal_weight_backtest(
        top_tickers, start, end, request.interval
    )

    if "longName" in universe.columns:
        top_stocks = [
            str(universe.at[t, "longName"])
            if t in universe.index and pd.notna(universe.at[t, "longName"])
            else t
            for t in top_tickers
        ]
    else:
        top_stocks = top_tickers

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_stress  = executor.submit(calc_stress_tests, top_tickers)
        future_heatmap = executor.submit(calc_heatmap_returns, period_returns)
        stress_tests   = future_stress.result()
        heatmap_data   = future_heatmap.result()

    return BacktestResponse(
        cagr=calc_cagr(equity),
        mdd=calc_mdd(equity),
        sharpe=calc_sharpe(period_returns, request.interval),
        monthly_returns=calc_period_returns(period_returns, request.interval),
        top_stocks=top_stocks,
        top_tickers=top_tickers,
        heatmap_returns=heatmap_data,
        stress_tests=stress_tests,
        sector_weights=calc_sector_weights(universe, top_tickers),
        factor_correlations=calc_factor_correlations(universe, request.factors),
    )


# ── 기업 상세 엔드포인트 ──────────────────────────────────────────
def _is_kr_ticker(ticker: str) -> bool:
    """.KS suffix 또는 6자리 숫자 → 한국 종목"""
    return ticker.upper().endswith(".KS") or (ticker.isdigit() and len(ticker) == 6)


def _get_kr_stock_detail(ticker: str) -> StockDetailResponse:
    """yfinance .KS 티커로 한국 종목 상세 정보 조회"""
    ks = ticker if ticker.upper().endswith(".KS") else f"{ticker}.KS"
    try:
        t = yf.Ticker(ks)
        info = t.info or {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"yfinance 오류: {e}")

    # DART 종목명 → KR_TICKER_NAMES fallback → yfinance longName
    dart_names = _load_dart_kr_names()
    name = (
        dart_names.get(ks)
        or KR_TICKER_NAMES.get(ks)
        or str(info.get("longName") or info.get("shortName") or ks)
    )

    try:
        hist = t.history(period="1y")
        price_history = [
            PricePoint(date=str(idx.date()), close=round(float(close), 2))
            for idx, close in zip(hist.index, hist["Close"])
            if pd.notna(close)
        ]
    except Exception:
        price_history = []

    if not price_history:
        raise HTTPException(status_code=404, detail=f"종목을 찾을 수 없습니다: {ks}")

    def safe_float(key: str) -> float | None:
        return _finite_float(info.get(key))

    raw_cap = info.get("marketCap")
    market_cap = int(raw_cap) if raw_cap and np.isfinite(float(raw_cap)) else None

    return StockDetailResponse(
        ticker=ks,
        name=name,
        sector=info.get("sector") or None,
        industry=info.get("industry") or None,
        price_history=price_history,
        per=safe_float("trailingPE"),
        pbr=safe_float("priceToBook"),
        roe=safe_float("returnOnEquity"),
        ev_ebitda=safe_float("enterpriseToEbitda"),
        psr=safe_float("priceToSalesTrailing12Months"),
        debt_ratio=safe_float("debtToEquity"),
        operating_margin=safe_float("operatingMargins"),
        market_cap=market_cap,
        week_52_high=safe_float("fiftyTwoWeekHigh"),
        week_52_low=safe_float("fiftyTwoWeekLow"),
        current_price=price_history[-1].close,
    )


@app.get("/stock/{ticker}", response_model=StockDetailResponse)
def get_stock_detail(ticker: str):
    # 6자리 숫자면 한국 종목
    if _is_kr_ticker(ticker):
        return _get_kr_stock_detail(ticker)

    # ── 미국 종목 (yfinance) ─────────────────────────────────────
    upper = ticker.upper()
    try:
        t = yf.Ticker(upper)
        info = t.info or {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"yfinance 오류: {e}")

    if not info.get("regularMarketPrice") and not info.get("currentPrice"):
        raise HTTPException(status_code=404, detail=f"종목을 찾을 수 없습니다: {upper}")

    try:
        hist = t.history(period="1y")
        price_history = [
            PricePoint(date=str(idx.date()), close=round(float(close), 4))
            for idx, close in zip(hist.index, hist["Close"])
            if pd.notna(close)
        ]
    except Exception:
        price_history = []

    def safe_float(key: str) -> float | None:
        return _finite_float(info.get(key))

    raw_cap = info.get("marketCap")
    market_cap = int(raw_cap) if raw_cap and np.isfinite(float(raw_cap)) else None

    return StockDetailResponse(
        ticker=upper,
        name=str(info.get("longName") or info.get("shortName") or upper),
        sector=info.get("sector") or None,
        industry=info.get("industry") or None,
        price_history=price_history,
        per=safe_float("trailingPE"),
        pbr=safe_float("priceToBook"),
        roe=safe_float("returnOnEquity"),
        ev_ebitda=safe_float("enterpriseToEbitda"),
        psr=safe_float("priceToSalesTrailing12Months"),
        debt_ratio=safe_float("debtToEquity"),
        operating_margin=safe_float("operatingMargins"),
        market_cap=market_cap,
        week_52_high=safe_float("fiftyTwoWeekHigh"),
        week_52_low=safe_float("fiftyTwoWeekLow"),
        current_price=safe_float("currentPrice") or safe_float("regularMarketPrice"),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
