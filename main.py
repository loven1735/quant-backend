from __future__ import annotations

import io
import logging
import os
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
DART_API_KEY: str = os.environ.get(
    "DART_API_KEY", "50b787d351a2bdb6e499293a663069be9047d462"
)

KOSPI200_TICKERS: list[str] = [
    "005930", "000660", "373220", "207940", "005380",  # 삼성전자, SK하이닉스, LG에너지솔루션, 삼성바이오로직스, 현대차
    "006400", "051910", "035420", "000270", "105560",  # 삼성SDI, LG화학, NAVER, 기아, KB금융
    "055550", "012330", "028260", "066570", "316140",  # 신한지주, 현대모비스, 삼성물산, LG전자, 우리금융지주
    "086790", "003550", "032830", "034730", "017670",  # 하나금융지주, LG, 삼성생명, SK, SK텔레콤
    "011200", "018260", "096770", "009150", "010130",  # HMM, 삼성에스디에스, SK이노베이션, 삼성전기, 고려아연
    "030200", "003490", "015760", "036570", "086280",  # KT, 대한항공, 한국전력, 엔씨소프트, 현대글로비스
    "259960", "068270", "035720", "003670", "010950",  # 크래프톤, 셀트리온, 카카오, 포스코퓨처엠, S-Oil
    "034220", "000810", "005490", "011070", "042700",  # LG디스플레이, 삼성화재, POSCO홀딩스, LG이노텍, 한미반도체
    "000100", "033780", "139480", "097950", "009830",  # 유한양행, KT&G, 이마트, CJ제일제당, 한화솔루션
    "024110", "001040", "023530", "029780", "005830",  # IBK기업은행, CJ, 롯데쇼핑, 삼성카드, DB손해보험
    "000720", "004020", "271560", "326030", "079550",  # 현대건설, 현대제철, 오리온, SK바이오팜, LIG넥스원
    "088350", "002790", "069620", "036460", "047050",  # 한화생명, 아모레G, 대웅제약, 한국가스공사, 포스코인터내셔널
    "251270", "000880", "047810", "010060", "069960",  # 넷마블, 한화, 한국항공우주, OCI홀딩스, 현대백화점
    "002380", "175330", "022100", "012750", "111770",  # KCC, JB금융지주, 포스코DX, 에스원, 영원무역
]

KR_THEME_TICKERS: dict[str, list[str]] = {
    "all": KOSPI200_TICKERS,
    "semiconductor": ["005930", "000660", "009150", "011070", "042700", "096770"],
    "battery": ["373220", "006400", "051910", "003670", "009830"],
    "finance": ["105560", "055550", "086790", "316140", "024110", "005830", "088350", "029780", "175330"],
    "auto": ["005380", "000270", "012330", "086280"],
    "pharma_bio": ["207940", "068270", "000100", "069620", "326030"],
    "tech": ["035420", "035720", "036570", "018260", "259960", "251270"],
    "energy": ["096770", "010950", "015760", "036460", "010060"],
    "consumer": ["139480", "023530", "097950", "001040", "002790", "271560"],
}

KR_STRESS_PERIODS: list[dict[str, str]] = [
    {"name": "코로나 폭락",   "start": "2020-01-20", "end": "2020-03-19"},
    {"name": "금리인상 충격", "start": "2022-01-03", "end": "2022-10-12"},
    {"name": "금융위기",      "start": "2008-09-15", "end": "2009-03-09"},
    {"name": "IT버블",        "start": "2000-03-10", "end": "2001-09-30"},
]

# KR 팩터 캐시 + DART corp_code 캐시
_kr_factor_cache: dict[str, pd.DataFrame] = {}
_kr_corp_codes: dict[str, str] = {}


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
        if value is not None:
            row[factor_id] = value

    gpa = _extract_gpa(t, info)
    if gpa is not None:
        row["gpa"] = gpa

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

    close = hist["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])

    momentum: dict[str, float] = {}
    for ticker in close.columns:
        prices = close[ticker].dropna()
        if len(prices) < 2:
            continue
        ret = float(prices.iloc[-1] / prices.iloc[0] - 1)
        if np.isfinite(ret):
            momentum[str(ticker)] = ret

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
            factors = future.result()
            if factors:
                rows.append({"ticker": ticker, **factors})

    if not rows:
        raise HTTPException(
            status_code=503,
            detail="팩터 데이터를 가져오지 못했습니다. 잠시 후 다시 시도해 주세요.",
        )

    df = pd.DataFrame(rows).set_index("ticker")
    mom_1m = _load_momentum_1m(tickers)
    if not mom_1m.empty:
        df["momentum_1m"] = mom_1m
    mom_3m = _load_momentum_3m(tickers)
    if not mom_3m.empty:
        df["momentum_3m"] = mom_3m

    _factor_cache[theme] = df
    logger.info("Loaded factor data for theme '%s': %d tickers", theme, len(df))
    return df


# ── KR 팩터 데이터 로딩 ───────────────────────────────────────────
def _get_recent_trading_date() -> str:
    """최근 평일(거래일 근사) 날짜 반환 (YYYYMMDD)"""
    d = datetime.now()
    for _ in range(10):
        if d.weekday() < 5:
            return d.strftime("%Y%m%d")
        d -= timedelta(days=1)
    return datetime.now().strftime("%Y%m%d")


def _parse_kr_number(raw: str) -> float | None:
    """한국 재무제표 숫자 파싱 (쉼표·괄호 처리)"""
    s = raw.strip().replace(",", "")
    if not s:
        return None
    try:
        if s.startswith("(") and s.endswith(")"):
            return -float(s[1:-1])
        return float(s)
    except ValueError:
        return None


def _load_kr_pykrx_factors(tickers: list[str]) -> pd.DataFrame:
    """pykrx로 KR 기본 팩터 로드 (PER, PBR, ROE 근사)"""
    try:
        from pykrx import stock as krx_stock  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("pykrx 미설치. pip install pykrx 실행 필요.")
        return pd.DataFrame()

    date_str = _get_recent_trading_date()
    try:
        df = krx_stock.get_market_fundamental_by_ticker(date_str, market="KOSPI")
        if df is None or df.empty:
            return pd.DataFrame()

        # 인덱스를 6자리 문자열로 정규화
        df.index = df.index.astype(str).str.zfill(6)
        filtered = df[df.index.isin(tickers)].copy()
        if filtered.empty:
            return pd.DataFrame()

        result = pd.DataFrame(index=filtered.index)

        per = filtered["PER"].replace(0, np.nan).astype(float)
        result["per"] = per.where(np.isfinite(per))

        pbr = filtered["PBR"].replace(0, np.nan).astype(float)
        result["pbr"] = pbr.where(np.isfinite(pbr))

        eps = filtered["EPS"].replace(0, np.nan).astype(float)
        bps = filtered["BPS"].replace(0, np.nan).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            roe = eps / bps
        result["roe"] = roe.where(np.isfinite(roe))

        logger.info("pykrx 팩터 로드 완료: %d종목", len(result))
        return result
    except Exception as e:
        logger.warning("pykrx 팩터 로드 실패: %s", e)
        return pd.DataFrame()


def _load_kr_market_caps(tickers: list[str]) -> dict[str, float]:
    """pykrx로 시가총액 조회"""
    try:
        from pykrx import stock as krx_stock  # type: ignore[import-not-found]
    except ImportError:
        return {}

    date_str = _get_recent_trading_date()
    try:
        df = krx_stock.get_market_cap_by_ticker(date_str, market="KOSPI")
        if df is None or df.empty:
            return {}
        df.index = df.index.astype(str).str.zfill(6)
        result = {}
        col = "시가총액" if "시가총액" in df.columns else df.columns[0]
        for ticker in tickers:
            if ticker in df.index:
                v = df.at[ticker, col]
                if v and np.isfinite(float(v)):
                    result[ticker] = float(v)
        return result
    except Exception as e:
        logger.warning("pykrx 시가총액 조회 실패: %s", e)
        return {}


def _load_kr_company_names(tickers: list[str]) -> dict[str, str]:
    """pykrx로 KR 종목명 조회"""
    try:
        from pykrx import stock as krx_stock  # type: ignore[import-not-found]
    except ImportError:
        return {}

    result: dict[str, str] = {}
    for ticker in tickers:
        try:
            name = krx_stock.get_market_ticker_name(ticker)
            if name:
                result[ticker] = str(name)
        except Exception:
            pass
    return result


def _fetch_kr_prices(ticker: str, start_str: str, end_str: str) -> "pd.Series | None":
    """pykrx로 단일 KR 종목 종가 시계열 반환"""
    try:
        from pykrx import stock as krx_stock  # type: ignore[import-not-found]
        df = krx_stock.get_market_ohlcv_by_date(start_str, end_str, ticker)
        if df is None or df.empty:
            return None
        close_col = "종가" if "종가" in df.columns else "Close"
        prices = df[close_col].dropna()
        return prices if len(prices) >= 2 else None
    except Exception as e:
        logger.debug("KR 가격 조회 실패 (%s): %s", ticker, e)
        return None


def _load_kr_momentum(tickers: list[str], days: int, name: str) -> pd.Series:
    """pykrx로 KR 모멘텀 계산 (병렬)"""
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days + 10)
    start_str = start_dt.strftime("%Y%m%d")
    end_str = end_dt.strftime("%Y%m%d")

    momentum: dict[str, float] = {}

    def _calc(ticker: str) -> "tuple[str, float] | None":
        prices = _fetch_kr_prices(ticker, start_str, end_str)
        if prices is None:
            return None
        ret = float(prices.iloc[-1] / prices.iloc[0] - 1)
        return (ticker, ret) if np.isfinite(ret) else None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_calc, t): t for t in tickers}
        for future in as_completed(futures):
            result = future.result()
            if result:
                momentum[result[0]] = result[1]

    return pd.Series(momentum, name=name)


def _load_dart_corp_codes() -> dict[str, str]:
    """DART에서 종목코드 → corp_code 매핑 로드 (최초 1회)"""
    global _kr_corp_codes
    if _kr_corp_codes:
        return _kr_corp_codes

    url = f"https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={DART_API_KEY}"
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code != 200:
            logger.warning("DART corpCode API 오류: %d", resp.status_code)
            return {}

        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            xml_bytes = z.read("CORPCODE.xml")

        root = ET.fromstring(xml_bytes)
        for item in root.findall(".//list"):
            sc = (item.findtext("stock_code") or "").strip()
            cc = (item.findtext("corp_code") or "").strip()
            if sc and cc:
                _kr_corp_codes[sc] = cc

        logger.info("DART corp_code 로드 완료: %d개", len(_kr_corp_codes))
    except Exception as e:
        logger.warning("DART corp_code 로드 실패: %s", e)

    return _kr_corp_codes


def _fetch_dart_financials(corp_code: str) -> dict[str, float]:
    """DART 단일 회사 재무제표 조회 (전년도 사업보고서)"""
    year = datetime.now().year - 1
    for fs_div in ("CFS", "OFS"):
        params = {
            "crtfc_key": DART_API_KEY,
            "corp_code": corp_code,
            "bsns_year": str(year),
            "reprt_code": "11011",
            "fs_div": fs_div,
        }
        try:
            resp = requests.get(
                "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json",
                params=params,
                timeout=15,
            )
            data = resp.json()
            if data.get("status") != "000":
                continue

            result: dict[str, float] = {}
            for item in data.get("list", []):
                nm = item.get("account_nm", "")
                val = _parse_kr_number(item.get("thstrm_amount", ""))
                if val is None:
                    continue

                if nm in ("매출액", "수익(매출액)", "영업수익", "매출"):
                    result.setdefault("revenue", val)
                elif nm == "매출총이익":
                    result["gross_profit"] = val
                elif nm in ("영업이익", "영업이익(손실)"):
                    result["operating_income"] = val
                elif nm in ("당기순이익", "분기순이익", "당기순이익(손실)"):
                    result.setdefault("net_income", val)
                elif nm == "자산총계":
                    result["total_assets"] = val
                elif nm == "자본총계":
                    result["total_equity"] = val
                elif nm == "부채총계":
                    result["total_liabilities"] = val

            if result:
                return result
        except Exception as e:
            logger.debug("DART 조회 실패 (corp=%s fs=%s): %s", corp_code, fs_div, e)

    return {}


def _load_kr_dart_factors(tickers: list[str]) -> pd.DataFrame:
    """DART API로 KR 재무 팩터 배치 로드"""
    corp_codes = _load_dart_corp_codes()
    if not corp_codes:
        return pd.DataFrame()

    rows: list[dict] = []

    def _fetch_one(ticker: str) -> dict | None:
        cc = corp_codes.get(ticker)
        if not cc:
            return None
        data = _fetch_dart_financials(cc)
        if not data:
            return None
        return {"ticker": ticker, **data}

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_fetch_one, t): t for t in tickers}
        for future in as_completed(futures):
            r = future.result()
            if r:
                rows.append(r)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("ticker")


def load_kr_factor_universe(theme: str = "all") -> pd.DataFrame:
    global _kr_factor_cache
    cache_key = f"kr_{theme}"
    if cache_key in _kr_factor_cache:
        return _kr_factor_cache[cache_key]

    tickers = list(KR_THEME_TICKERS.get(theme, KR_THEME_TICKERS["all"]))

    # pykrx 팩터, DART 재무, 종목명 병렬 로드
    with ThreadPoolExecutor(max_workers=3) as executor:
        f_pykrx = executor.submit(_load_kr_pykrx_factors, tickers)
        f_dart  = executor.submit(_load_kr_dart_factors, tickers)
        f_names = executor.submit(_load_kr_company_names, tickers)
        pykrx_df = f_pykrx.result()
        dart_df  = f_dart.result()
        names    = f_names.result()

    market_caps = _load_kr_market_caps(tickers)

    # base DataFrame
    idx = pd.Index(tickers, name="ticker")
    df = pykrx_df.copy() if not pykrx_df.empty else pd.DataFrame(index=idx)

    if not dart_df.empty:
        for col in dart_df.columns:
            df[col] = dart_df.reindex(df.index)[col]

    # index.map 으로 할당: names 미매칭 시 종목코드를 그대로 사용
    df["longName"] = df.index.map(lambda t: names.get(str(t), str(t)))
    df["market_cap"] = pd.Series(market_caps, dtype=float).reindex(df.index)

    # PSR = 시가총액 / 매출액
    if "revenue" in df.columns and "market_cap" in df.columns:
        rev = df["revenue"].replace(0, np.nan)
        psr = df["market_cap"] / rev
        df["psr"] = psr.where(np.isfinite(psr) & (psr > 0))

    # GPA = 매출총이익 / 자산총계
    if "gross_profit" in df.columns and "total_assets" in df.columns:
        ta = df["total_assets"].replace(0, np.nan)
        gpa = df["gross_profit"] / ta
        df["gpa"] = gpa.where(np.isfinite(gpa))

    # 부채비율 = 부채총계 / 자본총계
    if "total_liabilities" in df.columns and "total_equity" in df.columns:
        eq = df["total_equity"].replace(0, np.nan)
        dr = df["total_liabilities"] / eq
        df["debt_ratio"] = dr.where(np.isfinite(dr))

    # ROE from DART (DART 데이터로 pykrx ROE 보완)
    if "net_income" in df.columns and "total_equity" in df.columns:
        eq = df["total_equity"].replace(0, np.nan)
        roe_dart = (df["net_income"] / eq).where(
            lambda s: np.isfinite(s)
        )
        if "roe" not in df.columns:
            df["roe"] = roe_dart
        else:
            df["roe"] = df["roe"].fillna(roe_dart)

    # EV/EBITDA 근사: EV ≈ 시가총액 + 부채총계, EBITDA ≈ 영업이익
    if "operating_income" in df.columns and "market_cap" in df.columns:
        liab = df.get("total_liabilities", pd.Series(0.0, index=df.index)).fillna(0)
        ev = df["market_cap"].fillna(0) + liab
        ebitda = df["operating_income"].replace(0, np.nan)
        ev_ebitda = ev / ebitda
        df["ev_ebitda"] = ev_ebitda.where(np.isfinite(ev_ebitda) & (ev_ebitda > 0))

    # 모멘텀 (pykrx)
    mom_1m = _load_kr_momentum(tickers, days=30, name="momentum_1m")
    if not mom_1m.empty:
        df["momentum_1m"] = mom_1m

    mom_3m = _load_kr_momentum(tickers, days=90, name="momentum_3m")
    if not mom_3m.empty:
        df["momentum_3m"] = mom_3m

    # 중간 계산용 컬럼 제거
    for col in ("revenue", "gross_profit", "net_income", "total_assets",
                "total_equity", "total_liabilities", "operating_income", "market_cap"):
        df.drop(columns=[col], inplace=True, errors="ignore")

    _kr_factor_cache[cache_key] = df
    logger.info("KR 팩터 유니버스 로드 완료 (theme='%s'): %d종목", theme, len(df))
    return df


def run_kr_equal_weight_backtest(
    tickers: list[str], start: datetime, end: datetime, interval: str = "1d"
) -> tuple[pd.Series, pd.Series]:
    """KR 종목 백테스트 (pykrx 기반)"""
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")

    close_dict: dict[str, pd.Series] = {}

    def _fetch(ticker: str) -> "tuple[str, pd.Series] | None":
        prices = _fetch_kr_prices(ticker, start_str, end_str)
        return (ticker, prices) if prices is not None else None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_fetch, t): t for t in tickers}
        for future in as_completed(futures):
            result = future.result()
            if result:
                close_dict[result[0]] = result[1]

    if not close_dict:
        raise HTTPException(status_code=503, detail="KR 주가 데이터를 가져오지 못했습니다.")

    close = pd.DataFrame(close_dict)

    if interval == "1wk":
        close = close.resample("W-FRI").last()
    elif interval == "1mo":
        close = close.resample("ME").last()

    close = close.dropna(axis=1, how="all").dropna(how="all")
    if close.shape[1] == 0:
        raise HTTPException(status_code=503, detail="유효한 KR 주가 시계열이 없습니다.")

    daily_returns = close.pct_change().dropna(how="all")
    portfolio_returns = daily_returns.mean(axis=1).dropna()
    if portfolio_returns.empty:
        raise HTTPException(status_code=503, detail="KR 포트폴리오 수익률을 계산할 수 없습니다.")

    equity = (1 + portfolio_returns).cumprod()
    return portfolio_returns, equity


def _calc_kr_single_stress(
    tickers: list[str], period: dict[str, str]
) -> StressTestResult | None:
    start_str = period["start"].replace("-", "")
    end_str = period["end"].replace("-", "")
    try:
        # 포트폴리오 수익률 (pykrx)
        close_dict: dict[str, pd.Series] = {}

        def _fetch(ticker: str) -> "tuple[str, pd.Series] | None":
            prices = _fetch_kr_prices(ticker, start_str, end_str)
            return (ticker, prices) if prices is not None else None

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_fetch, t): t for t in tickers}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    close_dict[result[0]] = result[1]

        portfolio_return: float = 0.0
        if close_dict:
            port_close = pd.DataFrame(close_dict).dropna(how="all")
            if len(port_close) >= 2:
                ret_df = port_close.pct_change().dropna(how="all")
                portfolio_return = float((1 + ret_df.mean(axis=1)).prod() - 1)

        # KOSPI 벤치마크 (pykrx 인덱스 코드 1001)
        kospi_return: float = 0.0
        try:
            from pykrx import stock as krx_stock  # type: ignore[import-not-found]
            kospi_df = krx_stock.get_index_ohlcv_by_date(start_str, end_str, "1001")
            if kospi_df is not None and not kospi_df.empty:
                close_col = "종가" if "종가" in kospi_df.columns else "Close"
                kp = kospi_df[close_col].dropna()
                if len(kp) >= 2:
                    kospi_return = float(kp.iloc[-1] / kp.iloc[0] - 1)
        except Exception as e:
            logger.debug("KOSPI 벤치마크 조회 실패 (%s): %s", period["name"], e)

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

    return composite.sort_values(ascending=False)


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
        all_tickers = list(set(tickers + ["^GSPC"]))
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
    """6자리 숫자 → 한국 종목코드"""
    return ticker.isdigit() and len(ticker) == 6


def _get_kr_stock_detail(ticker: str) -> StockDetailResponse:
    """pykrx로 한국 종목 상세 정보 조회"""
    try:
        from pykrx import stock as krx_stock  # type: ignore[import-not-found]
    except ImportError:
        raise HTTPException(status_code=500, detail="pykrx가 설치되어 있지 않습니다.")

    # ── 종목명 ──────────────────────────────────────────────────
    try:
        name: str = krx_stock.get_market_ticker_name(ticker) or ticker
    except Exception:
        name = ticker

    # ── 펀더멘털 (PER, PBR, EPS, BPS) ──────────────────────────
    date_str = _get_recent_trading_date()
    per: float | None = None
    pbr: float | None = None
    roe: float | None = None

    try:
        fund_df = krx_stock.get_market_fundamental_by_ticker(date_str, market="KOSPI")
        if fund_df is not None and not fund_df.empty:
            fund_df.index = fund_df.index.astype(str).str.zfill(6)
            if ticker in fund_df.index:
                row = fund_df.loc[ticker]
                per_raw = _finite_float(row.get("PER"))
                pbr_raw = _finite_float(row.get("PBR"))
                eps_raw = _finite_float(row.get("EPS"))
                bps_raw = _finite_float(row.get("BPS"))

                per = per_raw if per_raw and per_raw != 0 else None
                pbr = pbr_raw if pbr_raw and pbr_raw != 0 else None

                # ROE ≈ EPS / BPS (소수, yfinance returnOnEquity 형식 통일)
                if eps_raw is not None and bps_raw and bps_raw != 0:
                    r = eps_raw / bps_raw
                    roe = r if np.isfinite(r) else None
    except Exception as e:
        logger.warning("KR 펀더멘털 조회 실패 (%s): %s", ticker, e)

    # ── 1년 가격 히스토리 ────────────────────────────────────────
    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=365)
    price_history: list[PricePoint] = []
    current_price: float | None = None
    week_52_high:  float | None = None
    week_52_low:   float | None = None

    try:
        ohlcv = krx_stock.get_market_ohlcv_by_date(
            start_dt.strftime("%Y%m%d"),
            end_dt.strftime("%Y%m%d"),
            ticker,
        )
        if ohlcv is not None and not ohlcv.empty:
            # pykrx 는 컬럼명이 한글(종가) 또는 영문(Close) 일 수 있음
            close_col = "종가" if "종가" in ohlcv.columns else "Close"
            for idx, row in ohlcv.iterrows():
                c = row.get(close_col)
                if c is not None and np.isfinite(float(c)) and float(c) > 0:
                    price_history.append(
                        PricePoint(date=str(idx.date()), close=round(float(c), 2))
                    )

            if price_history:
                current_price = price_history[-1].close
                closes = [p.close for p in price_history]
                week_52_high = max(closes)
                week_52_low  = min(closes)
    except Exception as e:
        logger.warning("KR 가격 히스토리 조회 실패 (%s): %s", ticker, e)

    # ── 시가총액 ────────────────────────────────────────────────
    market_cap: int | None = None
    try:
        cap_df = krx_stock.get_market_cap_by_ticker(date_str, market="KOSPI")
        if cap_df is not None and not cap_df.empty:
            cap_df.index = cap_df.index.astype(str).str.zfill(6)
            if ticker in cap_df.index:
                col = "시가총액" if "시가총액" in cap_df.columns else cap_df.columns[0]
                v = cap_df.at[ticker, col]
                if v and np.isfinite(float(v)):
                    market_cap = int(v)
    except Exception as e:
        logger.warning("KR 시가총액 조회 실패 (%s): %s", ticker, e)

    if not name and not price_history:
        raise HTTPException(status_code=404, detail=f"종목을 찾을 수 없습니다: {ticker}")

    return StockDetailResponse(
        ticker=ticker,
        name=name,
        sector=None,
        industry=None,
        price_history=price_history,
        per=per,
        pbr=pbr,
        roe=roe,
        ev_ebitda=None,
        psr=None,
        debt_ratio=None,
        operating_margin=None,
        market_cap=market_cap,
        week_52_high=week_52_high,
        week_52_low=week_52_low,
        current_price=current_price,
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
