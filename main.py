from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Literal

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


# ── Pydantic 모델 ─────────────────────────────────────────────────
class FactorInput(BaseModel):
    id: str
    weight: float = Field(ge=0)


class BacktestRequest(BaseModel):
    factors: list[FactorInput]
    start_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="YYYY-MM-DD")
    end_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="YYYY-MM-DD")
    interval: Literal["1d", "1wk", "1mo"] = "1d"
    theme: str = "all"  # "all" | "semiconductor" | "tech" | "pharma_bio" | ... (THEME_TICKERS 키)


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

    theme       = request.theme if request.theme in THEME_TICKERS else "all"
    universe    = load_factor_universe(theme)
    scores      = compute_composite_scores(universe, request.factors)
    top_tickers = scores.head(TOP_N).index.tolist()

    period_returns, equity = run_equal_weight_backtest(
        top_tickers, start, end, request.interval
    )

    # 표시용 회사명
    if "longName" in universe.columns:
        top_stocks = [
            str(universe.at[t, "longName"])
            if t in universe.index and pd.notna(universe.at[t, "longName"])
            else t
            for t in top_tickers
        ]
    else:
        top_stocks = top_tickers

    # 분석 섹션 (스트레스 테스트는 병렬 실행)
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_stress  = executor.submit(calc_stress_tests, top_tickers)
        future_heatmap = executor.submit(calc_heatmap_returns, period_returns)
        stress_tests   = future_stress.result()
        heatmap_data   = future_heatmap.result()

    sector_weights       = calc_sector_weights(universe, top_tickers)
    factor_correlations  = calc_factor_correlations(universe, request.factors)

    return BacktestResponse(
        cagr=calc_cagr(equity),
        mdd=calc_mdd(equity),
        sharpe=calc_sharpe(period_returns, request.interval),
        monthly_returns=calc_period_returns(period_returns, request.interval),
        top_stocks=top_stocks,
        top_tickers=top_tickers,
        heatmap_returns=heatmap_data,
        stress_tests=stress_tests,
        sector_weights=sector_weights,
        factor_correlations=factor_correlations,
    )


# ── 기업 상세 엔드포인트 ──────────────────────────────────────────
@app.get("/stock/{ticker}", response_model=StockDetailResponse)
def get_stock_detail(ticker: str):
    upper = ticker.upper()
    try:
        t = yf.Ticker(upper)
        info = t.info or {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"yfinance 오류: {e}")

    # 종목이 존재하는지 최소 검증
    if not info.get("regularMarketPrice") and not info.get("currentPrice"):
        raise HTTPException(status_code=404, detail=f"종목을 찾을 수 없습니다: {upper}")

    # 최근 1년 일봉 가격 히스토리
    try:
        hist = t.history(period="1y")
        price_history: list[PricePoint] = [
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
