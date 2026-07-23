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
from dotenv import load_dotenv
from pydantic import BaseModel, Field
try:
    from supabase import create_client, Client as SupabaseClient
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

load_dotenv()

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
# KOSPI 대표 종목 (yfinance .KS 티커 → 한글 종목명)
# pykrx 동적 로딩 실패 시 fallback으로 사용; 8개 섹터를 고르게 커버
KR_TICKER_NAMES: dict[str, str] = {
    # ── 반도체/전자 ──────────────────────────────────────────────
    "005930.KS": "삼성전자",
    "000660.KS": "SK하이닉스",
    "009150.KS": "삼성전기",
    "011070.KS": "LG이노텍",
    "042700.KS": "한미반도체",
    "000990.KS": "DB하이텍",
    "007660.KS": "이수페타시스",         # 반도체 PCB (KOSPI)
    "066570.KS": "LG전자",
    "034220.KS": "LG디스플레이",
    "006400.KS": "삼성SDI",
    "373220.KS": "LG에너지솔루션",
    "018260.KS": "삼성에스디에스",
    "402340.KS": "SK스퀘어",          # SK하이닉스 최대주주
    "003670.KS": "포스코퓨처엠",       # 구 포스코케미칼, 배터리 소재
    # ── 자동차 ───────────────────────────────────────────────────
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
    # ── 금융 ──────────────────────────────────────────────────────
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
    "071050.KS": "한국금융지주",        # 한국투자증권 모회사
    # ── 바이오/제약 ──────────────────────────────────────────────
    "207940.KS": "삼성바이오로직스",
    "068270.KS": "셀트리온",
    "006280.KS": "녹십자",              # 셀트리온헬스케어 대체 (KOSPI)
    "128940.KS": "한미약품",
    "000100.KS": "유한양행",
    "185750.KS": "종근당",
    "069620.KS": "대웅제약",
    "009420.KS": "한미사이언스",         # 한미약품 모회사
    "302440.KS": "SK바이오사이언스",
    "326030.KS": "SK바이오팜",
    "003090.KS": "대웅",               # 대웅제약 지주
    "017900.KS": "광동제약",
    "001060.KS": "JW중외제약",          # 휴젤 대체 (KOSPI)
    # ── IT/소프트웨어/통신 ────────────────────────────────────────
    "035420.KS": "NAVER",
    "035720.KS": "카카오",
    "259960.KS": "크래프톤",
    "036570.KS": "엔씨소프트",
    "251270.KS": "넷마블",
    "377300.KS": "카카오페이",           # 카카오게임즈 대체 (KOSPI)
    "017670.KS": "SK텔레콤",
    "030200.KS": "KT",
    "032640.KS": "LG유플러스",
    "012510.KS": "더존비즈온",           # 안랩 대체 (KOSPI, ERP·클라우드)
    "030000.KS": "제일기획",
    # ── 에너지/화학 ──────────────────────────────────────────────
    "051910.KS": "LG화학",
    "096770.KS": "SK이노베이션",
    "011170.KS": "롯데케미칼",
    "009830.KS": "한화솔루션",
    "011780.KS": "금호석유",
    "010950.KS": "S-Oil",
    "036460.KS": "한국가스공사",
    "015760.KS": "한국전력",
    "010060.KS": "OCI",               # 태양광/화학
    "000880.KS": "한화",               # 에너지·화학 지주
    # ── 건설/인프라 ──────────────────────────────────────────────
    "000720.KS": "현대건설",
    "006360.KS": "GS건설",
    "047040.KS": "대우건설",
    "375500.KS": "DL이앤씨",
    "294870.KS": "HDC현대산업개발",
    "028260.KS": "삼성물산",
    "034020.KS": "두산에너빌리티",       # EPC/발전소 건설
    "009410.KS": "태영건설",
    "003070.KS": "코오롱글로벌",
    # ── 소비재/유통 ───────────────────────────────────────────────
    "139480.KS": "이마트",
    "023530.KS": "롯데쇼핑",
    "097950.KS": "CJ제일제당",
    "004170.KS": "신세계",
    "008770.KS": "호텔신라",
    "282330.KS": "BGF리테일",
    "271560.KS": "오리온",              # CJ ENM 대체 (KOSPI)
    "001040.KS": "CJ",
    "004370.KS": "농심",
    "090430.KS": "아모레퍼시픽",
    "051900.KS": "LG생활건강",
    "002790.KS": "아모레G",
    "005300.KS": "롯데칠성",
    "000080.KS": "하이트진로",
    "003230.KS": "삼양식품",
    # ── 기타 대형주/중공업 ────────────────────────────────────────
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

KOSPI200_TICKERS: list[str] = list(KR_TICKER_NAMES.keys())

KR_THEME_TICKERS: dict[str, list[str]] = {
    "all": KOSPI200_TICKERS,
    # ── 반도체/전자 ──────────────────────────────────────────────
    # KOSPI200_TICKERS(fallback)에 전부 포함됨
    "semiconductor": [
        "005930.KS",  # 삼성전자
        "000660.KS",  # SK하이닉스
        "009150.KS",  # 삼성전기
        "011070.KS",  # LG이노텍
        "042700.KS",  # 한미반도체
        "000990.KS",  # DB하이텍
        "007660.KS",  # 이수페타시스 (반도체 PCB)
        "066570.KS",  # LG전자
        "034220.KS",  # LG디스플레이
        "006400.KS",  # 삼성SDI (배터리/소재)
        "373220.KS",  # LG에너지솔루션
        "018260.KS",  # 삼성에스디에스
        "402340.KS",  # SK스퀘어 (SK하이닉스 모회사)
        "003670.KS",  # 포스코퓨처엠 (배터리 소재)
        "034730.KS",  # SK (SK하이닉스 최종 지주)
    ],
    # ── 자동차/부품 ──────────────────────────────────────────────
    "automobile": [
        "005380.KS",  # 현대차
        "000270.KS",  # 기아
        "012330.KS",  # 현대모비스
        "011210.KS",  # 현대위아
        "204320.KS",  # 만도
        "018880.KS",  # 한온시스템
        "086280.KS",  # 현대글로비스 (물류)
        "161390.KS",  # 한국타이어앤테크놀로지
        "002350.KS",  # 넥센타이어
        "073240.KS",  # 금호타이어
        "064350.KS",  # 현대로템 (철도/방산차량)
        "004020.KS",  # 현대제철 (자동차 강판)
        "066570.KS",  # LG전자 (전장부품)
        "011070.KS",  # LG이노텍 (카메라모듈/전장)
        "009150.KS",  # 삼성전기 (전장부품)
    ],
    # ── 금융/보험/증권 ────────────────────────────────────────────
    "finance": [
        "105560.KS",  # KB금융
        "055550.KS",  # 신한지주
        "086790.KS",  # 하나금융지주
        "316140.KS",  # 우리금융지주
        "032830.KS",  # 삼성생명
        "000810.KS",  # 삼성화재
        "006800.KS",  # 미래에셋증권
        "005940.KS",  # NH투자증권
        "016360.KS",  # 삼성증권
        "138930.KS",  # BNK금융지주 (부산·경남은행)
        "139130.KS",  # DGB금융지주 (대구은행)
        "175330.KS",  # JB금융지주
        "088350.KS",  # 한화생명
        "204210.KS",  # 메리츠금융지주
        "071050.KS",  # 한국금융지주 (한국투자증권)
    ],
    # ── 바이오/제약 ──────────────────────────────────────────────
    "bio": [
        "207940.KS",  # 삼성바이오로직스
        "068270.KS",  # 셀트리온
        "006280.KS",  # 녹십자
        "128940.KS",  # 한미약품
        "000100.KS",  # 유한양행
        "185750.KS",  # 종근당
        "069620.KS",  # 대웅제약
        "009420.KS",  # 한미사이언스 (한미약품 모회사)
        "302440.KS",  # SK바이오사이언스
        "326030.KS",  # SK바이오팜
        "003090.KS",  # 대웅 (대웅제약 지주)
        "017900.KS",  # 광동제약
        "001060.KS",  # JW중외제약
    ],
    # ── IT/소프트웨어/통신 ────────────────────────────────────────
    "it": [
        "035420.KS",  # NAVER
        "035720.KS",  # 카카오
        "259960.KS",  # 크래프톤
        "036570.KS",  # 엔씨소프트
        "251270.KS",  # 넷마블
        "018260.KS",  # 삼성에스디에스
        "377300.KS",  # 카카오페이
        "017670.KS",  # SK텔레콤
        "030200.KS",  # KT
        "032640.KS",  # LG유플러스
        "066570.KS",  # LG전자
        "012510.KS",  # 더존비즈온 (ERP·클라우드)
        "030000.KS",  # 제일기획
        "034730.KS",  # SK (IT투자 지주)
        "034220.KS",  # LG디스플레이
    ],
    # ── 에너지/화학 ──────────────────────────────────────────────
    "energy": [
        "051910.KS",  # LG화학
        "006400.KS",  # 삼성SDI
        "096770.KS",  # SK이노베이션
        "011170.KS",  # 롯데케미칼
        "009830.KS",  # 한화솔루션 (태양광)
        "011780.KS",  # 금호석유
        "010950.KS",  # S-Oil
        "036460.KS",  # 한국가스공사
        "015760.KS",  # 한국전력
        "373220.KS",  # LG에너지솔루션
        "003670.KS",  # 포스코퓨처엠 (배터리 소재)
        "010060.KS",  # OCI (태양광/화학)
        "000880.KS",  # 한화 (에너지·화학 지주)
        "034730.KS",  # SK (에너지 투자 지주)
        "005490.KS",  # POSCO홀딩스 (수소/그린철강)
    ],
    # ── 건설/인프라/건자재 ───────────────────────────────────────
    "construction": [
        "000720.KS",  # 현대건설
        "006360.KS",  # GS건설
        "047040.KS",  # 대우건설
        "375500.KS",  # DL이앤씨
        "294870.KS",  # HDC현대산업개발
        "028260.KS",  # 삼성물산 (건설·상사)
        "034020.KS",  # 두산에너빌리티 (EPC/발전소 건설)
        "009410.KS",  # 태영건설
        "003070.KS",  # 코오롱글로벌 (건설)
        "004020.KS",  # 현대제철 (건설용 철강)
        "005490.KS",  # POSCO홀딩스 (철강)
        "000880.KS",  # 한화 (건설 계열 보유)
    ],
    # ── 소비재/유통/식품/뷰티 ────────────────────────────────────
    "consumer": [
        "139480.KS",  # 이마트
        "023530.KS",  # 롯데쇼핑
        "097950.KS",  # CJ제일제당
        "004170.KS",  # 신세계
        "008770.KS",  # 호텔신라
        "282330.KS",  # BGF리테일
        "271560.KS",  # 오리온
        "001040.KS",  # CJ (지주)
        "004370.KS",  # 농심
        "090430.KS",  # 아모레퍼시픽
        "051900.KS",  # LG생활건강
        "002790.KS",  # 아모레G
        "005300.KS",  # 롯데칠성
        "000080.KS",  # 하이트진로
        "003230.KS",  # 삼양식품
    ],
    # ── 기존 테마 (하위 호환) ──────────────────────────────────
    "auto":        ["005380.KS", "000270.KS", "012330.KS", "011210.KS", "204320.KS"],
    "pharma_bio":  ["207940.KS", "068270.KS", "000100.KS", "128940.KS", "009420.KS"],
    "tech":        ["035420.KS", "035720.KS", "259960.KS", "036570.KS", "251270.KS"],
    "energy_chem": ["051910.KS", "096770.KS", "011170.KS", "010950.KS", "006400.KS"],
    "battery":     ["006400.KS", "051910.KS", "003670.KS", "373220.KS"],
    "defense":     ["047810.KS", "012450.KS"],
    "shipbuilding":["009540.KS", "010140.KS", "042660.KS"],
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

# KR 팩터 캐시 / DART 캐시
_kr_factor_cache: dict[str, pd.DataFrame] = {}
_kr_names_cache: dict[str, str] = {}   # "005930.KS" → "삼성전자"
_kr_corp_codes: dict[str, str] = {}    # "005930" → DART corp_code

# Supabase 클라이언트 캐시
_supabase_client: SupabaseClient | None = None


def _get_supabase():
    """Supabase 클라이언트 반환 (지연 초기화, 1회 캐시). 패키지 없으면 None."""
    if not SUPABASE_AVAILABLE:
        return None
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        logger.warning("SUPABASE_URL / SUPABASE_KEY 미설정 — KR 역사적 재무 데이터 사용 불가")
        return None
    try:
        _supabase_client = create_client(url, key)
        logger.info("Supabase 연결 완료")
    except Exception as exc:
        logger.warning("Supabase 연결 실패: %s", exc)
    return _supabase_client


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
    """DART 단일 회사 재무제표 조회 → 당기순이익·자기자본 반환 (단위: 원).
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


# ── KR 팩터 데이터 로딩 (Supabase factor_scores) ────────────────────────
def load_kr_factor_universe(theme: str = "all") -> pd.DataFrame:
    """
    Supabase factor_scores 테이블에서 최신 날짜 팩터 데이터를 로드.

    precompute_factors.py가 매일 새벽 저장한 데이터를 그대로 사용하므로
    DART / yfinance 실시간 호출 없이 즉시 반환.

    테마 필터링은 KR_THEME_TICKERS 기준으로 적용.
    """
    global _kr_factor_cache
    cache_key = f"kr_{theme}"
    if cache_key in _kr_factor_cache:
        return _kr_factor_cache[cache_key]

    sb = _get_supabase()
    if sb is None:
        raise HTTPException(
            status_code=503,
            detail="Supabase 연결 불가 — SUPABASE_URL / SUPABASE_KEY 환경변수를 확인하세요.",
        )

    # ── 1. 최신 날짜 확인 ────────────────────────────────────────
    try:
        date_resp = (
            sb.table("factor_scores")
            .select("date")
            .order("date", desc=True)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"factor_scores 날짜 조회 실패: {exc}")

    if not date_resp.data:
        raise HTTPException(status_code=503, detail="factor_scores 테이블에 데이터가 없습니다.")

    latest_date: str = date_resp.data[0]["date"]
    logger.info("factor_scores 기준 날짜: %s", latest_date)

    # ── 2. 해당 날짜 전체 데이터 조회 (1000행 단위 페이지네이션) ─
    COLS = "ticker, per, pbr, roe, gpa, momentum_1m, momentum_3m, psr, debt_ratio, market_cap"
    PAGE = 1000
    all_rows: list[dict] = []
    offset = 0

    try:
        while True:
            resp = (
                sb.table("factor_scores")
                .select(COLS)
                .eq("date", latest_date)
                .range(offset, offset + PAGE - 1)
                .execute()
            )
            if not resp.data:
                break
            all_rows.extend(resp.data)
            if len(resp.data) < PAGE:
                break
            offset += PAGE
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"factor_scores 조회 실패: {exc}")

    if not all_rows:
        raise HTTPException(
            status_code=503,
            detail=f"factor_scores에 {latest_date} 날짜 데이터가 없습니다.",
        )

    logger.info("factor_scores 로드: %d종목 (%s 기준)", len(all_rows), latest_date)

    # ── 3. DataFrame 구성 (6자리 코드 → .KS suffix 변환) ─────────
    NUM_COLS = (
        "per", "pbr", "roe", "gpa",
        "momentum_1m", "momentum_3m",
        "psr", "debt_ratio", "market_cap",
    )
    rows = [
        {"ticker": f"{row['ticker']}.KS", **{c: row.get(c) for c in NUM_COLS}}
        for row in all_rows
    ]
    df = pd.DataFrame(rows).set_index("ticker")
    df = df.apply(pd.to_numeric, errors="coerce")

    # ── 4. 테마 필터링 ───────────────────────────────────────────
    if theme != "all" and theme in KR_THEME_TICKERS:
        theme_set = set(KR_THEME_TICKERS[theme])
        df = df[df.index.isin(theme_set)]
        logger.info("테마 필터 '%s' 적용: %d종목", theme, len(df))

    if df.empty:
        raise HTTPException(
            status_code=503,
            detail=f"theme='{theme}'에 해당하는 factor_scores 데이터가 없습니다.",
        )

    # ── 5. 종목명 (DART corpCode.xml, 1회 캐시) ──────────────────
    dart_names = _load_dart_kr_names()
    df["longName"] = df.index.map(
        lambda t: dart_names.get(t) or KR_TICKER_NAMES.get(t, t)
    )

    _kr_factor_cache[cache_key] = df
    logger.info(
        "KR 팩터 유니버스 로드 완료 (Supabase factor_scores, theme='%s'): %d종목",
        theme, len(df),
    )
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


# ── KR 역사적 재무 데이터 (룩어헤드 바이어스 방지) ─────────────────────────────

def _load_kr_financials_from_supabase(
    tickers: list[str], fin_year: int
) -> pd.DataFrame:
    """
    kr_financials 테이블에서 fin_year 연도의 재무 데이터 로드.
    반환: ticker(.KS) 인덱스 DataFrame / 데이터 없으면 빈 DataFrame.
    """
    sb = _get_supabase()
    if sb is None:
        return pd.DataFrame()

    stock_codes = [t.split(".")[0] for t in tickers]
    FIN_COLS = "stock_code, net_income, total_equity, total_assets, total_debt, gross_profit, revenue"

    try:
        resp = (
            sb.table("kr_financials")
            .select(FIN_COLS)
            .eq("year", fin_year)
            .in_("stock_code", stock_codes)
            .execute()
        )
    except Exception as exc:
        logger.warning("Supabase kr_financials 조회 실패 (year=%d): %s", fin_year, exc)
        return pd.DataFrame()

    if not resp.data:
        logger.debug("kr_financials: %d년 데이터 없음", fin_year)
        return pd.DataFrame()

    cols = ("net_income", "total_equity", "total_assets", "total_debt", "gross_profit", "revenue")
    rows = [
        {"ticker": f"{row['stock_code']}.KS", **{c: row.get(c) for c in cols}}
        for row in resp.data
    ]
    df = pd.DataFrame(rows).set_index("ticker")
    df = df.apply(pd.to_numeric, errors="coerce")
    logger.info("kr_financials %d년 로드: %d종목", fin_year, len(df))
    return df


def _apply_supabase_financials(
    universe: pd.DataFrame, fin_df: pd.DataFrame
) -> pd.DataFrame:
    """
    현재 팩터 유니버스에 Supabase 역사적 재무 데이터를 덮어씀.

    완전 교체 (순수 재무 비율):
      · roe       = net_income / total_equity
      · gpa       = gross_profit / total_assets
      · debt_ratio= total_debt / total_equity

    시가총액이 필요한 팩터(per, pbr, psr)는 현재 market cap 기반 값을 유지.
    """
    if fin_df.empty:
        return universe

    df = universe.copy()

    for ticker in df.index:
        if ticker not in fin_df.index:
            continue

        fin = fin_df.loc[ticker]

        def fv(col: str) -> float | None:
            v = fin.get(col)
            if v is None:
                return None
            f = float(v)
            return f if np.isfinite(f) else None

        ni     = fv("net_income")
        eq     = fv("total_equity")
        assets = fv("total_assets")
        debt   = fv("total_debt")
        gp     = fv("gross_profit")

        if ni is not None and eq is not None and eq > 0:
            roe = ni / eq
            if np.isfinite(roe):
                df.at[ticker, "roe"] = round(roe, 4)

        if gp is not None and assets is not None and assets > 0:
            gpa = gp / assets
            if np.isfinite(gpa):
                df.at[ticker, "gpa"] = round(gpa, 4)

        if debt is not None and eq is not None and eq > 0:
            dr = debt / eq
            if np.isfinite(dr):
                df.at[ticker, "debt_ratio"] = round(dr, 4)

    return df


def run_kr_annual_rebalancing_backtest(
    universe: pd.DataFrame,
    factors: list[FactorInput],
    start: datetime,
    end: datetime,
    interval: str,
) -> tuple[pd.Series, pd.Series, list[str]]:
    """
    연도별 전년도 재무 데이터로 연초 리밸런싱하는 KR 백테스트.

    각 연도마다 (year - 1) 연도의 Supabase 재무 데이터로 팩터 스코어를 재계산하고
    상위 종목을 새로 선정한 뒤 해당 연도 수익률을 계산한다.
    Supabase 패키지 미설치 또는 데이터 없으면 현재 universe 그대로 fallback.
    """
    # Supabase 패키지 없으면 기존 방식(yfinance/DART 현재 데이터)으로 단일 백테스트
    if not SUPABASE_AVAILABLE:
        logger.warning("supabase 패키지 없음 — 현재 재무 데이터로 fallback")
        scores      = compute_composite_scores(universe, factors)
        top_tickers = scores.head(TOP_N).index.tolist()
        period_returns, equity = run_equal_weight_backtest(
            top_tickers, start, end + timedelta(days=1), interval
        )
        return period_returns, equity, top_tickers

    tickers_all = universe.index.tolist()
    years = list(range(start.year, end.year + 1))

    all_returns: list[pd.Series] = []
    last_top_tickers: list[str] = []

    for year in years:
        seg_start = max(start, datetime(year, 1, 1))
        seg_end   = min(end,   datetime(year, 12, 31))

        if seg_start >= seg_end:
            continue

        fin_year  = year - 1
        fin_df    = _load_kr_financials_from_supabase(tickers_all, fin_year)
        year_univ = _apply_supabase_financials(universe, fin_df)

        logger.info(
            "KR 리밸런싱 %d년 (재무 기준: %d년%s): %s ~ %s",
            year, fin_year,
            f" [{len(fin_df)}종목 Supabase]" if not fin_df.empty else " [현재 데이터 fallback]",
            seg_start.date(), seg_end.date(),
        )

        try:
            scores      = compute_composite_scores(year_univ, factors)
            top_tickers = scores.head(TOP_N).index.tolist()
        except HTTPException:
            if last_top_tickers:
                logger.warning("  %d년 팩터 계산 실패 — 직전 포트폴리오 유지", year)
                top_tickers = last_top_tickers
            else:
                continue

        last_top_tickers = top_tickers

        # yfinance end는 exclusive이므로 +1일 전달
        bt_end = seg_end + timedelta(days=1)
        try:
            seg_returns, _ = run_equal_weight_backtest(
                top_tickers, seg_start, bt_end, interval
            )
        except HTTPException as exc:
            logger.warning("  %d년 백테스트 실패: %s — 스킵", year, exc.detail)
            continue

        if not seg_returns.empty:
            all_returns.append(seg_returns)

    if not all_returns:
        raise HTTPException(
            status_code=503,
            detail="연도별 백테스트 수익률을 계산할 수 없습니다.",
        )

    combined = pd.concat(all_returns).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    equity   = (1 + combined).cumprod()

    return combined, equity, last_top_tickers


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

        # 연도별 전년도 재무 데이터로 연초 리밸런싱 (룩어헤드 바이어스 방지)
        period_returns, equity, top_tickers = run_kr_annual_rebalancing_backtest(
            universe=universe,
            factors=request.factors,
            start=start,
            end=end,
            interval=request.interval,
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
    """yfinance .KS 티커로 한국 종목 상세 정보 조회 (PER/PBR/ROE는 DART 우선)"""
    ks = ticker if ticker.upper().endswith(".KS") else f"{ticker}.KS"
    try:
        t = yf.Ticker(ks)
        info = t.info or {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"yfinance 오류: {e}")

    # DART 종목명 → KR_TICKER_NAMES fallback → yfinance longName
    dart_names = _load_dart_kr_names()  # 부수 효과: _kr_corp_codes 채움
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

    # ── DART PER/PBR/ROE 계산 ────────────────────────────────────
    dart_per: float | None = None
    dart_pbr: float | None = None
    dart_roe: float | None = None

    base_code = ks.split(".")[0]
    corp_code = _kr_corp_codes.get(base_code)
    if corp_code and market_cap:
        fin = _fetch_dart_financials(corp_code)
        ni = fin.get("net_income")    # 원
        eq = fin.get("total_equity")  # 원
        mc = float(market_cap)        # 원 (yfinance marketCap)

        if ni is not None and ni > 0:
            per_val = mc / ni          # 둘 다 원 단위
            if np.isfinite(per_val) and 0 < per_val < 1000:
                dart_per = round(per_val, 2)

        if eq is not None and eq > 0:
            pbr_val = mc / eq          # 둘 다 원 단위
            if np.isfinite(pbr_val) and 0 < pbr_val < 100:
                dart_pbr = round(pbr_val, 2)
            if ni is not None:
                roe_val = ni / eq
                if np.isfinite(roe_val):
                    dart_roe = round(roe_val, 4)

    # DART 값 우선, 없으면 yfinance fallback
    per = dart_per if dart_per is not None else safe_float("trailingPE")
    pbr = dart_pbr if dart_pbr is not None else safe_float("priceToBook")
    roe = dart_roe if dart_roe is not None else safe_float("returnOnEquity")

    return StockDetailResponse(
        ticker=ks,
        name=name,
        sector=info.get("sector") or None,
        industry=info.get("industry") or None,
        price_history=price_history,
        per=per,
        pbr=pbr,
        roe=roe,
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
