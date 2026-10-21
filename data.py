"""데이터 수집 모듈.

- 한국 종목(6자리 숫자 코드): yfinance(.KS/.KQ) → 실패 시 FinanceDataReader
- 미국 종목(영문 티커): yfinance
- 환율(원/달러): yfinance 'KRW=X' → 실패 시 FinanceDataReader 'USD/KRW'
- 연습용 데이터: 인터넷 없이 원리를 익힐 수 있도록 만든 가상 주가
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
CACHE_HOURS = 12  # 이 시간 안에 받은 데이터는 다시 내려받지 않음

# ---------------------------------------------------------------------------
# 종목 프리셋
# ---------------------------------------------------------------------------
NAMES = {
    # 미국 ETF
    "SPY": "S&P500", "QQQ": "나스닥100", "EFA": "선진국 주식", "EEM": "신흥국 주식",
    "TLT": "미국 장기채", "IEF": "미국 중기채", "SHY": "미국 단기채", "GLD": "금",
    "DBC": "원자재", "VNQ": "미국 리츠", "BIL": "미국 초단기채(현금성)",
    # 미국 대형주
    "AAPL": "애플", "MSFT": "마이크로소프트", "NVDA": "엔비디아", "AMZN": "아마존",
    "GOOGL": "알파벳", "META": "메타", "TSLA": "테슬라", "JPM": "JP모건",
    "BRK-B": "버크셔해서웨이", "JNJ": "존슨앤존슨",
    # 한국 ETF
    "069500": "KODEX 200", "229200": "KODEX 코스닥150", "148070": "KOSEF 국고채10년",
    "132030": "KODEX 골드선물(H)", "153130": "KODEX 단기채권",
    "360750": "TIGER 미국S&P500", "133690": "TIGER 미국나스닥100",
    # 한국 대형주
    "005930": "삼성전자", "000660": "SK하이닉스", "035420": "NAVER", "005380": "현대차",
    "051910": "LG화학", "035720": "카카오", "068270": "셀트리온", "105560": "KB금융",
    "055550": "신한지주", "005490": "POSCO홀딩스",
}

PRESETS = {
    "글로벌 자산배분 (미국 ETF)": ["SPY", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ"],
    "한국 자산배분 (국내 ETF)": ["069500", "229200", "148070", "132030", "360750", "133690"],
    "한·미 혼합 자산배분": ["069500", "SPY", "QQQ", "EFA", "TLT", "148070", "GLD"],
    "한국 대형주": ["005930", "000660", "035420", "005380", "051910", "068270", "105560", "005490"],
    "미국 대형주": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "JPM", "BRK-B", "JNJ"],
}


def is_korean(ticker: str) -> bool:
    return ticker.isdigit() and len(ticker) == 6


def label(ticker: str) -> str:
    name = NAMES.get(ticker)
    return f"{name} ({ticker})" if name else ticker


# ---------------------------------------------------------------------------
# 캐시
# ---------------------------------------------------------------------------
def _cache_path(key: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
    return os.path.join(CACHE_DIR, f"{safe}.csv")


def _read_cache(key: str) -> pd.Series | None:
    path = _cache_path(key)
    if not os.path.exists(path):
        return None
    if time.time() - os.path.getmtime(path) > CACHE_HOURS * 3600:
        return None
    try:
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        return s.dropna()
    except Exception:
        return None


def _write_cache(key: str, s: pd.Series) -> None:
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        s.to_frame("close").to_csv(_cache_path(key))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 개별 다운로드
# ---------------------------------------------------------------------------
def _clean(s) -> pd.Series | None:
    if s is None:
        return None
    if isinstance(s, pd.DataFrame):
        if s.shape[1] == 0:
            return None
        s = s.iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce").dropna()
    s = s[s > 0]
    if len(s) < 30:
        return None
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.astype(float)


def _yf(symbol: str, start: str) -> pd.Series | None:
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        df = yf.download(symbol, start=start, auto_adjust=True, progress=False, threads=False)
        if df is None or df.empty:
            return None
        close = df["Close"]
        return _clean(close)
    except Exception:
        return None


def _fdr(symbol: str, start: str) -> pd.Series | None:
    try:
        import FinanceDataReader as fdr
    except ImportError:
        return None
    try:
        df = fdr.DataReader(symbol, start)
        if df is None or df.empty:
            return None
        return _clean(df["Close"])
    except Exception:
        return None


def fetch_one(ticker: str, start: str = "2000-01-01") -> tuple[pd.Series | None, str]:
    """종목 하나의 (수정)종가를 가져온다. (시리즈, 출처) 반환."""
    cached = _read_cache(ticker)
    if cached is not None:
        return cached, "캐시"

    if is_korean(ticker):
        attempts = [("yfinance", lambda: _yf(f"{ticker}.KS", start)),
                    ("yfinance", lambda: _yf(f"{ticker}.KQ", start)),
                    ("FinanceDataReader", lambda: _fdr(ticker, start))]
    else:
        attempts = [("yfinance", lambda: _yf(ticker, start)),
                    ("FinanceDataReader", lambda: _fdr(ticker, start))]

    for source, fn in attempts:
        s = fn()
        if s is not None:
            _write_cache(ticker, s)
            return s, source
    return None, "실패"


def fetch_usdkrw(start: str = "2000-01-01") -> pd.Series | None:
    cached = _read_cache("USDKRW")
    if cached is not None:
        return cached
    s = _yf("KRW=X", start)
    if s is None:
        s = _fdr("USD/KRW", start)
    if s is not None:
        s = s[(s > 500) & (s < 3000)]  # 비정상 값 제거
        _write_cache("USDKRW", s)
    return s


# ---------------------------------------------------------------------------
# 연습용 가상 데이터
# ---------------------------------------------------------------------------
# (연 기대수익률, 연 변동성, 그룹) — 실제 자산군의 대략적 성격만 흉내 낸 값
_SAMPLE_PROFILE = {
    "bond": (0.035, 0.07), "short": (0.02, 0.01), "gold": (0.05, 0.15),
    "commodity": (0.03, 0.20), "equity": (0.08, 0.17), "growth": (0.11, 0.24),
    "stock": (0.09, 0.26), "reit": (0.06, 0.20),
}
_SAMPLE_KIND = {
    "TLT": "bond", "IEF": "bond", "148070": "bond", "SHY": "short", "BIL": "short",
    "153130": "short", "GLD": "gold", "132030": "gold", "DBC": "commodity", "VNQ": "reit",
    "QQQ": "growth", "229200": "growth", "133690": "growth",
}


def _sample_kind(t: str) -> str:
    if t in _SAMPLE_KIND:
        return _SAMPLE_KIND[t]
    if t in PRESETS["한국 대형주"] or t in PRESETS["미국 대형주"]:
        return "stock"
    return "equity"


def make_sample(tickers: list[str], start: str = "2008-01-01", end: str | None = None,
                seed: int = 9) -> tuple[pd.DataFrame, pd.Series]:
    """주식-채권 음의 상관, 국면 전환(약세장) 등을 넣은 가상 주가와 환율."""
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    idx = pd.bdate_range(start, end)
    n = len(idx)
    rng = np.random.default_rng(seed)

    # 시장 국면: 평소 / 약세장(변동성↑, 수익률↓)
    regime = np.zeros(n, dtype=int)
    state, i = 0, 0
    while i < n:
        length = rng.integers(40, 160) if state == 1 else rng.integers(300, 1000)
        regime[i:i + length] = state
        i += length
        state = 1 - state
    market = rng.standard_normal(n)  # 공통 주식 요인
    dt = 1 / 252

    data = {}
    for k, t in enumerate(tickers):
        mu, vol = _SAMPLE_PROFILE[_sample_kind(t)]
        kind = _sample_kind(t)
        beta = {"bond": -0.35, "short": 0.0, "gold": 0.05, "commodity": 0.4,
                "reit": 0.7}.get(kind, 0.8)
        own = np.random.default_rng(seed + 1000 + k).standard_normal(n)
        z = beta * market + np.sqrt(max(1 - beta ** 2, 0.05)) * own
        bear = regime == 1
        drift = np.where(bear, mu - (0.25 if beta > 0.3 else -0.03), mu + (0.04 if beta > 0.3 else 0.0))
        sigma = np.where(bear, vol * 1.6, vol)
        r = (drift - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
        data[t] = 100 * np.exp(np.cumsum(r))

    prices = pd.DataFrame(data, index=idx)
    fx_r = 0.0 * dt + 0.08 * np.sqrt(dt) * (-0.3 * market + 0.95 * rng.standard_normal(n))
    fx = pd.Series(1100 * np.exp(np.cumsum(fx_r)), index=idx, name="USDKRW")
    return prices, fx


# ---------------------------------------------------------------------------
# 통합 로더
# ---------------------------------------------------------------------------
@dataclass
class PriceData:
    prices: pd.DataFrame            # 정리된 종가 (모든 종목이 존재하는 구간)
    sources: dict = field(default_factory=dict)
    failed: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def load_prices(tickers: list[str], start: str, end: str, sample: bool = False,
                to_krw: bool = True, seed: int = 9) -> PriceData:
    tickers = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
    notes, failed, sources = [], [], {}

    if sample:
        raw, fx = make_sample(tickers, start="2005-01-01", seed=seed)
        sources = {t: "연습용 가상 데이터" for t in tickers}
    else:
        series = {}
        for t in tickers:
            s, src = fetch_one(t)
            if s is None:
                failed.append(t)
            else:
                series[t] = s
                sources[t] = src
        if not series:
            return PriceData(pd.DataFrame(), sources, failed,
                             ["데이터를 하나도 받지 못했습니다. 인터넷 연결과 종목 코드를 확인하세요."])
        raw = pd.DataFrame(series)
        fx = None
        if to_krw and any(not is_korean(t) for t in raw.columns):
            fx = fetch_usdkrw()
            if fx is None:
                notes.append("환율 데이터를 받지 못해 미국 자산은 달러 기준으로 계산했습니다.")

    # 한국·미국 휴장일이 달라서 합집합 날짜에 직전 가격을 채움
    raw = raw.sort_index().ffill()

    if to_krw and fx is not None:
        fx = fx.reindex(raw.index.union(fx.index)).sort_index().ffill().reindex(raw.index)
        for t in raw.columns:
            if not is_korean(t):
                raw[t] = raw[t] * fx

    raw = raw.loc[pd.Timestamp(start):pd.Timestamp(end)]
    first_valid = raw.apply(lambda c: c.first_valid_index())
    prices = raw.dropna()
    if prices.empty:
        notes.append("선택한 기간에 모든 종목이 함께 존재하는 구간이 없습니다.")
    else:
        late = first_valid[first_valid > raw.index[0] + pd.Timedelta(days=10)]
        if len(late):
            latest = late.idxmax()
            notes.append(f"{label(latest)}의 데이터가 {late.max():%Y-%m-%d}부터 있어서, "
                         f"분석은 모든 종목이 존재하는 이 날짜부터 시작합니다.")
    return PriceData(prices, sources, failed, notes)
