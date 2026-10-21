"""전략 모음.

모든 전략은 hist(리밸런싱일까지의 종가 DataFrame)를 받아
종목별 목표 비중(pd.Series)을 돌려준다. 비중 합이 1보다 작으면 나머지는 현금.
데이터가 부족해 판단할 수 없으면 None을 돌려준다(그동안 현금 보유).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MONTH = 21  # 한 달 ≈ 21 거래일


def _period_return(hist: pd.DataFrame, months: int, skip: int = 0) -> pd.Series | None:
    need = months * MONTH + skip * MONTH + 1
    if len(hist) < need:
        return None
    end = hist.iloc[-1 - skip * MONTH]
    begin = hist.iloc[-1 - skip * MONTH - months * MONTH]
    return end / begin - 1


# ---------------------------------------------------------------------------
def buy_and_hold(hist, weights=None):
    """처음에 한 번 사고 끝까지 보유."""
    cols = hist.columns
    if weights:
        return pd.Series(weights).reindex(cols).fillna(0.0)
    return pd.Series(1 / len(cols), index=cols)


buy_and_hold.rebalance_once = True


def equal_weight(hist):
    """모든 종목을 같은 비중으로 주기적으로 맞춤."""
    return pd.Series(1 / hist.shape[1], index=hist.columns)


def fixed_weight(hist, weights=None):
    """사용자가 정한 비중(예: 주식 60 / 채권 40)으로 주기적으로 맞춤."""
    if not weights:
        return equal_weight(hist)
    w = pd.Series(weights, dtype=float).reindex(hist.columns).fillna(0.0)
    total = w.sum()
    return w / total if total > 0 else equal_weight(hist)


def relative_momentum(hist, lookback=12, top_n=3, skip=0):
    """최근 lookback개월 수익률 상위 top_n 종목에 동일 비중."""
    r = _period_return(hist, lookback, skip)
    if r is None:
        return None
    picks = r.sort_values(ascending=False).head(top_n).index
    return pd.Series(1 / len(picks), index=picks)


def dual_momentum(hist, lookback=12, top_n=1, safe=None, skip=0):
    """상대 모멘텀으로 고른 뒤, 수익률이 0 이하인 종목은 안전자산(없으면 현금)으로 대체."""
    r = _period_return(hist, lookback, skip)
    if r is None:
        return None
    candidates = r.drop(safe) if safe in r.index else r
    picks = candidates.sort_values(ascending=False).head(top_n)
    w = pd.Series(0.0, index=hist.columns)
    slot = 1 / top_n
    for t, ret in picks.items():
        if ret > 0:
            w[t] += slot
        elif safe in w.index:
            w[safe] += slot
        # 안전자산이 없으면 그 몫은 현금으로 남김
    return w


def trend_following(hist, ma_days=200):
    """종목별로 가격이 이동평균선 위면 보유(1/N), 아래면 그 몫을 현금으로."""
    if len(hist) < ma_days:
        return None
    ma = hist.iloc[-ma_days:].mean()
    last = hist.iloc[-1]
    n = hist.shape[1]
    return (last > ma).astype(float) / n


def inverse_volatility(hist, vol_days=60):
    """변동성이 낮은 종목에 더 많이: 비중 ∝ 1/변동성."""
    if len(hist) < vol_days + 1:
        return None
    vol = hist.iloc[-vol_days - 1:].pct_change().dropna().std()
    vol = vol.replace(0, np.nan).dropna()
    inv = 1 / vol
    return inv / inv.sum()


# ---------------------------------------------------------------------------
# 화면에 표시할 전략 정보 (이름, 함수, 설명, 파라미터 정의)
# 파라미터 정의: (키, 표시 이름, 종류, 기본값, 최소, 최대, 도움말)
# ---------------------------------------------------------------------------
STRATEGIES = {
    "매수 후 보유 (Buy & Hold)": {
        "fn": buy_and_hold,
        "params": [],
        "summary": "처음에 동일 비중으로 사서 그대로 둡니다. 다른 전략이 이것보다 나은지 비교하는 기준선이에요.",
    },
    "동일 비중 리밸런싱": {
        "fn": equal_weight,
        "params": [],
        "summary": "모든 종목을 1/N로 맞추고 주기마다 다시 맞춥니다. 오른 종목을 일부 팔고 내린 종목을 사는 효과가 생겨요.",
    },
    "고정 비중 자산배분": {
        "fn": fixed_weight,
        "params": [("weights", "종목별 비중", "weights", None, None, None,
                    "예: 주식 60%, 채권 40%. 합이 100%가 아니면 비율대로 맞춥니다.")],
        "summary": "60/40 포트폴리오처럼 정해 둔 비중을 유지합니다. 정적 자산배분의 기본형이에요.",
    },
    "상대 모멘텀 (Top N)": {
        "fn": relative_momentum,
        "params": [("lookback", "수익률 측정 기간(개월)", "int", 12, 1, 24,
                    "최근 몇 개월 수익률로 순위를 매길지"),
                   ("top_n", "보유 종목 수", "int", 3, 1, 10, "상위 몇 개를 살지"),
                   ("skip", "최근 제외 기간(개월)", "int", 0, 0, 3,
                    "개별 주식은 최근 1개월이 되돌림되는 경향이 있어 빼고 재기도 해요(12-1 모멘텀).")],
        "summary": "최근 많이 오른 자산이 당분간 더 오르는 경향(모멘텀)을 이용합니다.",
    },
    "듀얼 모멘텀": {
        "fn": dual_momentum,
        "params": [("lookback", "수익률 측정 기간(개월)", "int", 12, 1, 24, ""),
                   ("top_n", "보유 종목 수", "int", 1, 1, 5, ""),
                   ("safe", "안전자산", "asset", None, None, None,
                    "모멘텀이 꺾이면 피신할 자산. '현금'이면 현금 보유."),
                   ("skip", "최근 제외 기간(개월)", "int", 0, 0, 3, "")],
        "summary": "상대 모멘텀으로 고른 자산이라도 자기 수익률이 마이너스면 안전자산으로 피합니다. 하락장 방어가 목적이에요.",
    },
    "추세추종 (이동평균)": {
        "fn": trend_following,
        "params": [("ma_days", "이동평균 기간(거래일)", "int", 200, 20, 300,
                    "200일선이 널리 쓰여요. 짧을수록 반응은 빠르지만 잦은 매매가 생겨요.")],
        "summary": "가격이 이동평균선 위에 있는 자산만 보유하고, 아래로 내려가면 그 몫은 현금으로 둡니다.",
    },
    "역변동성 (리스크 균형)": {
        "fn": inverse_volatility,
        "params": [("vol_days", "변동성 측정 기간(거래일)", "int", 60, 20, 252, "")],
        "summary": "덜 흔들리는 자산에 돈을 더 넣어서, 각 자산이 포트폴리오 위험에 비슷하게 기여하도록 합니다.",
    },
}
