"""백테스트 엔진과 성과 지표.

규칙
- 리밸런싱일 종가까지의 데이터만 보고 목표 비중을 정한다(미래 데이터 사용 금지).
- 새 비중은 다음 거래일 수익률부터 적용된다.
- 리밸런싱 사이에는 가격 변동에 따라 비중이 자연스럽게 변한다(드리프트).
- 매매한 금액(회전율)만큼 거래비용을 차감한다.
- 목표 비중 합이 1보다 작으면 나머지는 현금(연 cash_rate 이자)으로 둔다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252
FREQ_RULE = {"매월": "ME", "분기": "QE", "매년": "YE"}


@dataclass
class BacktestResult:
    name: str
    equity: pd.Series          # 포트폴리오 가치 (시작=1)
    returns: pd.Series         # 일별 수익률
    weights: pd.DataFrame      # 리밸런싱 시점의 목표 비중 (현금 포함)
    turnover: pd.Series        # 리밸런싱별 회전율
    costs: float               # 누적 거래비용 비율(대략)


def rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """각 기간의 마지막 거래일."""
    s = pd.Series(index, index=index)
    last = s.groupby(pd.Grouper(freq=FREQ_RULE[freq])).last().dropna()
    return pd.DatetimeIndex(last.values)


def run_backtest(prices: pd.DataFrame, strategy, params: dict, freq: str = "매월",
                 fee: float = 0.002, cash_rate: float = 0.0, name: str = "") -> BacktestResult:
    """
    strategy(hist: DataFrame, **params) -> pd.Series(비중) 또는 None(데이터 부족, 현금 보유)
    fee: 매매 금액 대비 편도 비용 (0.002 = 0.2%)
    """
    prices = prices.dropna()
    rets = prices.pct_change().fillna(0.0)
    cash_daily = (1 + cash_rate) ** (1 / TRADING_DAYS) - 1

    rb_set = set(rebalance_dates(prices.index, freq))
    once = getattr(strategy, "rebalance_once", False)

    cols = list(prices.columns)
    w = np.zeros(len(cols))      # 현재 자산 비중
    cash = 1.0                   # 현재 현금 비중
    value = 1.0
    started = False

    eq_vals, day_rets = [], []
    w_records, to_records = {}, {}
    total_cost = 0.0

    r_mat = rets.values
    for i, day in enumerate(prices.index):
        # 1) 오늘 수익률 반영 (어제 정한 비중 기준)
        if i > 0:
            asset_growth = w * (1 + r_mat[i])
            cash_growth = cash * (1 + cash_daily)
            port_growth = asset_growth.sum() + cash_growth
            day_ret = port_growth - 1
            value *= port_growth
            if port_growth > 0:
                w = asset_growth / port_growth
                cash = cash_growth / port_growth
        else:
            day_ret = 0.0

        # 2) 오늘 종가 기준으로 리밸런싱 판단 → 내일부터 적용
        is_rb = (day in rb_set) or (i == 0)
        if once and started:
            is_rb = False
        if is_rb and i < len(prices) - 1:
            target = strategy(prices.iloc[: i + 1], **params)
            if target is not None:
                target = target.reindex(cols).fillna(0.0).clip(lower=0).values
                if target.sum() > 1:
                    target = target / target.sum()
                t_cash = 1 - target.sum()
                turnover = np.abs(target - w).sum()
                cost = turnover * fee
                value *= (1 - cost)
                total_cost += cost
                w, cash = target, t_cash
                started = True
                w_records[day] = np.append(target, t_cash)
                to_records[day] = turnover

        eq_vals.append(value)
        day_rets.append(day_ret)

    equity = pd.Series(eq_vals, index=prices.index)
    weights = pd.DataFrame.from_dict(w_records, orient="index", columns=cols + ["현금"])
    # 전략이 실제로 시작된 날(첫 비중 결정일)부터 보여줌
    if len(weights):
        first = weights.index[0]
        equity = equity.loc[first:] / equity.loc[first]
    returns = equity.pct_change().fillna(0.0)
    return BacktestResult(name, equity, returns, weights,
                          pd.Series(to_records, dtype=float), total_cost)


# ---------------------------------------------------------------------------
# 성과 지표
# ---------------------------------------------------------------------------
def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def metrics(equity: pd.Series, rf: float = 0.0) -> dict:
    equity = equity.dropna()
    if len(equity) < 2:
        return {}
    rets = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    total = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else np.nan
    vol = rets.std() * np.sqrt(TRADING_DAYS)
    excess = rets - ((1 + rf) ** (1 / TRADING_DAYS) - 1)
    sharpe = excess.mean() / rets.std() * np.sqrt(TRADING_DAYS) if rets.std() > 0 else np.nan
    downside = rets[rets < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = (excess.mean() * TRADING_DAYS) / downside if downside > 0 else np.nan
    dd = drawdown(equity)
    mdd = dd.min()
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    monthly = equity.resample("ME").last().pct_change().dropna()
    win = (monthly > 0).mean() if len(monthly) else np.nan

    # 가장 긴 손실 회복 기간(전고점 회복까지 걸린 날)
    underwater = dd < 0
    longest, cur_start = 0, None
    for d, u in underwater.items():
        if u and cur_start is None:
            cur_start = d
        elif not u and cur_start is not None:
            longest = max(longest, (d - cur_start).days)
            cur_start = None
    if cur_start is not None:
        longest = max(longest, (equity.index[-1] - cur_start).days)

    return {
        "누적 수익률": total, "연평균 수익률(CAGR)": cagr, "연 변동성": vol,
        "샤프 지수": sharpe, "소르티노 지수": sortino, "최대 낙폭(MDD)": mdd,
        "칼마 지수": calmar, "월간 승률": win, "최장 회복 기간(일)": longest,
        "기간(년)": years,
    }


PERCENT_KEYS = {"누적 수익률", "연평균 수익률(CAGR)", "연 변동성", "최대 낙폭(MDD)", "월간 승률"}


def format_metrics(m: dict) -> dict:
    out = {}
    for k, v in m.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out[k] = "-"
        elif k in PERCENT_KEYS:
            out[k] = f"{v:+.1%}" if k in {"누적 수익률", "연평균 수익률(CAGR)", "최대 낙폭(MDD)"} else f"{v:.1%}"
        elif k == "최장 회복 기간(일)":
            out[k] = f"{int(v):,}일"
        elif k == "기간(년)":
            out[k] = f"{v:.1f}년"
        else:
            out[k] = f"{v:.2f}"
    return out


def monthly_table(equity: pd.Series) -> pd.DataFrame:
    """연도 × 월 수익률 표 (마지막 열은 연간 수익률)."""
    m = equity.resample("ME").last()
    m = pd.concat([equity.iloc[:1], m]).sort_index()
    m = m[~m.index.duplicated(keep="last")].pct_change().dropna()
    tbl = pd.DataFrame({"연도": m.index.year, "월": m.index.month, "r": m.values})
    pivot = tbl.pivot_table(index="연도", columns="월", values="r", aggfunc="sum")
    pivot = pivot.reindex(columns=range(1, 13))
    pivot.columns = [f"{c}월" for c in pivot.columns]
    yearly = equity.resample("YE").last()
    yearly = pd.concat([equity.iloc[:1], yearly]).sort_index()
    yearly = yearly[~yearly.index.duplicated(keep="last")].pct_change().dropna()
    yearly.index = yearly.index.year
    pivot["연간"] = yearly
    return pivot
