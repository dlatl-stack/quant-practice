"""퀀트 투자 연구실 — Streamlit 대시보드.

실행:  streamlit run app.py
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest import (drawdown, format_metrics, metrics, monthly_table, run_backtest)
from data import PRESETS, is_korean, label, load_prices
from lessons import LESSONS
from strategies import STRATEGIES

st.set_page_config(page_title="퀀트 투자 연구실", page_icon="📈", layout="wide",
                   initial_sidebar_state="auto")


def is_mobile() -> bool:
    """휴대폰 브라우저인지 (User-Agent 기준)."""
    try:
        ua = st.context.headers.get("User-Agent", "") or ""
    except Exception:
        return False
    return any(k in ua for k in ("Mobi", "Android", "iPhone", "iPod"))


MOBILE = is_mobile()

# 좁은 화면에서 여백·글자 크기 줄이기
st.markdown("""
<style>
@media (max-width: 640px) {
  .block-container { padding: 3.2rem 0.8rem 2rem 0.8rem !important; }
  h1 { font-size: 1.55rem !important; }
  h2, h3 { font-size: 1.15rem !important; }
  [data-testid="stMetricValue"] { font-size: 1.45rem !important; }
  [data-testid="stMetric"] { padding: 0.1rem 0; }
  [data-testid="stTabs"] button p { font-size: 0.85rem; }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 색상 (색각 이상 검증을 거친 범주형 팔레트, 라이트/다크 각각)
# 한국식 관례: 상승=빨강, 하락=파랑
# ---------------------------------------------------------------------------
PALETTE_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
PALETTE_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]


def is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


DARK = is_dark()
PAL = PALETTE_DARK if DARK else PALETTE_LIGHT
MID = "#383835" if DARK else "#f0efec"
UP, DOWN = ("#e66767", "#3987e5") if DARK else ("#d03b3b", "#2a78d6")
MUTED = "#8a8984"
DIVERGING = [[0.0, "#1c5cab"], [0.25, "#6da7ec"], [0.5, MID], [0.75, "#ec8a8a"], [1.0, "#b72e2e"]]


def color_map(names: list[str]) -> dict:
    """종목마다 고정 색. 8개를 넘으면 회색 계열로 접음."""
    cmap = {n: PAL[i] if i < len(PAL) else MUTED for i, n in enumerate(names)}
    cmap["현금"] = MUTED
    return cmap


def style(fig: go.Figure, height: int = 420, pct_y: bool = False, legend: bool = True) -> go.Figure:
    if MOBILE:
        fig.update_layout(dragmode=False, font_size=10, legend_font_size=10, title_font_size=13)
        height = min(height, 330) if height <= 440 else height
    fig.update_layout(
        height=height, margin=dict(l=4 if MOBILE else 10, r=4 if MOBILE else 10,
                                   t=44 if (MOBILE or not legend) else 78, b=6),
        hovermode="x unified", showlegend=legend,
        title=dict(x=0, xanchor="left", y=1, yanchor="top", yref="container", pad=dict(t=8)),
        legend=(dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0, title=None)
                if MOBILE else
                dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, title=None)),
        font=dict(family="Pretendard, 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif"),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    if pct_y:
        fig.update_yaxes(tickformat=".0%")
    return fig


def show(fig: go.Figure, key: str | None = None) -> None:
    """차트 출력. 휴대폰에서는 도구 막대와 확대를 꺼서 스크롤할 때 차트가 움직이지 않게 함."""
    config = {"displayModeBar": False, "scrollZoom": False} if MOBILE else {"displaylogo": False}
    st.plotly_chart(fig, key=key, config=config)


def line_chart(df: pd.DataFrame, cmap: dict, title: str = "", log: bool = False,
               pct: bool = False, fmt: str = ".2f", height: int = 420, width: dict | None = None) -> go.Figure:
    fig = go.Figure()
    for c in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[c], mode="lines",
            name=(c if MOBILE else label(c)) if c in cmap and c != "현금" else c,
            line=dict(width=(width or {}).get(c, 2), color=cmap.get(c, MUTED)),
            hovertemplate=f"%{{y:{'.1%' if pct else fmt}}}<extra>%{{fullData.name}}</extra>",
        ))
    fig.update_layout(title=title)
    if log:
        fig.update_yaxes(type="log")
    return style(fig, height, pct_y=pct, legend=df.shape[1] > 1)


def heatmap(tbl: pd.DataFrame, title: str, zmax: float, pct: bool = True, height: int = 420,
            cell_text: bool = True) -> go.Figure:
    if MOBILE:
        fmt = (lambda v: "0" if round(v * 100) == 0 else f"{round(v * 100):+d}") if pct else (lambda v: f"{v:.1f}")
    else:
        fmt = (lambda v: f"{v:+.1%}") if pct else (lambda v: f"{v:.2f}")
    text = tbl.map(lambda v: "" if pd.isna(v) else fmt(v))
    hover = tbl.map(lambda v: "" if pd.isna(v) else (f"{v:+.1%}" if pct else f"{v:.2f}"))
    fig = go.Figure(go.Heatmap(
        z=tbl.values, x=list(tbl.columns), y=[str(i) for i in tbl.index],
        colorscale=DIVERGING, zmin=-zmax, zmax=zmax, zmid=0,
        text=text.values, texttemplate="%{text}" if cell_text else "",
        textfont=dict(size=9 if MOBILE else 11), customdata=hover.values,
        xgap=2, ygap=2, showscale=False,
        hovertemplate="%{y} · %{x}: %{customdata}<extra></extra>",
    ))
    fig.update_layout(title=title)
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_xaxes(type="category", side="top")
    return style(fig, height, legend=False)


# ---------------------------------------------------------------------------
# 사이드바 — 데이터와 공통 설정
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 설정")
    mode = st.radio("데이터", ["연습용 데이터", "실제 시장 데이터"],
                    help="연습용: 인터넷 없이 원리를 익히는 가상 시장. 실제: yfinance·FinanceDataReader로 내려받음.")
    sample = mode == "연습용 데이터"
    seed = 9
    if sample:
        seed = st.number_input("가상 시장 번호", 1, 999, 9,
                               help="번호를 바꾸면 전혀 다른 가상 시장이 만들어져요. 같은 전략이 시장에 따라 어떻게 달라지는지 실험해 보세요.")

    preset = st.selectbox("종목 구성", list(PRESETS.keys()) + ["직접 입력"])
    default_list = ", ".join(PRESETS.get(preset, ["SPY", "TLT", "069500"]))
    raw = st.text_area("종목 코드 (쉼표로 구분)", default_list, key=f"tickers_{preset}",
                       help="한국: 6자리 코드(예: 005930, 069500)  /  미국: 티커(예: SPY, AAPL)")
    tickers = [t.strip().upper() for t in raw.replace("\n", ",").split(",") if t.strip()]

    start = st.date_input("시작일", dt.date(2010, 1, 1), min_value=dt.date(1995, 1, 1), format="YYYY-MM-DD")
    end = st.date_input("종료일", dt.date.today(), format="YYYY-MM-DD")

    if not sample and st.button("🔄 데이터 새로 받기", help="저장해 둔 데이터를 지우고 최신 데이터를 다시 내려받아요."):
        import shutil
        from data import CACHE_DIR
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
        st.cache_data.clear()

    to_krw = st.toggle("원화 기준으로 환산", True,
                       help="미국 자산 가격에 원/달러 환율을 곱해 한국 투자자가 실제로 체감하는 수익률로 계산해요.")

    st.divider()
    st.subheader("백테스트 공통 조건")
    freq = st.radio("리밸런싱 주기", ["매월", "분기", "매년"], horizontal=True)
    fee = st.number_input("거래비용 (편도, %)", 0.0, 2.0, 0.2, 0.05,
                          help="수수료+세금+호가 차이를 합쳐 매매 금액의 몇 %로 볼지") / 100
    rf = st.number_input("무위험 수익률 / 현금 이자 (연, %)", 0.0, 10.0, 2.5, 0.25,
                         help="현금으로 들고 있을 때 받는 이자이자, 샤프 지수 계산의 기준") / 100

if not tickers:
    st.warning("종목 코드를 하나 이상 입력하세요.")
    st.stop()
if start >= end:
    st.warning("시작일이 종료일보다 앞이어야 해요.")
    st.stop()


@st.cache_data(show_spinner="데이터를 불러오는 중…", ttl=3600 * 6)
def get_data(tickers, start, end, sample, to_krw, seed):
    return load_prices(list(tickers), str(start), str(end), sample=sample, to_krw=to_krw, seed=seed)


data = get_data(tuple(tickers), start, end, sample, to_krw, seed)
prices = data.prices

if data.failed:
    get_data.clear()  # 실패한 결과는 기억하지 않고 다음에 다시 시도
    rest = " (나머지 종목으로 계속 분석합니다)" if len(data.failed) < len(tickers) else ""
    st.error("데이터를 받지 못한 종목: " + ", ".join(data.failed)
             + " — 종목 코드와 인터넷 연결을 확인하세요." + rest)
for n in data.notes:
    st.info(n)
if prices.empty or len(prices) < 60:
    st.warning("분석할 데이터가 부족해요. 기간을 늘리거나 종목을 바꿔 보세요.")
    st.stop()

assets = list(prices.columns)
CMAP = color_map(assets)
ccy = "원화" if to_krw else "현지 통화"


@st.cache_data(show_spinner="백테스트 계산 중…")
def backtest_cached(prices: pd.DataFrame, sname: str, params: tuple, freq: str, fee: float, rf: float):
    p = {k: (dict(v) if isinstance(v, tuple) else v) for k, v in params}
    return run_backtest(prices, STRATEGIES[sname]["fn"], p, freq, fee, rf, sname)


def freeze(params: dict) -> tuple:
    return tuple(sorted((k, tuple(sorted(v.items())) if isinstance(v, dict) else v)
                        for k, v in params.items()))


SAFE_CANDIDATES = ["BIL", "SHY", "153130", "IEF", "148070", "TLT"]


def default_params(sname: str) -> dict:
    out = {}
    for key, _, kind, default, *_ in STRATEGIES[sname]["params"]:
        if kind == "int":
            out[key] = default
        elif kind == "asset":
            out[key] = next((t for t in SAFE_CANDIDATES if t in assets), None)
        elif kind == "weights":
            out[key] = {t: 1 / len(assets) for t in assets}
    return out


# ---------------------------------------------------------------------------
st.title("📈 퀀트 투자 연구실")
if MOBILE:
    st.caption("⚙️ 종목·기간 등 설정은 왼쪽 위 **›** 버튼을 누르면 열려요. 차트는 손가락으로 누르면 값이 보여요.")
src = "연습용 가상 데이터" if sample else "실제 시장 데이터"
st.caption(f"{src} · {prices.index[0]:%Y-%m-%d} ~ {prices.index[-1]:%Y-%m-%d} · "
           f"{len(assets)}개 종목 · {ccy} 기준 · 학습용 도구이며 투자 권유가 아닙니다")

tab_learn, tab_assets, tab_bt, tab_cmp = st.tabs(["📚 퀀트 기초", "🔍 자산 분석", "🧪 백테스트", "⚖️ 전략 비교"])

# ===========================================================================
# 📚 퀀트 기초
# ===========================================================================
with tab_learn:
    if MOBILE:
        lesson = st.selectbox("목차", [t for t, _ in LESSONS])
        st.markdown(dict(LESSONS)[lesson])
    else:
        left, right = st.columns([1, 3])
        with left:
            lesson = st.radio("목차", [t for t, _ in LESSONS], label_visibility="collapsed")
        with right:
            st.markdown(dict(LESSONS)[lesson])

# ===========================================================================
# 🔍 자산 분석
# ===========================================================================
with tab_assets:
    st.subheader("가격 흐름")
    c1, c2 = st.columns([1, 4])
    log = c1.toggle("로그 스케일", True, key="a_log",
                    help="로그 스케일에서는 같은 기울기 = 같은 수익률. 긴 기간 비교에 적합해요.")
    norm = prices / prices.iloc[0] * 100
    show(line_chart(norm, CMAP, f"시작일을 100으로 맞춘 가격 ({ccy})", log=log, fmt=",.1f"),
                    key="a_price")

    st.subheader("종목별 성과")
    rows = {label(t): format_metrics(metrics(prices[t], rf)) for t in assets}
    mt = pd.DataFrame(rows).T[["연평균 수익률(CAGR)", "연 변동성", "샤프 지수", "최대 낙폭(MDD)",
                               "최장 회복 기간(일)", "누적 수익률"]]
    st.dataframe(mt)

    c1, c2 = st.columns(2)
    with c1:
        dd = prices.apply(drawdown)
        show(line_chart(dd, CMAP, "낙폭 (고점 대비 하락률)", pct=True, height=380), key="a_dd")
    with c2:
        corr = prices.pct_change().dropna().corr()
        corr.index = corr.columns = assets if MOBILE else [label(t) for t in assets]
        show(heatmap(corr, "상관관계 (일간 수익률)", 1.0, pct=False, height=380), key="a_corr")
        st.caption("빨강일수록 함께 움직이고, 파랑일수록 반대로 움직여요. 상관이 낮은 자산을 섞어야 분산 효과가 커요.")

    st.subheader("모멘텀 순위")
    look = st.slider("수익률 측정 기간 (개월)", 1, 24, 12, key="a_look")
    days = look * 21
    if len(prices) > days:
        mom = (prices.iloc[-1] / prices.iloc[-1 - days] - 1).sort_values()
        fig = go.Figure(go.Bar(
            y=[label(t) for t in mom.index], x=mom.values, orientation="h",
            marker=dict(color=[UP if v > 0 else DOWN for v in mom.values], cornerradius=4),
            text=[f"{v:+.1%}" for v in mom.values], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x:+.1%}<extra></extra>",
        ))
        fig.update_layout(title=f"최근 {look}개월 수익률 ({prices.index[-1]:%Y-%m-%d} 기준)")
        lo_, hi_ = min(mom.min(), 0), max(mom.max(), 0)
        pad = (hi_ - lo_) * 0.15 or 0.05
        fig.update_xaxes(tickformat=".0%", range=[lo_ - (pad if lo_ < 0 else 0), hi_ + (pad if hi_ > 0 else 0)])
        show(style(fig, max(260, 44 * len(mom)), legend=False).update_layout(hovermode="closest"),
                        key="a_mom")
        roll = prices.pct_change(days).dropna()
        show(line_chart(roll, CMAP, f"{look}개월 수익률의 변화 (롤링)", pct=True, height=380,
                                   width={t: 1.5 for t in assets}),
                        key="a_roll")
    else:
        st.info("측정 기간보다 데이터가 짧아요.")

# ===========================================================================
# 🧪 백테스트
# ===========================================================================
with tab_bt:
    c1, c2 = st.columns([2, 1])
    sname = c1.selectbox("전략", list(STRATEGIES.keys()), index=4)
    bench_opts = ["동일 비중 매수 후 보유"] + [label(t) for t in assets]
    bench = c2.selectbox("비교 기준 (벤치마크)", bench_opts)
    st.caption(STRATEGIES[sname]["summary"])

    # --- 파라미터 입력 ---
    params = {}
    specs = STRATEGIES[sname]["params"]
    if specs:
        with st.container(border=True):
            st.markdown("**전략 설정**")
            int_specs = [s for s in specs if s[2] == "int"]
            cols = st.columns(max(len(int_specs), 1))
            for col, (key, name, kind, default, lo, hi, helptext) in zip(cols, int_specs):
                params[key] = col.slider(name, lo, hi, default, help=helptext or None, key=f"p_{sname}_{key}")
            for key, name, kind, default, lo, hi, helptext in specs:
                if kind == "asset":
                    opts = ["현금"] + assets
                    d = default_params(sname)[key]
                    choice = st.selectbox(name, opts, index=opts.index(d) if d in opts else 0,
                                          format_func=lambda t: t if t == "현금" else label(t),
                                          help=helptext, key=f"p_{sname}_{key}")
                    params[key] = None if choice == "현금" else choice
                elif kind == "weights":
                    st.caption(helptext)
                    wcols = st.columns(min(len(assets), 4))
                    w = {}
                    for i, t in enumerate(assets):
                        w[t] = wcols[i % len(wcols)].number_input(
                            label(t), 0, 100, round(100 / len(assets)), 5, key=f"w_{t}") / 100
                    params[key] = w

    res = backtest_cached(prices, sname, freeze(params), freq, fee, rf)
    if len(res.equity) < 30:
        st.warning("전략이 시작하기엔 데이터가 부족해요. 시작일을 앞당기거나 측정 기간을 줄여 보세요.")
        st.stop()

    # 벤치마크: 전략 시작일부터 같은 기간으로 맞춤
    if bench == "동일 비중 매수 후 보유":
        b = backtest_cached(prices.loc[res.equity.index[0]:], "매수 후 보유 (Buy & Hold)", (), freq, fee, rf).equity
    else:
        bt = assets[bench_opts.index(bench) - 1]
        b = prices[bt].loc[res.equity.index[0]:]
        b = b / b.iloc[0]
    b = b.reindex(res.equity.index).ffill()

    m, mb = metrics(res.equity, rf), metrics(b, rf)
    k = st.columns(5)
    k[0].metric("연평균 수익률", f"{m['연평균 수익률(CAGR)']:+.1%}",
                f"{(m['연평균 수익률(CAGR)'] - mb['연평균 수익률(CAGR)']) * 100:+.1f}%p vs 기준")
    k[1].metric("최대 낙폭", f"{m['최대 낙폭(MDD)']:.1%}",
                f"{(m['최대 낙폭(MDD)'] - mb['최대 낙폭(MDD)']) * 100:+.1f}%p vs 기준")
    k[2].metric("샤프 지수", f"{m['샤프 지수']:.2f}", f"{m['샤프 지수'] - mb['샤프 지수']:+.2f} vs 기준")
    k[3].metric("연 변동성", f"{m['연 변동성']:.1%}",
                f"{(m['연 변동성'] - mb['연 변동성']) * 100:+.1f}%p vs 기준", delta_color="inverse")
    k[4].metric("누적 거래비용", f"{res.costs:.1%}", f"리밸런싱 {len(res.weights)}회", delta_color="off")

    st.caption(f"전략 시작일 {res.equity.index[0]:%Y-%m-%d} (측정 기간만큼의 데이터가 쌓인 뒤 시작) · "
               f"{freq} 리밸런싱 · 거래비용 {fee:.2%} · 현금 이자 연 {rf:.1%}")

    c1, _ = st.columns([1, 4])
    log_bt = c1.toggle("로그 스케일", True, key="bt_log")
    comp = pd.DataFrame({sname: res.equity, f"기준: {bench}": b}) * 100
    ccmap = {sname: PAL[0], f"기준: {bench}": MUTED}
    show(line_chart(comp, ccmap, "자산 가치 (시작=100)", log=log_bt, fmt=",.1f",
                               width={f"기준: {bench}": 1.5}), key="bt_eq")

    ddc = pd.DataFrame({sname: drawdown(res.equity), f"기준: {bench}": drawdown(b)})
    show(line_chart(ddc, ccmap, "낙폭", pct=True, height=300), key="bt_dd")

    # --- 비중 변화 ---
    st.subheader("보유 비중의 변화")
    wts = res.weights.loc[:, (res.weights > 1e-9).any()]
    fig = go.Figure()
    for c in wts.columns:
        fig.add_trace(go.Scatter(
            x=wts.index, y=wts[c], name=label(c) if c != "현금" else "현금", stackgroup="one",
            mode="lines", line=dict(width=0.5, color=CMAP.get(c, MUTED), shape="hv"),
            fillcolor=CMAP.get(c, MUTED), hovertemplate="%{y:.0%}<extra>%{fullData.name}</extra>",
        ))
    show(style(fig, 340, pct_y=True).update_yaxes(range=[0, 1]), key="bt_w")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(f"**규칙이 계산한 최신 비중** ({prices.index[-1]:%Y-%m-%d} 종가 기준)")
        latest = STRATEGIES[sname]["fn"](prices, **params)
        if latest is None:
            st.info("데이터 부족")
        else:
            latest = latest.reindex(assets).fillna(0.0)
            latest["현금"] = max(0.0, 1 - latest.sum())
            latest = latest[latest > 1e-9].sort_values(ascending=False)
            st.dataframe(pd.DataFrame({"종목": [label(t) if t != "현금" else "현금" for t in latest.index],
                                       "비중": [f"{v:.1%}" for v in latest.values]}), hide_index=True)
            st.caption("학습용 계산 결과일 뿐 매매 추천이 아니에요.")
    with c2:
        st.markdown("**성과 지표 전체**")
        st.dataframe(pd.DataFrame({sname: format_metrics(m), f"기준: {bench}": format_metrics(mb)}))

    # --- 월별 수익률 ---
    st.subheader("월별 수익률")
    mtab = monthly_table(res.equity)
    zmax = float(np.nanpercentile(np.abs(mtab.drop(columns="연간").values), 95)) or 0.05
    if MOBILE:
        st.caption("칸을 누르면 수익률이 보여요. 숫자는 월, '연'은 연간 수익률이에요.")
        mtab.columns = [c.replace("월", "").replace("연간", "연") for c in mtab.columns]
    show(heatmap(mtab, "", zmax, height=max(260, (24 if MOBILE else 30) * len(mtab) + 80),
                 cell_text=not MOBILE), key="bt_month")

    # --- 파라미터 민감도 ---
    int_params = [s for s in specs if s[2] == "int"]
    if int_params:
        st.subheader("파라미터 민감도 (과최적화 점검)")
        st.caption("설정값 하나를 바꿔 가며 결과가 얼마나 흔들리는지 봐요. 특정 값에서만 유독 좋다면 우연일 가능성이 커요.")
        c1, c2 = st.columns([1, 3])
        pick = c1.selectbox("바꿔 볼 설정", [s[1] for s in int_params], key="sens_pick")
        key, name, _, _, lo, hi, _ = next(s for s in int_params if s[1] == pick)
        span = list(range(lo, hi + 1))
        if len(span) > 12:
            span = sorted(set(np.linspace(lo, hi, 10).round().astype(int).tolist() + [params[key]]))
        if c1.button("민감도 계산", type="primary"):
            out = []
            bar = st.progress(0.0)
            for i, v in enumerate(span):
                pp = dict(params, **{key: int(v)})
                r = backtest_cached(prices, sname, freeze(pp), freq, fee, rf)
                mm = metrics(r.equity, rf)
                out.append((v, mm.get("연평균 수익률(CAGR)", np.nan), mm.get("최대 낙폭(MDD)", np.nan),
                            mm.get("샤프 지수", np.nan)))
                bar.progress((i + 1) / len(span))
            bar.empty()
            sdf = pd.DataFrame(out, columns=[name, "CAGR", "MDD", "샤프"]).set_index(name)
            with c2:
                fig = go.Figure(go.Bar(
                    x=[str(v) for v in sdf.index], y=sdf["샤프"],
                    marker=dict(color=[PAL[1] if v == params[key] else PAL[0] for v in sdf.index], cornerradius=4),
                    customdata=np.c_[sdf["CAGR"], sdf["MDD"]],
                    hovertemplate=f"{name} %{{x}}<br>샤프 %{{y:.2f}}<br>CAGR %{{customdata[0]:+.1%}}"
                                  f"<br>MDD %{{customdata[1]:.1%}}<extra></extra>",
                ))
                fig.update_layout(title=f"{name}별 샤프 지수 (주황 = 현재 설정)")
                fig.update_xaxes(type="category")
                show(style(fig, 320, legend=False).update_layout(hovermode="closest"), key="bt_sens")
            st.dataframe(sdf.style.format({"CAGR": "{:+.1%}", "MDD": "{:.1%}", "샤프": "{:.2f}"}))

    st.download_button("📥 자산 가치·비중 CSV 내려받기",
                       pd.concat([res.equity.rename("전략"), b.rename("기준")], axis=1).to_csv().encode("utf-8-sig"),
                       file_name="backtest_result.csv", mime="text/csv")

# ===========================================================================
# ⚖️ 전략 비교
# ===========================================================================
with tab_cmp:
    st.caption("모든 전략을 기본 설정으로, 같은 기간·같은 비용 조건에서 비교해요. "
               "전략마다 시작 시점이 다르면 공정하지 않으니, 가장 늦게 시작한 전략의 시작일에 맞춥니다.")
    chosen = st.multiselect("비교할 전략", list(STRATEGIES.keys()),
                            default=["매수 후 보유 (Buy & Hold)", "동일 비중 리밸런싱", "듀얼 모멘텀",
                                     "추세추종 (이동평균)", "역변동성 (리스크 균형)"])
    if chosen:
        results = {s: backtest_cached(prices, s, freeze(default_params(s)), freq, fee, rf) for s in chosen}
        common = max(r.equity.index[0] for r in results.values())
        eq = pd.DataFrame({s: r.equity.loc[common:] / r.equity.loc[common] for s, r in results.items()})
        scmap = {s: PAL[i % len(PAL)] for i, s in enumerate(STRATEGIES)}

        c1, _ = st.columns([1, 4])
        log_c = c1.toggle("로그 스케일", True, key="cmp_log")
        show(line_chart(eq * 100, scmap, f"자산 가치 (시작=100, {common:%Y-%m-%d}부터)",
                                   log=log_c, fmt=",.1f"), key="cmp_eq")

        ms = {s: metrics(eq[s], rf) for s in eq}
        c1, c2 = st.columns([3, 2])
        with c1:
            show(line_chart(eq.apply(drawdown), scmap, "낙폭", pct=True, height=380), key="cmp_dd")
        with c2:
            fig = go.Figure()
            for s in eq:
                fig.add_trace(go.Scatter(
                    x=[ms[s]["연 변동성"]], y=[ms[s]["연평균 수익률(CAGR)"]], name=s, mode="markers+text",
                    marker=dict(size=13, color=scmap[s], line=dict(width=2, color="rgba(255,255,255,0.9)")),
                    text=[s.split(" (")[0]], textposition="top center", textfont=dict(size=11), cliponaxis=False,
                    hovertemplate=f"{s}<br>변동성 %{{x:.1%}}<br>CAGR %{{y:+.1%}}<extra></extra>",
                ))
            fig.update_layout(title="위험 vs 수익 (왼쪽 위일수록 좋음)")
            xs = [ms[s]["연 변동성"] for s in eq]
            ys = [ms[s]["연평균 수익률(CAGR)"] for s in eq]
            px_, py_ = (max(xs) - min(xs)) * 0.2 or 0.01, (max(ys) - min(ys)) * 0.25 or 0.01
            fig.update_xaxes(title="연 변동성", tickformat=".0%", range=[min(xs) - px_, max(xs) + px_])
            fig.update_yaxes(title="연평균 수익률", tickformat=".1%", range=[min(ys) - py_, max(ys) + py_])
            show(style(fig, 380, legend=False).update_layout(hovermode="closest"), key="cmp_rr")
            st.caption("색은 왼쪽 그래프의 전략 색과 같아요.")

        tbl = pd.DataFrame({s: format_metrics(ms[s]) for s in eq}).T
        st.dataframe(tbl[["연평균 수익률(CAGR)", "연 변동성", "샤프 지수", "소르티노 지수", "최대 낙폭(MDD)",
                          "칼마 지수", "월간 승률", "최장 회복 기간(일)", "누적 수익률"]])

        yearly = pd.DataFrame({s: monthly_table(eq[s])["연간"] for s in eq}).T
        zmax = float(np.nanpercentile(np.abs(yearly.values), 95)) or 0.1
        if MOBILE:
            yt = yearly.T
            yt.columns = [c.split(" (")[0].replace(" 리밸런싱", "").replace("매수 후 보유", "보유")
                          for c in yt.columns]
            st.markdown("**연도별 수익률 (%)**")
            show(heatmap(yt, "", zmax, height=60 + 26 * len(yt)), key="cmp_year")
        else:
            show(heatmap(yearly, "연도별 수익률", zmax, height=80 + 46 * len(yearly)), key="cmp_year")
        st.caption("어떤 해에 강하고 어떤 해에 약한지 보세요. 하락장(파랑으로 물든 해)에서의 차이가 전략의 성격을 잘 드러내요.")
