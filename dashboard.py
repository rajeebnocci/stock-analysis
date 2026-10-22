"""
dashboard.py - one-box stock dashboard.

Setup:  pip install streamlit yfinance pandas numpy plotly
Run:    streamlit run dashboard.py      (keep stock_report.py and analytics.py in the same folder)
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

import analytics as an
import stock_report as sr

st.set_page_config("Stock Dashboard", layout="wide")
show = lambda d: st.dataframe(d, use_container_width=True)


def tryit(fn, default=None):
    try:
        out = fn()
        return default if out is None else out
    except Exception:
        return default


@st.cache_data(ttl=300, show_spinner="Loading market data...")
def load(tk, period, bench):
    return sr.fetch(tk, period, bench)


@st.cache_data(ttl=900, show_spinner=False)
def extra(tk):
    t = yf.Ticker(tk)
    d = lambda f: tryit(f, pd.DataFrame())
    return dict(fin=d(lambda: t.financials), bs=d(lambda: t.balance_sheet), news=tryit(lambda: t.news, []),
                cal=tryit(lambda: t.calendar, {}), earn=d(lambda: t.earnings_dates),
                recs=d(lambda: t.recommendations), ins=d(lambda: t.insider_transactions),
                inst=d(lambda: t.institutional_holders), exps=tryit(lambda: t.options, ()))


@st.cache_data(ttl=300, show_spinner=False)
def chain(tk, exp):
    c = yf.Ticker(tk).option_chain(exp)
    return c.calls, c.puts


@st.cache_data(ttl=600, show_spinner=False)
def snap(tk):
    px, info = yf.Ticker(tk).history(period="2y", auto_adjust=True), tryit(lambda: yf.Ticker(tk).info, {})
    px.index = px.index.tz_localize(None)
    return an.screen_row(tk, px, info), px["Close"]


@st.cache_data(ttl=600, show_spinner=False)
def closes(tickers, period="3y"):
    return yf.download(list(tickers), period=period, progress=False)["Close"].ffill().dropna()


def price_chart(df, tk):
    d = df.iloc[-504:]
    f = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[.5, .12, .19, .19], vertical_spacing=.02)
    f.add_trace(go.Candlestick(x=d.index, open=d.Open, high=d.High, low=d.Low, close=d.Close, name=tk), 1, 1)
    for col in ("SMA50", "SMA200", "BB_up", "BB_lo"):
        f.add_trace(go.Scatter(x=d.index, y=d[col], name=col, line=dict(width=1)), 1, 1)
    f.add_trace(go.Bar(x=d.index, y=d.Volume, name="Volume"), 2, 1)
    f.add_trace(go.Scatter(x=d.index, y=d.RSI, name="RSI"), 3, 1)
    f.add_hline(y=70, line_dash="dash", row=3, col=1)
    f.add_hline(y=30, line_dash="dash", row=3, col=1)
    f.add_trace(go.Bar(x=d.index, y=d.MACD_hist, name="MACD hist"), 4, 1)
    f.add_trace(go.Scatter(x=d.index, y=d.MACD, name="MACD"), 4, 1)
    f.add_trace(go.Scatter(x=d.index, y=d.MACD_sig, name="Signal"), 4, 1)
    f.update_layout(height=800, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10))
    return f


def sig_table(sigs):
    t = pd.DataFrame(sigs, columns=["Signal", "s", "Reading"])
    t["View"] = t.pop("s").map({1: "BULL", -1: "BEAR", 0: "neutral"})
    return t[["Signal", "View", "Reading"]]


# ------------------------------------------------------------------ sidebar
st.sidebar.title("Stock Dashboard")
tk = st.sidebar.text_input("Ticker", "AAPL").strip().upper()
period = st.sidebar.selectbox("History", ["1y", "2y", "5y", "10y"], index=1)
bench = st.sidebar.text_input("Benchmark", "SPY").strip().upper()
with st.sidebar.expander("DCF assumptions (optional)"):
    g_in = st.number_input("FCF growth % (0 = auto)", 0.0, 50.0, 0.0)
    d_in = st.number_input("Discount rate % (0 = auto)", 0.0, 30.0, 0.0)
    tg_in = st.number_input("Terminal growth %", 0.0, 5.0, 2.5)
st.sidebar.caption("Free Yahoo Finance data, delayed. Not financial advice.")

tabs = st.tabs(["Overview", "Chart", "Financials", "Risk & Simulation", "Backtest", "Options",
                "News & Events", "Compare", "Portfolio", "Markets"])

# ------------------------------------------------- single-stock tabs
try:
    px, bench_px, info, cf = load(tk, period, bench)
except SystemExit as e:
    px = None
    for t in tabs[:7]:
        t.warning(str(e) or f"No data for {tk}")
if px is not None:
    df = sr.add_indicators(px.copy())
    close, price = df["Close"], df["Close"].iloc[-1]
    rs = sr.risk_stats(close, bench_px, 0.042)
    beta = sr.pick(info, "beta")
    beta = rs["beta"] if np.isnan(beta) else beta
    ex = extra(tk)
    dres, why = sr.dcf(info, cf, price, beta, g_in / 100 or None, d_in / 100 or None, tg_in / 100, 0.042)
    tsig, fsig = sr.tech_signals(df), sr.fund_signals(info, dres)
    ts, fs = sum(s[1] for s in tsig), sum(s[1] for s in fsig)
    comb = ((ts / len(tsig) if tsig else 0) + (fs / len(fsig) if fsig else 0)) / 2

    with tabs[0]:
        st.header(f"{tk} - {info.get('longName') or info.get('shortName') or ''}")
        c = st.columns(5)
        c[0].metric("Price", f"{price:,.2f}", f"{price / close.iloc[-2] - 1:+.2%}")
        c[1].metric("Market cap", sr.fmt(sr.pick(info, "marketCap"), "big"))
        c[2].metric("Technical", sr.verdict(ts, len(tsig)))
        c[3].metric("Fundamental", sr.verdict(fs, len(fsig)))
        c[4].metric("Overall", sr.verdict(comb * 100, 100))
        st.caption(f"{info.get('sector', '')} | {info.get('industry', '')}")
        pr = sr.period_returns(close)
        st.write("**Returns:** " + "  |  ".join(f"{k} {v:+.1%}" for k, v in pr.items()))
        a, b = st.columns(2)
        a.subheader("Technical signals")
        a.dataframe(sig_table(tsig), hide_index=True, use_container_width=True)
        b.subheader("Fundamental signals")
        b.dataframe(sig_table(fsig), hide_index=True, use_container_width=True)
        st.subheader("DCF valuation")
        if dres:
            c = st.columns(3)
            c[0].metric("Intrinsic value / share", f"${dres['iv']:,.2f}")
            c[1].metric("Margin of safety", f"{dres['margin']:+.1%}")
            c[2].metric("Assumptions", f"g {dres['growth']:.1%} / r {dres['disc']:.1%}")
            show(dres["sens"].round(2))
        else:
            st.info(f"DCF skipped: {why}")
        tgt = sr.pick(info, "targetMeanPrice")
        if not np.isnan(tgt):
            st.write(f"**Analyst target (mean):** ${tgt:,.2f} ({tgt / price - 1:+.1%}) - consensus: {info.get('recommendationKey', 'n/a')}")
        with st.expander("All fundamentals"):
            for sec, items in sr.fund_table(info).items():
                st.write(f"**{sec}**")
                st.write(" | ".join(f"{lb}: {sr.fmt(v, k)}" for lb, k, v in items))
        st.download_button("Download indicator data (CSV)", df.to_csv(), f"{tk}_indicators.csv")

    with tabs[1]:
        st.plotly_chart(price_chart(df, tk), use_container_width=True)

    with tabs[2]:
        fin, bs = ex["fin"], ex["bs"]
        rows = {k: an.row(fin, *n) for k, n in {"Revenue": ("Total Revenue",), "Net income": ("Net Income",),
                                                  "Operating income": ("Operating Income",)}.items()}
        fcf = an.row(cf, "Free Cash Flow")
        rows["Free cash flow"] = fcf
        tbl = pd.DataFrame({k: v for k, v in rows.items() if v is not None})
        if len(tbl):
            tbl.index = pd.to_datetime(tbl.index).year
            st.plotly_chart(go.Figure([go.Bar(x=tbl.index, y=tbl[c], name=c) for c in tbl]).update_layout(
                barmode="group", height=350, margin=dict(t=10)), use_container_width=True)
            show(tbl.T.map(lambda v: sr.fmt(v, "big")) if hasattr(tbl, "map") else tbl.T)
        else:
            st.info("Statements unavailable.")
        a, b = st.columns(2)
        pt = an.piotroski(fin, bs, cf)
        a.metric("Piotroski F-score", f"{sum(pt.values())} / {len(pt)}" if pt else "n/a",
                 help="8-9 strong, 0-2 weak")
        for k, v in pt.items():
            a.write(("PASS  " if v else "FAIL  ") + k)
        z = an.altman(fin, bs, sr.pick(info, "marketCap"))
        b.metric("Altman Z-score", f"{z:.2f}" if z is not None else "n/a",
                 "safe" if z and z > 2.99 else "grey zone" if z and z > 1.81 else "distress risk" if z else None)
        b.caption("Built for industrial companies; unreliable for banks and financials.")

    with tabs[3]:
        c = st.columns(4)
        for i, (k, v, f) in enumerate([("Ann. return", rs["ann_ret"], "pct"), ("Volatility", rs["vol"], "pct"),
                                       ("Sharpe", rs["sharpe"], "n"), ("Sortino", rs["sortino"], "n"),
                                       ("Max drawdown", rs["max_dd"], "pct"), ("1-day VaR 95%", rs["var95"], "pct"),
                                       ("Beta", beta, "n"), (f"Corr vs {bench}", rs["corr"], "n")]):
            c[i % 4].metric(k, sr.fmt(v, f))
        st.subheader("Monte Carlo: 1-year price paths")
        paths = an.monte_carlo(close)
        q = np.percentile(paths, [5, 25, 50, 75, 95], axis=1)
        x = np.arange(1, paths.shape[0] + 1)
        f = go.Figure([go.Scatter(x=x, y=q[i], name=n) for i, n in enumerate(["5th", "25th", "median", "75th", "95th"])])
        st.plotly_chart(f.update_layout(height=350, xaxis_title="trading days ahead", margin=dict(t=10)), use_container_width=True)
        end = paths[-1]
        st.write(f"Median in 1y: **${np.median(end):,.2f}** | 5th-95th: ${q[0, -1]:,.2f} - ${q[4, -1]:,.2f} | "
                 f"chance of gain: **{(end > price).mean():.0%}**")
        st.caption("Assumes future looks like the past; a rough guide to the range, not a forecast.")

    with tabs[4]:
        curves, stats = an.backtest(df)
        st.line_chart(curves)
        show(stats.style.format({"Total return": "{:.1%}", "CAGR": "{:.1%}", "Volatility": "{:.1%}",
                                 "Sharpe": "{:.2f}", "Max drawdown": "{:.1%}"}))
        st.caption("Long/flat, trades next day, no costs or taxes. Past results do not predict future returns.")

    with tabs[5]:
        if not ex["exps"]:
            st.info("No options listed for this ticker.")
        else:
            exp = st.selectbox("Expiry", ex["exps"])
            calls, puts = chain(tk, exp)
            c = st.columns(4)
            c[0].metric("Put/call (open interest)", sr.fmt(puts.openInterest.sum() / max(calls.openInterest.sum(), 1)))
            c[1].metric("Put/call (volume)", sr.fmt(puts.volume.sum() / max(calls.volume.sum(), 1)))
            c[2].metric("Max pain", f"${an.max_pain(calls, puts):,.2f}")
            c[3].metric("ATM implied vol", sr.fmt(calls.iloc[(calls.strike - price).abs().argmin()].impliedVolatility, "pct"))
            f = go.Figure()
            for n, d in (("Calls", calls), ("Puts", puts)):
                d = d[(d.impliedVolatility > 0.01) & d.strike.between(price * .7, price * 1.3)]
                f.add_trace(go.Scatter(x=d.strike, y=d.impliedVolatility, mode="markers+lines", name=n))
            st.plotly_chart(f.update_layout(title="Implied volatility by strike", height=350), use_container_width=True)
            a, b = st.columns(2)
            cols = ["strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"]
            a.write("Calls"); a.dataframe(calls[cols], hide_index=True)
            b.write("Puts"); b.dataframe(puts[cols], hide_index=True)

    with tabs[6]:
        st.subheader("Headlines")
        for n in ex["news"][:10]:
            c_ = n.get("content", n)
            url = (c_.get("canonicalUrl") or {}).get("url") or c_.get("link")
            if c_.get("title"):
                st.markdown(f"- [{c_['title']}]({url})" if url else f"- {c_['title']}")
        cal = ex["cal"]
        if cal:
            st.subheader("Upcoming events")
            st.write({k: str(v) for k, v in (cal.items() if isinstance(cal, dict) else cal.to_dict().items())})
        for title, d in (("Earnings history / surprises", ex["earn"]), ("Analyst recommendations", ex["recs"]),
                         ("Insider transactions", ex["ins"]), ("Institutional holders", ex["inst"])):
            if d is not None and len(d):
                st.subheader(title)
                show(d.head(15))

# ---------------------------------------------------------------- Compare
with tabs[7]:
    txt = st.text_area("Tickers to compare (peers or watchlist)", "AAPL, MSFT, GOOGL, AMZN, META")
    if st.button("Compare"):
        tks = [t.strip().upper() for t in txt.replace("\n", ",").split(",") if t.strip()]
        rows, series = [], {}
        for t in tks:
            r = tryit(lambda: snap(t))
            if r:
                rows.append(r[0]); series[t] = r[1]
        if rows:
            show(pd.DataFrame(rows).set_index("Ticker").style.format(
                {"Price": "{:,.2f}", "1Y return": "{:.1%}", "P/E": "{:.1f}", "Fwd P/E": "{:.1f}", "Net margin": "{:.1%}",
                 "ROE": "{:.1%}", "Debt/Equity": "{:.0f}", "RSI": "{:.0f}"}, na_rep="n/a"))
            s = pd.DataFrame(series).ffill().dropna()
            st.plotly_chart(go.Figure([go.Scatter(x=s.index, y=s[c] / s[c].iloc[0] * 100, name=c) for c in s])
                            .update_layout(title="Performance (start = 100)", height=380), use_container_width=True)

# -------------------------------------------------------------- Portfolio
with tabs[8]:
    txt = st.text_area("Holdings as TICKER:weight", "AAPL:40, MSFT:30, JNJ:20, XOM:10")
    if st.button("Analyse portfolio"):
        h = {}
        for p in txt.replace("\n", ",").split(","):
            if p.strip():
                k, _, w = p.partition(":")
                h[k.strip().upper()] = float(w) if w.strip() else 1.0
        if len(h) < 2:
            st.warning("Enter at least two holdings.")
        else:
            px_ = closes(tuple(h))
            res = an.portfolio(px_, [h[c] for c in px_.columns])
            c = st.columns(3)
            c[0].metric("Expected return (hist.)", f"{res['ret']:.1%}")
            c[1].metric("Volatility", f"{res['vol']:.1%}")
            c[2].metric("Sharpe", f"{res['sharpe']:.2f}")
            a, b = st.columns(2)
            a.plotly_chart(go.Figure(go.Heatmap(z=res["corr"].values, x=res["corr"].columns, y=res["corr"].columns,
                                                zmin=-1, zmax=1, colorscale="RdBu_r")).update_layout(title="Correlation", height=380),
                           use_container_width=True)
            fr = res["frontier"]
            f = go.Figure(go.Scatter(x=fr.vol, y=fr.ret, mode="markers", marker=dict(size=3), name="Random portfolios"))
            f.add_trace(go.Scatter(x=[res["vol"]], y=[res["ret"]], mode="markers", marker=dict(size=14, color="red"), name="Yours"))
            f.add_trace(go.Scatter(x=[res["best"][0]], y=[res["best"][1]], mode="markers", marker=dict(size=14, color="green"), name="Max Sharpe"))
            b.plotly_chart(f.update_layout(title="Risk vs return", height=380, xaxis_title="volatility", yaxis_title="return"),
                           use_container_width=True)
            st.write("**Max-Sharpe weights (from past data, treat as a rough guide):** "
                     + ", ".join(f"{k} {v:.0%}" for k, v in res["best_w"].items()))

# ---------------------------------------------------------------- Markets
with tabs[9]:
    MK = {"S&P 500": "^GSPC", "Nasdaq": "^IXIC", "Dow": "^DJI", "Russell 2000": "^RUT", "VIX": "^VIX",
          "10Y yield %": "^TNX", "US Dollar idx": "DX-Y.NYB", "Gold": "GC=F", "Oil (WTI)": "CL=F",
          "Bitcoin": "BTC-USD", "EUR/USD": "EURUSD=X"}
    m = tryit(lambda: closes(tuple(MK.values()), "1y"), None)
    if m is not None:
        m = m.rename(columns={v: k for k, v in MK.items()})
        yr = m[m.index.year == m.index[-1].year]
        show(pd.DataFrame({"Last": m.iloc[-1], "1D": m.iloc[-1] / m.iloc[-2] - 1, "1M": m.iloc[-1] / m.iloc[-22] - 1,
                           "YTD": m.iloc[-1] / yr.iloc[0] - 1}).style.format({"Last": "{:,.2f}", "1D": "{:+.2%}", "1M": "{:+.1%}", "YTD": "{:+.1%}"}))
    yc = tryit(lambda: yf.download(["^IRX", "^FVX", "^TNX", "^TYX"], period="5d", progress=False)["Close"].ffill().iloc[-1], None)
    if yc is not None:
        yc = yc.reindex(["^IRX", "^FVX", "^TNX", "^TYX"])
        st.plotly_chart(go.Figure(go.Scatter(x=["3M", "5Y", "10Y", "30Y"], y=yc.values, mode="lines+markers"))
                        .update_layout(title="US Treasury yield curve (%)", height=320), use_container_width=True)
