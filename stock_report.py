#!/usr/bin/env python3
"""
stock_report.py - Bloomberg-style stock analysis in your terminal.

Setup:   pip install yfinance pandas numpy matplotlib
Usage:   python stock_report.py AAPL
         python stock_report.py NVDA --period 5y --benchmark QQQ --growth 0.12
         python stock_report.py MSFT --no-chart

Data: Yahoo Finance (free, delayed, occasionally patchy). Not financial advice.
"""
import argparse
import sys

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ------------------------------------------------------------------ helpers
def num(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def pick(info, *keys):
    """First valid numeric value among several possible Yahoo field names."""
    for k in keys:
        v = num(info.get(k))
        if not np.isnan(v):
            return v
    return np.nan


def fmt(x, kind="n"):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    if kind == "pct":
        return f"{x * 100:.2f}%"
    if kind == "usd":
        return f"${x:,.2f}"
    if kind == "big":
        for div, s in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
            if abs(x) >= div:
                return f"{x / div:.2f}{s}"
        return f"{x:,.0f}"
    return f"{x:,.2f}"


def header(title):
    print(f"\n{'=' * 66}\n {title}\n{'=' * 66}")


# --------------------------------------------------------------------- data
def fetch(ticker, period, benchmark):
    import yfinance as yf

    t = yf.Ticker(ticker)
    px = t.history(period=period, auto_adjust=True)
    if px.empty:
        sys.exit(f"No price data for '{ticker}'. Check the symbol.")
    px.index = px.index.tz_localize(None)
    bench = yf.Ticker(benchmark).history(period=period, auto_adjust=True)["Close"]
    bench.index = bench.index.tz_localize(None)
    try:
        info = t.info or {}
    except Exception:
        info = {}
    try:
        cf = t.cashflow
    except Exception:
        cf = pd.DataFrame()
    return px, bench, info, cf


# -------------------------------------------------------------- technicals
def add_indicators(df):
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    for n in (20, 50, 200):
        df[f"SMA{n}"] = c.rolling(n).mean()

    d = c.diff()  # RSI (Wilder smoothing)
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["RSI"] = 100 - 100 / (1 + up / dn)

    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["MACD"] = macd
    df["MACD_sig"] = macd.ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_sig"]

    sd = c.rolling(20).std()
    df["BB_up"], df["BB_lo"] = df["SMA20"] + 2 * sd, df["SMA20"] - 2 * sd
    df["BB_pctB"] = (c - df["BB_lo"]) / (df["BB_up"] - df["BB_lo"])

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    lo14, hi14 = l.rolling(14).min(), h.rolling(14).max()
    df["STOCH_K"] = 100 * (c - lo14) / (hi14 - lo14)
    df["OBV"] = (np.sign(c.diff()).fillna(0) * v).cumsum()
    return df


def tech_signals(df):
    x, out = df.iloc[-1], []

    def sig(name, bull, bear, note):
        out.append((name, 1 if bull else -1 if bear else 0, note))

    have = lambda *k: all(pd.notna(x[i]) for i in k)
    if have("SMA200"):
        sig("Price vs 200d SMA", x.Close > x.SMA200, x.Close < x.SMA200,
            f"{x.Close:.2f} vs {x.SMA200:.2f}")
    if have("SMA50"):
        sig("Price vs 50d SMA", x.Close > x.SMA50, x.Close < x.SMA50,
            f"{x.Close:.2f} vs {x.SMA50:.2f}")
    if have("SMA50", "SMA200"):
        sig("50d vs 200d (cross)", x.SMA50 > x.SMA200, x.SMA50 < x.SMA200,
            "golden-cross regime" if x.SMA50 > x.SMA200 else "death-cross regime")
    if have("RSI"):
        sig("RSI(14)", x.RSI < 30, x.RSI > 70, f"{x.RSI:.1f} (30 oversold / 70 overbought)")
    if have("MACD", "MACD_sig"):
        sig("MACD vs signal", x.MACD > x.MACD_sig, x.MACD < x.MACD_sig,
            f"hist {x.MACD_hist:+.3f}")
    if have("BB_pctB"):
        sig("Bollinger %B", x.BB_pctB < 0.05, x.BB_pctB > 0.95, f"{x.BB_pctB:.2f}")
    if have("STOCH_K"):
        sig("Stochastic %K", x.STOCH_K < 20, x.STOCH_K > 80, f"{x.STOCH_K:.1f}")
    obv_ma = df["OBV"].rolling(50).mean().iloc[-1]
    if pd.notna(obv_ma):
        sig("On-balance volume", x.OBV > obv_ma, x.OBV < obv_ma, "above 50d avg" if x.OBV > obv_ma else "below 50d avg")
    if len(df) > 126:
        m = df["Close"].iloc[-1] / df["Close"].iloc[-127] - 1
        sig("6-month momentum", m > 0.05, m < -0.05, fmt(m, "pct"))
    return out


# --------------------------------------------------------------------- risk
def risk_stats(close, bench, rf):
    r = close.pct_change().dropna()
    ann_ret = (close.iloc[-1] / close.iloc[0]) ** (TRADING_DAYS / len(r)) - 1
    vol = r.std() * np.sqrt(TRADING_DAYS)
    down = r[r < 0].std() * np.sqrt(TRADING_DAYS)
    dd = close / close.cummax() - 1
    j = pd.concat([r, bench.pct_change()], axis=1, join="inner").dropna()
    j.columns = ["s", "b"]
    beta = j.cov().iloc[0, 1] / j["b"].var() if len(j) > 30 else np.nan
    return dict(ann_ret=ann_ret, vol=vol, sharpe=(ann_ret - rf) / vol,
                sortino=(ann_ret - rf) / down if down else np.nan,
                max_dd=dd.min(), cur_dd=dd.iloc[-1], var95=np.percentile(r, 5),
                beta=beta, corr=j.corr().iloc[0, 1] if len(j) > 30 else np.nan)


def period_returns(close):
    out = {}
    for k, n in {"1M": 21, "3M": 63, "6M": 126, "1Y": 252}.items():
        if len(close) > n:
            out[k] = close.iloc[-1] / close.iloc[-1 - n] - 1
    prev = close[close.index.year < close.index[-1].year]
    out["YTD"] = close.iloc[-1] / (prev.iloc[-1] if len(prev) else close.iloc[0]) - 1
    return out


# ------------------------------------------------------------ fundamentals
def fund_table(i):
    p = lambda *k: pick(i, *k)
    return {
        "Valuation": [("Market cap", "big", p("marketCap")), ("P/E (trailing)", "n", p("trailingPE")),
                      ("P/E (forward)", "n", p("forwardPE")), ("PEG", "n", p("pegRatio", "trailingPegRatio")),
                      ("Price/Sales", "n", p("priceToSalesTrailing12Months")), ("Price/Book", "n", p("priceToBook")),
                      ("EV/EBITDA", "n", p("enterpriseToEbitda"))],
        "Profitability": [("Gross margin", "pct", p("grossMargins")), ("Operating margin", "pct", p("operatingMargins")),
                          ("Net margin", "pct", p("profitMargins")), ("ROE", "pct", p("returnOnEquity")),
                          ("ROA", "pct", p("returnOnAssets"))],
        "Growth (YoY)": [("Revenue", "pct", p("revenueGrowth")), ("Earnings", "pct", p("earningsGrowth"))],
        "Financial health": [("Debt/Equity (%)", "n", p("debtToEquity")), ("Current ratio", "n", p("currentRatio")),
                             ("Free cash flow", "big", p("freeCashflow")), ("Total cash", "big", p("totalCash")),
                             ("Total debt", "big", p("totalDebt"))],
        "Dividends": [("Yield", "pct", p("trailingAnnualDividendYield")), ("Payout ratio", "pct", p("payoutRatio"))],
        "Analyst targets": [("Mean", "usd", p("targetMeanPrice")), ("High", "usd", p("targetHighPrice")),
                            ("Low", "usd", p("targetLowPrice"))],
    }


def fund_signals(i, dcf_res):
    out, p = [], (lambda *k: pick(i, *k))

    def sig(name, v, good, bad, text):
        if not np.isnan(v):
            out.append((name, 1 if good(v) else -1 if bad(v) else 0, text))

    pe = p("trailingPE")
    sig("P/E (trailing)", pe, lambda v: v < 25, lambda v: v > 40, fmt(pe))
    peg = p("pegRatio", "trailingPegRatio")
    sig("PEG", peg, lambda v: 0 < v < 1.5, lambda v: v > 2.5, fmt(peg))
    nm = p("profitMargins")
    sig("Net margin", nm, lambda v: v > 0.15, lambda v: v < 0.05, fmt(nm, "pct"))
    roe = p("returnOnEquity")
    sig("Return on equity", roe, lambda v: v > 0.15, lambda v: v < 0.08, fmt(roe, "pct"))
    de = p("debtToEquity")
    sig("Debt/Equity", de, lambda v: v < 100, lambda v: v > 200, f"{fmt(de)}%")
    cr = p("currentRatio")
    sig("Current ratio", cr, lambda v: v > 1.5, lambda v: v < 1.0, fmt(cr))
    rg = p("revenueGrowth")
    sig("Revenue growth", rg, lambda v: v > 0.10, lambda v: v < 0, fmt(rg, "pct"))
    fcf = p("freeCashflow")
    sig("Free cash flow", fcf, lambda v: v > 0, lambda v: v <= 0, fmt(fcf, "big"))
    if dcf_res:
        mos = dcf_res["margin"]
        sig("DCF margin of safety", mos, lambda v: v > 0.15, lambda v: v < -0.15, fmt(mos, "pct"))
    return out


def dcf(info, cf, price, beta, growth, disc, tg, rf, erp=0.05, years=10):
    """Two-stage DCF on free cash flow; growth fades linearly to terminal rate."""
    hist = None
    if cf is not None and not cf.empty:
        if "Free Cash Flow" in cf.index:
            hist = cf.loc["Free Cash Flow"].dropna().sort_index()
        elif {"Operating Cash Flow", "Capital Expenditure"} <= set(cf.index):
            hist = (cf.loc["Operating Cash Flow"] + cf.loc["Capital Expenditure"]).dropna().sort_index()
    fcf = pick(info, "freeCashflow")
    if np.isnan(fcf) and hist is not None and len(hist):
        fcf = float(hist.iloc[-1])
    shares = pick(info, "sharesOutstanding", "impliedSharesOutstanding")
    if np.isnan(fcf) or fcf <= 0 or np.isnan(shares):
        return None, "needs positive free cash flow and share count (not available)"

    if growth is None:
        c = []
        if hist is not None and len(hist) >= 3 and hist.iloc[0] > 0 and hist.iloc[-1] > 0:
            c.append((hist.iloc[-1] / hist.iloc[0]) ** (1 / (len(hist) - 1)) - 1)
        c += [v for v in (pick(info, "revenueGrowth"), pick(info, "earningsGrowth")) if not np.isnan(v)]
        growth = float(np.clip(np.median(c), 0.0, 0.20)) if c else 0.05
    if disc is None:
        disc = float(np.clip(rf + (beta if np.isfinite(beta) else 1.0) * erp, 0.07, 0.12))
    net_debt = num(info.get("totalDebt"), 0.0) - num(info.get("totalCash"), 0.0)

    def value(g, r):
        r = max(r, tg + 0.01)
        pv, f = 0.0, fcf
        for y in range(1, years + 1):
            f *= 1 + g + (tg - g) * (y - 1) / (years - 1)
            pv += f / (1 + r) ** y
        tv = f * (1 + tg) / (r - tg) / (1 + r) ** years
        return (pv + tv - net_debt) / shares

    iv = value(growth, disc)
    gs = [max(growth - 0.03, 0), growth, growth + 0.03]
    rs = [disc - 0.01, disc, disc + 0.01]
    sens = pd.DataFrame([[value(g, r) for r in rs] for g in gs],
                        index=[f"g={g:.1%}" for g in gs], columns=[f"r={r:.1%}" for r in rs])
    return dict(iv=iv, margin=iv / price - 1, growth=growth, disc=disc, tg=tg, fcf=fcf, sens=sens), None


# ------------------------------------------------------------------ output
def verdict(score, n):
    p = score / n if n else 0
    return ("BULLISH" if p >= 0.4 else "LEAN BULLISH" if p >= 0.15 else "NEUTRAL" if p > -0.15
            else "LEAN BEARISH" if p > -0.4 else "BEARISH")


def print_signals(sigs):
    tag = {1: "BULL", -1: "BEAR", 0: "NEUT"}
    for name, pts, note in sigs:
        print(f" {name:<24}{tag[pts]:<6}{note}")
    score = sum(s[1] for s in sigs)
    print(f" {'-' * 60}\n {'Score':<24}{score:+d} / {len(sigs)}  ->  {verdict(score, len(sigs))}")
    return score, len(sigs)


def chart(df, ticker, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df.iloc[-TRADING_DAYS:]
    fig, ax = plt.subplots(4, 1, figsize=(12, 11), sharex=True,
                           gridspec_kw={"height_ratios": [4, 1.2, 1.5, 1.5]})
    ax[0].plot(d.index, d.Close, lw=1.4, label="Close")
    ax[0].plot(d.index, d.SMA50, lw=1, color="orange", label="SMA50")
    ax[0].plot(d.index, d.SMA200, lw=1, color="red", label="SMA200")
    ax[0].fill_between(d.index, d.BB_lo, d.BB_up, alpha=0.12, label="Bollinger(20,2)")
    ax[0].set_title(f"{ticker} - technical overview"); ax[0].legend(loc="upper left")
    ax[1].bar(d.index, d.Volume, color="gray"); ax[1].set_ylabel("Volume")
    ax[2].plot(d.index, d.RSI, color="purple"); ax[2].set_ylabel("RSI")
    ax[2].axhline(70, ls="--", color="red", lw=.8); ax[2].axhline(30, ls="--", color="green", lw=.8)
    ax[3].bar(d.index, d.MACD_hist, color=np.where(d.MACD_hist >= 0, "green", "red"))
    ax[3].plot(d.index, d.MACD, lw=1, label="MACD"); ax[3].plot(d.index, d.MACD_sig, lw=1, label="Signal")
    ax[3].set_ylabel("MACD"); ax[3].legend(loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Bloomberg-style stock report")
    ap.add_argument("ticker")
    ap.add_argument("--period", default="2y", help="history window, e.g. 1y 2y 5y (default 2y)")
    ap.add_argument("--benchmark", default="SPY")
    ap.add_argument("--growth", type=float, help="override DCF FCF growth, e.g. 0.10")
    ap.add_argument("--discount", type=float, help="override DCF discount rate, e.g. 0.09")
    ap.add_argument("--terminal", type=float, default=0.025, help="terminal growth (default 2.5%%)")
    ap.add_argument("--rf", type=float, default=0.042, help="risk-free rate (default 4.2%%)")
    ap.add_argument("--no-chart", action="store_true")
    a = ap.parse_args()
    tk = a.ticker.upper()

    px, bench, info, cf = fetch(tk, a.period, a.benchmark.upper())
    df = add_indicators(px.copy())
    close = df["Close"]
    price = close.iloc[-1]
    chg = price / close.iloc[-2] - 1
    rs = risk_stats(close, bench, a.rf)
    beta = pick(info, "beta")
    beta = rs["beta"] if np.isnan(beta) else beta

    header(f"{tk} - {info.get('longName') or info.get('shortName') or ''}")
    print(f" {info.get('sector', 'n/a')} | {info.get('industry', 'n/a')} | {info.get('currency', 'USD')}")
    print(f" Last: {price:,.2f} ({chg:+.2%})   52w range: {df.High.iloc[-252:].max():,.2f} / {df.Low.iloc[-252:].min():,.2f}")
    print(" Returns: " + "  ".join(f"{k} {v:+.1%}" for k, v in period_returns(close).items()))

    header("TECHNICAL INDICATORS")
    x = df.iloc[-1]
    rows = [("SMA 20 / 50 / 200", f"{x.SMA20:.2f} / {x.SMA50:.2f} / {x.SMA200:.2f}"),
            ("RSI(14)", f"{x.RSI:.1f}"), ("MACD / signal", f"{x.MACD:.3f} / {x.MACD_sig:.3f}"),
            ("Bollinger up / low", f"{x.BB_up:.2f} / {x.BB_lo:.2f}"),
            ("ATR(14)", f"{x.ATR:.2f} ({x.ATR / price:.1%} of price)"),
            ("20d support / resist.", f"{df.Low.iloc[-20:].min():.2f} / {df.High.iloc[-20:].max():.2f}")]
    for k, v in rows:
        print(f" {k:<24}{v}")
    header("TECHNICAL SIGNALS")
    ts, tn = print_signals(tech_signals(df))

    header(f"RISK (vs {a.benchmark.upper()})")
    for k, v in [("Annualised return", fmt(rs["ann_ret"], "pct")), ("Annualised volatility", fmt(rs["vol"], "pct")),
                 ("Sharpe / Sortino", f"{fmt(rs['sharpe'])} / {fmt(rs['sortino'])}"),
                 ("Max drawdown", fmt(rs["max_dd"], "pct")), ("Current drawdown", fmt(rs["cur_dd"], "pct")),
                 ("1-day VaR (95%)", fmt(rs["var95"], "pct")), ("Beta / correlation", f"{fmt(beta)} / {fmt(rs['corr'])}")]:
        print(f" {k:<24}{v}")

    header("FUNDAMENTALS")
    if not info:
        print(" Fundamental data unavailable from Yahoo right now.")
    for sec, items in fund_table(info).items():
        print(f" [{sec}]")
        for label, kind, v in items:
            print(f"   {label:<22}{fmt(v, kind)}")
    tgt = pick(info, "targetMeanPrice")
    if not np.isnan(tgt):
        print(f"   {'Implied upside':<22}{tgt / price - 1:+.1%}  (consensus: {info.get('recommendationKey', 'n/a')})")

    header("DCF VALUATION (10y, growth fades to terminal)")
    res, why = dcf(info, cf, price, beta, a.growth, a.discount, a.terminal, a.rf)
    if res:
        print(f" FCF growth {res['growth']:.1%} | discount {res['disc']:.1%} | terminal {res['tg']:.1%}")
        print(f" Intrinsic value/share  {fmt(res['iv'], 'usd')}   vs price {fmt(price, 'usd')}  ->  {res['margin']:+.1%}")
        print("\n Sensitivity (value/share):\n" + res["sens"].round(2).to_string())
    else:
        print(f" Skipped: {why}")

    header("FUNDAMENTAL SIGNALS")
    fs, fn = print_signals(fund_signals(info, res))

    header("OVERALL")
    comb = ((ts / tn if tn else 0) + (fs / fn if fn else 0)) / (2 if fn else 1)
    print(f" Technical:   {verdict(ts, tn)}\n Fundamental: {verdict(fs, fn)}\n"
          f" Overall:     {verdict(comb * 100, 100)}")
    if not a.no_chart:
        path = f"{tk}_report.png"
        chart(df, tk, path)
        print(f"\n Chart saved to {path}")
    print("\n Rule-based screen on delayed free data. Not financial advice.\n")


if __name__ == "__main__":
    main()
