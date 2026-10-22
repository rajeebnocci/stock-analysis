"""analytics.py - extra analysis for the dashboard (no UI code)."""
import numpy as np
import pandas as pd

import stock_report as sr

TD = sr.TRADING_DAYS


def row(stmt, *names):
    """First matching line of a yfinance statement, oldest -> newest."""
    if stmt is None or stmt.empty:
        return None
    for n in names:
        if n in stmt.index:
            s = stmt.loc[n].dropna().sort_index()
            if len(s):
                return s
    return None


def piotroski(fin, bs, cf):
    """Piotroski F-score tests (True = pass). Tests with missing data are skipped."""
    g = row
    ni, ta, rev = g(fin, "Net Income"), g(bs, "Total Assets"), g(fin, "Total Revenue")
    gp, ltd = g(fin, "Gross Profit"), g(bs, "Long Term Debt")
    ca, cl = g(bs, "Current Assets"), g(bs, "Current Liabilities")
    sh = g(bs, "Ordinary Shares Number", "Share Issued")
    cfo = g(cf, "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
    tests = {}

    def t(name, fn):
        try:
            tests[name] = bool(fn())
        except Exception:
            pass

    t("Net income > 0", lambda: ni.iloc[-1] > 0)
    t("Operating cash flow > 0", lambda: cfo.iloc[-1] > 0)
    t("ROA improving", lambda: ni.iloc[-1] / ta.iloc[-1] > ni.iloc[-2] / ta.iloc[-2])
    t("Cash flow > net income", lambda: cfo.iloc[-1] > ni.iloc[-1])
    t("Leverage falling", lambda: ltd.iloc[-1] / ta.iloc[-1] < ltd.iloc[-2] / ta.iloc[-2])
    t("Current ratio improving", lambda: ca.iloc[-1] / cl.iloc[-1] > ca.iloc[-2] / cl.iloc[-2])
    t("No new shares issued", lambda: sh.iloc[-1] <= sh.iloc[-2])
    t("Gross margin improving", lambda: gp.iloc[-1] / rev.iloc[-1] > gp.iloc[-2] / rev.iloc[-2])
    t("Asset turnover improving", lambda: rev.iloc[-1] / ta.iloc[-1] > rev.iloc[-2] / ta.iloc[-2])
    return tests


def altman(fin, bs, mcap):
    """Altman Z-score (original). >2.99 safe, 1.81-2.99 grey, <1.81 distress. None if data missing."""
    try:
        L = lambda d, *n: float(row(d, *n).iloc[-1])
        ta = L(bs, "Total Assets")
        wc = L(bs, "Current Assets") - L(bs, "Current Liabilities")
        tl = L(bs, "Total Liabilities Net Minority Interest", "Total Liabilities")
        return (1.2 * wc / ta + 1.4 * L(bs, "Retained Earnings") / ta + 3.3 * L(fin, "EBIT") / ta
                + 0.6 * mcap / tl + L(fin, "Total Revenue") / ta)
    except Exception:
        return None


def backtest(df):
    """Simple long/flat strategies on the stock itself (trade next bar, no costs)."""
    c = df["Close"]
    r = c.pct_change().fillna(0)
    pos, st = [], 0
    for v in df["RSI"]:
        st = 1 if v < 30 else 0 if v > 55 else st
        pos.append(st)
    sigs = {"Buy & hold": pd.Series(1, index=c.index),
            "SMA 50/200 trend": (df.SMA50 > df.SMA200).astype(int),
            "MACD cross": (df.MACD > df.MACD_sig).astype(int),
            "RSI dip-buyer": pd.Series(pos, index=c.index)}
    curves, rows = {}, []
    for k, s in sigs.items():
        s = s.shift(1).fillna(0)
        sr_ = s * r
        eq = (1 + sr_).cumprod()
        curves[k] = eq
        vol = sr_.std() * np.sqrt(TD)
        rows.append({"Strategy": k, "Total return": eq.iloc[-1] - 1,
                     "CAGR": eq.iloc[-1] ** (TD / len(sr_)) - 1, "Volatility": vol,
                     "Sharpe": sr_.mean() * TD / vol if vol else np.nan,
                     "Max drawdown": (eq / eq.cummax() - 1).min(), "Trades": int((s.diff().abs() > 0).sum())})
    return pd.DataFrame(curves), pd.DataFrame(rows).set_index("Strategy")


def monte_carlo(close, days=252, n=1000, seed=0):
    """Geometric random-walk price paths (days x n) from historical log-return mean/vol."""
    r = np.log(close).diff().dropna()
    z = np.random.default_rng(seed).normal(size=(days, n))
    return close.iloc[-1] * np.exp(np.cumsum(r.mean() + r.std() * z, axis=0))


def max_pain(calls, puts):
    ks = sorted(set(calls.strike) | set(puts.strike))
    pain = lambda k: ((np.maximum(k - calls.strike, 0) * calls.openInterest.fillna(0)).sum()
                      + (np.maximum(puts.strike - k, 0) * puts.openInterest.fillna(0)).sum())
    return min(ks, key=pain)


def portfolio(prices, weights, rf=0.042, n=4000):
    r = prices.pct_change().dropna()
    w = np.array(weights, float)
    w /= w.sum()
    mu, cov = r.mean().values * TD, r.cov().values * TD
    ret, vol = w @ mu, np.sqrt(w @ cov @ w)
    W = np.random.default_rng(0).dirichlet(np.ones(len(w)), n)
    fr, fv = W @ mu, np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))
    b = ((fr - rf) / fv).argmax()
    return dict(ret=ret, vol=vol, sharpe=(ret - rf) / vol, corr=r.corr(),
                frontier=pd.DataFrame({"vol": fv, "ret": fr}),
                best_w=pd.Series(W[b], index=prices.columns), best=(fv[b], fr[b], (fr[b] - rf) / fv[b]))


def screen_row(tk, px, info):
    df = sr.add_indicators(px.copy())
    c, g = df["Close"], (lambda *k: sr.pick(info, *k))
    ts, fs = sr.tech_signals(df), sr.fund_signals(info, None)
    return {"Ticker": tk, "Price": c.iloc[-1],
            "1Y return": c.iloc[-1] / c.iloc[-253] - 1 if len(c) > 252 else np.nan,
            "P/E": g("trailingPE"), "Fwd P/E": g("forwardPE"), "Net margin": g("profitMargins"),
            "ROE": g("returnOnEquity"), "Debt/Equity": g("debtToEquity"), "RSI": df.RSI.iloc[-1],
            "Technical": sr.verdict(sum(s[1] for s in ts), len(ts)),
            "Fundamental": sr.verdict(sum(s[1] for s in fs), len(fs))}
