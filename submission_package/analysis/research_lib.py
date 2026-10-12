"""
Track A research pipeline: volatility-adjusted cross-sectional strategy.

Stages
  1. load + EDA checks
  2. volatility estimators (compare, then pick one)
  3. signal construction + statistical tests (rank IC, Fama-MacBeth, quantile spread)
  4. vol-adjusted weights (inverse-vol, long/short, capped, periodic rebalance)
  5. quick sanity P&L (NOT for submission: final numbers must come from the
     organiser backtester; plug `build_weights()` output into it)

Only numpy + pandas. Fixed seed. No absolute paths.
Usage: python trackA_research.py --data Dataset_PS-A.csv
"""
import argparse
import os
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)
ANN = 252


# ---------------------------------------------------------------- 1. data
def load_wide(path):
    df = pd.read_csv(path, parse_dates=["date"]).sort_values(["date", "ticker"])
    assert not df.duplicated(["date", "ticker"]).any(), "duplicate date/ticker rows"
    wide = {c: df.pivot(index="date", columns="ticker", values=c)
            for c in ["open", "high", "low", "close", "volume"]}
    return wide


def eda_report(w):
    close = w["close"]
    ret = np.log(close).diff()
    print(f"tickers={close.shape[1]}  days={close.shape[0]}  "
          f"{close.index.min().date()} -> {close.index.max().date()}")
    print("missing close by ticker:\n", close.isna().sum()[close.isna().sum() > 0])
    print("max |daily log-ret| (possible unadjusted split if >0.4):")
    print(ret.abs().max().sort_values(ascending=False).head(3))
    r2 = (ret ** 2).stack()
    lag1 = ret.abs().apply(lambda s: s.autocorr(1)).mean()
    print(f"mean lag-1 autocorr of |ret| (vol clustering evidence): {lag1:.3f}")
    print(f"mean lag-1 autocorr of ret: {ret.apply(lambda s: s.autocorr(1)).mean():.3f}")
    print(f"avg pairwise corr: "
          f"{(ret.corr().values.sum() - close.shape[1]) / (close.shape[1] ** 2 - close.shape[1]):.3f}")


# ---------------------------------------------------------------- 2. vol
def vol_rolling(w, n=20):
    return np.log(w["close"]).diff().rolling(n).std()


def vol_ewma(w, halflife=10):
    r = np.log(w["close"]).diff()
    return np.sqrt((r ** 2).ewm(halflife=halflife, min_periods=halflife).mean())


def vol_garman_klass(w, n=20):
    o, h, l, c = w["open"], w["high"], w["low"], w["close"]
    gk = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    return np.sqrt(gk.rolling(n).mean().clip(lower=1e-12))


def qlike(forecast_vol, realized_ret, horizon=5):
    """Lower is better. Compare forecasts of variance vs realised variance over next `horizon` days."""
    f = forecast_vol ** 2
    real = (realized_ret ** 2).rolling(horizon).mean().shift(-horizon)
    m = f.notna() & real.notna() & (f > 0) & (real > 0)
    x = (real / f).where(m)
    return float((x - np.log(x) - 1).stack().mean())


# ---------------------------------------------------------------- 3. signals + tests
def signal_vol_adj_momentum(w, vol, lookback=60, skip=5):
    c = w["close"]
    mom = np.log(c.shift(skip) / c.shift(lookback))
    return mom / (vol * np.sqrt(lookback - skip))       # ~ t-stat-like momentum


def xs_zscore(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)


def forward_returns(close, horizon=5, lag=1):
    """Return earned from t+lag to t+lag+horizon; lag=1 = one-day execution delay."""
    return np.log(close.shift(-(lag + horizon)) / close.shift(-lag))


def nw_tstat(x, lags=None):
    x = pd.Series(x).dropna().values
    n = len(x)
    if n < 10:
        return np.nan
    e = x - x.mean()
    L = lags if lags is not None else int(4 * (n / 100) ** (2 / 9))
    s = e @ e / n
    for l in range(1, L + 1):
        s += 2 * (1 - l / (L + 1)) * (e[l:] @ e[:-l]) / n
    return x.mean() / np.sqrt(s / n)


def rank_ic(sig, fwd):
    m = sig.notna() & fwd.notna()
    s, f = sig.where(m).rank(axis=1), fwd.where(m).rank(axis=1)
    s, f = s.sub(s.mean(axis=1), axis=0), f.sub(f.mean(axis=1), axis=0)
    return (s * f).sum(axis=1) / np.sqrt((s ** 2).sum(axis=1) * (f ** 2).sum(axis=1))


def fama_macbeth(sig, fwd):
    z = xs_zscore(sig)
    m = z.notna() & fwd.notna()
    z, y = z.where(m), fwd.where(m)
    y = y.sub(y.mean(axis=1), axis=0)
    return (z * y).sum(axis=1) / (z ** 2).sum(axis=1)   # daily slope


def quantile_spread(sig, fwd, nq=5):
    pct = sig.rank(axis=1, pct=True)
    b = np.ceil(pct * nq).clip(1, nq)
    out = {q: fwd.where(b == q).mean(axis=1).mean() for q in range(1, nq + 1)}
    return pd.Series(out)


def signal_report(sig, fwd, name="signal"):
    ic = rank_ic(sig, fwd)
    fm = fama_macbeth(sig, fwd)
    q = quantile_spread(sig, fwd)
    print(f"\n== {name} ==")
    print(f"rank IC mean={ic.mean():.4f}  IC-IR={ic.mean() / ic.std():.3f}  NW t={nw_tstat(ic):.2f}")
    print(f"Fama-MacBeth slope mean={fm.mean():.5f}  NW t={nw_tstat(fm):.2f}")
    print("mean fwd return by quintile (1=low signal):\n", q.round(5).to_string())
    print(f"Q5-Q1 spread={q.iloc[-1] - q.iloc[0]:.5f}  monotonic={q.is_monotonic_increasing}")
    return ic, fm, q


# ---------------------------------------------------------------- 4. weights
def build_weights(sig, vol, k=5, rebal=5, max_w=0.15, warmup=80):
    """Dollar-neutral long/short, gross exposure = 1 (0.5 long + 0.5 short), no leverage.
    Position size within each side is proportional to 1/vol, capped at max_w."""
    W = pd.DataFrame(np.nan, index=sig.index, columns=sig.columns)
    for i in range(warmup, len(sig), rebal):
        s, v = sig.iloc[i].dropna(), vol.iloc[i].reindex(sig.columns)
        if len(s) < 2 * k:
            continue
        w = pd.Series(0.0, index=sig.columns)
        for side, names in ((+1, s.nlargest(k).index), (-1, s.nsmallest(k).index)):
            iv = (1 / v[names]).replace([np.inf, -np.inf], np.nan).fillna(0)
            x = iv / iv.sum() * 0.5
            for _ in range(10):                      # iterative cap + renormalise
                x = x.clip(upper=max_w)
                x = x / x.sum() * 0.5
            w[names] = side * x
        W.iloc[i] = w
    return W.ffill().fillna(0.0)


def concentration(W, ret_corr_window=None, ret=None):
    """Effective N (inverse HHI) of long book, and avg pairwise corr of held longs."""
    L = W.clip(lower=0)
    eff_n = 1 / ((L.div(L.sum(axis=1), axis=0)) ** 2).sum(axis=1).replace(0, np.nan)
    out = {"mean_effective_N_long": float(eff_n.mean())}
    if ret is not None:
        C = ret.corr().values
        held = (L > 0).mean()
        idx = np.where(held > 0.05)[0]
        sub = C[np.ix_(idx, idx)]
        out["avg_pairwise_corr_of_ever_held_longs"] = float((sub.sum() - len(idx)) / max(len(idx) ** 2 - len(idx), 1))
    return out


# ---------------------------------------------------------------- 5. sanity P&L only
def sanity_backtest(W, close, cost_bps=5, exec_lag=1):
    ret = close.pct_change()
    Wl = W.shift(1 + exec_lag)
    turn = W.diff().abs().sum(axis=1)
    pnl = (Wl * ret).sum(axis=1) - turn.shift(exec_lag) * cost_bps / 1e4
    return pnl.dropna()


def metrics(pnl):
    eq = (1 + pnl).cumprod()
    yrs = len(pnl) / ANN
    dd = (eq / eq.cummax() - 1).min()
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    sr = pnl.mean() / pnl.std() * np.sqrt(ANN)
    return dict(Sharpe=sr, CAGR=cagr, MaxDD=dd, Calmar=cagr / abs(dd) if dd < 0 else np.nan)


def block_bootstrap_sharpe(pnl, block=20, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    x, T = pnl.values, len(pnl)
    nb = int(np.ceil(T / block))
    out = np.empty(n)
    for b in range(n):
        starts = rng.integers(0, T - block, nb)
        s = np.concatenate([x[st:st + block] for st in starts])[:T]
        out[b] = s.mean() / s.std() * np.sqrt(ANN)
    return np.percentile(out, [2.5, 97.5])


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--skip", type=int, default=5)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--rebal", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=5)
    a = ap.parse_args()

    w = load_wide(a.data)
    eda_report(w)
    close = w["close"]
    ret = np.log(close).diff()

    print("\n== vol estimator comparison (QLIKE, lower=better) ==")
    vols = {"roll20": vol_rolling(w, 20), "roll60": vol_rolling(w, 60),
            "ewma_hl10": vol_ewma(w, 10), "garman_klass20": vol_garman_klass(w, 20)}
    for n, v in vols.items():
        print(f"{n:16s} {qlike(v, ret):.4f}")
    vol = vols["ewma_hl10"]          # replace with the winner from the table above

    sig = xs_zscore(signal_vol_adj_momentum(w, vol, a.lookback, a.skip))
    fwd = forward_returns(close, a.horizon, lag=1)
    signal_report(sig, fwd, f"vol-adj momentum L={a.lookback} skip={a.skip}")

    W = build_weights(sig, vol, k=a.k, rebal=a.rebal)
    print("\nconcentration:", concentration(W, ret=ret))
    pnl = sanity_backtest(W, close)
    print("sanity metrics (own calc, NOT the submission numbers):",
          {k: round(v, 3) for k, v in metrics(pnl).items()})
    lo, hi = block_bootstrap_sharpe(pnl)
    print(f"block-bootstrap Sharpe 95% CI: [{lo:.2f}, {hi:.2f}]")
    os.makedirs("results_analysis", exist_ok=True); W.to_csv("results_analysis/weights_sanity.csv")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- continuous (all-names) weights
def build_weights_continuous(sig, vol, max_w=0.12, warmup=260, smooth=1.0, rebal=1):
    """Weights proportional to signal z-score / vol across ALL names, dollar-neutral, gross=1.
    smooth in (0,1]: fraction of the gap to target closed on each rebalance (turnover control)."""
    inv = 1 / vol
    tgt = (sig * inv)
    W = pd.DataFrame(0.0, index=sig.index, columns=sig.columns)
    prev = pd.Series(0.0, index=sig.columns)
    for i in range(warmup, len(sig), rebal):
        x = tgt.iloc[i]
        if x.notna().sum() < 10:
            W.iloc[i] = prev
            continue
        x = x.fillna(0.0)
        new = pd.Series(0.0, index=sig.columns)
        for side in (1, -1):
            part = (x.clip(lower=0) if side == 1 else (-x).clip(lower=0))
            if part.sum() <= 0:
                continue
            p = part / part.sum() * 0.5
            for _ in range(10):
                p = p.clip(upper=max_w)
                p = p / p.sum() * 0.5
            new += side * p
        prev = (1 - smooth) * prev + smooth * new
        W.iloc[i] = prev
    return _hold(W, sig.index, warmup, rebal)


def _hold(W, idx, warmup, rebal):
    mask = pd.Series(False, index=idx)
    mask.iloc[warmup::rebal] = True
    return W.where(mask).ffill().fillna(0.0)
