import os, sys
if os.environ.get('PYTHONHASHSEED') != '0':   # fixed hash seed: the backtester fills orders in set-iteration order, so this makes runs bit-identical
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)
sys.path.insert(0, os.getcwd())          # repository root: backtester.py, strategy_base.py, src/
os.makedirs('results_analysis', exist_ok=True)
"""Expanding-window walk-forward parameter selection using the OFFICIAL backtester.
Every config is backtested once over the full sample (strategy is causal, so slicing its daily returns by date
is equivalent to running live). For each test year, the config with the best Sharpe on all PRIOR data is chosen
and only that config's returns inside the test year are kept. Stitched returns = honest out-of-sample record."""
import io, contextlib, itertools, numpy as np, pandas as pd
import backtester as B
from backtester import Backtester, BacktestConfig
import src.engine as E
from bench_engine import BenchEW

CAP = {}
def patched(self, result, equity_curve, leverage_curve, timestamps=None, ledger=None):
    CAP["eq"] = np.array(equity_curve, float); CAP["ts"] = timestamps
B.Backtester._save_output = patched
DATA = "data/Dataset_PS-A.csv"

def run(cls):
    cfg = BacktestConfig(track="A", data_path=DATA, initial_cash=100000.0, slippage_rate=float(__import__("os").environ.get("SLIP_BPS","0"))/1e4, commission_rate=float(__import__("os").environ.get("COMM_BPS","1"))/1e4,
                         symbol="ALL", max_ticks=None, save_results=True, output_dir="x", generate_plot=False)
    with contextlib.redirect_stdout(io.StringIO()):
        Backtester(cfg).run(cls())
    eq, ts = CAP["eq"], CAP["ts"]
    dates = pd.to_datetime(pd.Series(list(ts))) if ts is not None and len(ts) == len(eq) else None
    return pd.Series(eq, index=dates)

def rets(eq): return eq.pct_change().dropna()
def sharpe(r): return r.mean() / r.std() * np.sqrt(252) if len(r) > 5 and r.std() > 0 else -9
def stats(r):
    eq = (1 + r).cumprod(); y = len(r) / 252
    return (eq.iloc[-1] ** (1 / y) - 1) * 100, sharpe(r), (eq / eq.cummax() - 1).min() * 100

if __name__ == "__main__":
    ew = rets(run(BenchEW))
    print("equity/ts alignment ok, n =", len(ew), ew.index.min().date(), ew.index.max().date())
    grid = list(itertools.product([True, False], [1.0, 2.0, 3.0], [5, 10], [0.3, 0.5], [0.12, 0.15, 0.18]))
    R = {}
    for lv, tilt, rb, sm, vt in grid:
        cls = type("S", (E.ParticipantStrategy,), dict(USE_LOWVOL=lv, TILT=tilt, REBAL=rb, SMOOTH=sm, VOL_TARGET=vt))
        R[(lv, tilt, rb, sm, vt)] = rets(run(cls))
    R = pd.DataFrame(R)
    default = R[(True, 2.0, 5, 0.3, 0.15)]
    oos, picks = [], []
    for yr in (2019, 2020, 2021, 2022):
        train = R[R.index < f"{yr}-01-01"]
        best = train.apply(sharpe).idxmax()
        test = R.loc[R.index.year == yr, best]; oos.append(test); picks.append((yr, best, sharpe(train[best])))
    oos = pd.concat(oos); idx = oos.index
    print("\nChosen config per test year (lowvol, tilt, rebal, smooth, vol_target) and its TRAIN Sharpe:")
    for yr, b, s in picks: print(f"  {yr}: {b}  train Sharpe {s:.2f}")
    print(f"\nOUT-OF-SAMPLE {idx.min().date()} -> {idx.max().date()}  (CAGR %, Sharpe, MaxDD %)")
    for name, r in (("walk-forward selected", oos), ("fixed default (lowvol,2,5,0.3,0.15)", default.loc[idx]),
                    ("equal-weight benchmark", ew.loc[idx.intersection(ew.index)])):
        c, s, d = stats(r); print(f"  {name:38s} {c:6.1f} {s:6.2f} {d:7.1f}")
    # bootstrap CIs
    rng = np.random.default_rng(42); x = oos.values; e = ew.reindex(idx).fillna(0).values; T = len(x); out = []; diff = []
    for _ in range(3000):
        st = rng.integers(0, T - 20, int(np.ceil(T / 20))); ii = np.concatenate([np.arange(j, j + 20) for j in st])[:T]
        a, b = x[ii], e[ii]; sa, sb = a.mean() / a.std() * np.sqrt(252), b.mean() / b.std() * np.sqrt(252)
        out.append(sa); diff.append(sa - sb)
    print("  OOS Sharpe 95% CI:", np.percentile(out, [2.5, 97.5]).round(2), " | vs EW Sharpe diff 95% CI:", np.percentile(diff, [2.5, 97.5]).round(2),
          f" P(diff<=0)={np.mean(np.array(diff) <= 0):.2f}")
    pd.DataFrame({'selected': oos, 'default': default.loc[idx], 'ew': ew.reindex(idx)}).to_csv('results_analysis/wf_oos.csv')
    print("\nSharpe by year (selected / default / EW):")
    for yr in (2019, 2020, 2021, 2022):
        m = idx.year == yr; print(f"  {yr}: {sharpe(oos[m]):5.2f} / {sharpe(default.loc[idx][m]):5.2f} / {sharpe(ew.reindex(idx)[m].fillna(0)):5.2f}")
    print(f"\nGrid median full-sample Sharpe {R.apply(sharpe).median():.2f}; share of grid configs with Sharpe>EW: {(R.apply(sharpe) > sharpe(ew)).mean():.0%}")
    R.apply(sharpe).to_frame("sharpe_full").to_csv("results_analysis/wf_grid_sharpe.csv")
