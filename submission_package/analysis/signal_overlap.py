"""How similar are momentum (MOM20) and trend strength (TS20)? Does using both change the strategy? Run from the repository root."""
import os, sys
if os.environ.get('PYTHONHASHSEED') != '0':   # fixed hash seed: the backtester fills orders in set-iteration order, so this makes runs bit-identical
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)
sys.path.insert(0, os.getcwd())          # repository root: backtester.py, strategy_base.py, src/
os.makedirs('results_analysis', exist_ok=True)
import io, json, contextlib
import numpy as np, pandas as pd
import research_lib as t
import backtester as B
from backtester import Backtester, BacktestConfig
import src.engine_ladder as L
from src.engine_ladder import _z

OUT = {}
# ---- 1. similarity of the two scores across the 24 stocks
w = t.load_wide('data/Dataset_PS-A.csv'); c = w['close']; r = c.pct_change()
mom = c / c.shift(20) - 1; ts = mom / r.rolling(20).std()
m = mom.notna() & ts.notna(); a, b = mom.where(m), ts.where(m)
sp = a.rank(axis=1).corrwith(b.rank(axis=1), axis=1).dropna(); pe = a.corrwith(b, axis=1).dropna()
fwd = c.shift(-20) / c - 1
ic_m = t.rank_ic(-mom, fwd).dropna(); ic_t = t.rank_ic(-ts, fwd).dropna()
top_m = (-mom).rank(axis=1, ascending=False) <= 5; top_t = (-ts).rank(axis=1, ascending=False) <= 5
OUT['similarity'] = dict(spearman_mean=float(sp.mean()), spearman_min=float(sp.min()), pearson_mean=float(pe.mean()), share_days_spearman_gt_0_8=float((sp > 0.8).mean()),
                         ic_series_corr=float(ic_m.corr(ic_t)), top5_overlap=float((top_m & top_t).sum(axis=1)[m.all(axis=1)].mean()))
print('similarity', {k: round(v, 3) for k, v in OUT['similarity'].items()})

# ---- 2. the same strategy with each version of the reversal score, in the official backtester
CAP = {}
def _patched(self, result, equity_curve, leverage_curve, timestamps=None, ledger=None): CAP['eq'] = np.array(equity_curve, float)
B.Backtester._save_output = _patched
def make(kind, lowvol):
    class V(L.LadderStrategy):
        SIGNAL = 'x'
        def _signal(self, vol, C):
            mo = C[-1] / C[0] - 1.0; rets = C[1:] / C[:-1] - 1.0; tr = mo / np.maximum(rets.std(axis=0, ddof=1), 1e-8)
            raw = {'mom': -_z(mo), 'ts': -_z(tr), 'both': -(_z(mo) + _z(tr)) / 2}[kind]
            if lowvol: raw = _z(raw) + _z(-vol)
            return _z(raw)
    return V
def run(cls):
    cfg = BacktestConfig(track='A', data_path='data/Dataset_PS-A.csv', initial_cash=100000.0, slippage_rate=0.0, commission_rate=0.0001, symbol='ALL', max_ticks=None, save_results=True, output_dir='results_analysis/_tmp', generate_plot=False)
    with contextlib.redirect_stdout(io.StringIO()):
        res = Backtester(cfg).run(cls())
    ms = res.metrics_summary; e = CAP['eq']
    return dict(cagr=ms['CAGR (%)'], sharpe=ms['Sharpe Ratio'], ci=ms['Sharpe 95% CI'], dd=ms['Max Drawdown (%)'], calmar=ms['Calmar Ratio'], turn=ms['Total Turnover (%)']), e[1:] / e[:-1] - 1
OUT['variants'] = {}; RET = {}
for lv in (False, True):
    for kind in ('mom', 'ts', 'both'):
        key = f"{kind}{'+lowvol' if lv else ''}"; OUT['variants'][key], RET[key] = run(make(kind, lv))
        v = OUT['variants'][key]; print(f"{key:14s} CAGR {v['cagr']:.2f}  Sharpe {v['sharpe']:.2f} [{v['ci'][0]:.2f}, {v['ci'][1]:.2f}]  MaxDD {v['dd']:.2f}  Calmar {v['calmar']:.2f}")

# ---- 3. paired block bootstrap: trend strength + low-vol minus (momentum + trend strength) + low-vol
def met(x):
    eq = np.cumprod(1 + x); y = len(x) / 252; cg = eq[-1] ** (1 / y) - 1; dd = (eq / np.maximum.accumulate(eq) - 1).min()
    return np.array([cg * 100, x.mean() / x.std() * np.sqrt(252), dd * 100])
A_, B_ = RET['ts+lowvol'], RET['both+lowvol']; T = len(A_); rng = np.random.default_rng(42); d = []
for _ in range(5000):
    st = rng.integers(0, T - 20, int(np.ceil(T / 20))); idx = np.concatenate([np.arange(j, j + 20) for j in st])[:T]
    d.append(met(A_[idx]) - met(B_[idx]))
d = np.array(d); lo, hi = np.percentile(d, [2.5, 97.5], axis=0); pt = met(A_) - met(B_)
OUT['paired_ts_vs_both_lowvol'] = {n: dict(point=float(pt[i]), lo=float(lo[i]), hi=float(hi[i]), p_le0=float(np.mean(d[:, i] <= 0))) for i, n in enumerate(['cagr', 'sharpe', 'maxdd'])}
OUT['paired_ts_vs_both_lowvol']['daily_return_corr'] = float(np.corrcoef(A_, B_)[0, 1])
print('paired (TS+lowvol minus both+lowvol):', {k: (round(v['point'], 3), round(v['lo'], 3), round(v['hi'], 3)) if isinstance(v, dict) else round(v, 4) for k, v in OUT['paired_ts_vs_both_lowvol'].items()})
json.dump(OUT, open('results_analysis/signal_overlap.json', 'w'), indent=1)
