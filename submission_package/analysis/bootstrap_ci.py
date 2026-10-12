import os, sys
if os.environ.get('PYTHONHASHSEED') != '0':   # fixed hash seed: the backtester fills orders in set-iteration order, so this makes runs bit-identical
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)
sys.path.insert(0, os.getcwd())          # repository root: backtester.py, strategy_base.py, src/
os.makedirs('results_analysis', exist_ok=True)
import json, numpy as np, pandas as pd
rng_seed = 42
def metrics(r):
    eq = np.cumprod(1 + r); yrs = len(r) / 252
    cagr = eq[-1] ** (1 / yrs) - 1; dd = (eq / np.maximum.accumulate(eq) - 1).min()
    sh = r.mean() / r.std() * np.sqrt(252)
    return np.array([cagr * 100, sh, dd * 100, (cagr / abs(dd)) if dd < 0 else np.nan])
def boot(series_dict, block=20, n=3000):
    keys = list(series_dict); X = np.column_stack([series_dict[k] for k in keys]); T = len(X)
    rng = np.random.default_rng(rng_seed); res = {k: [] for k in keys}; diffs = []
    for _ in range(n):
        st = rng.integers(0, T - block, int(np.ceil(T / block))); idx = np.concatenate([np.arange(j, j + block) for j in st])[:T]
        ms = [metrics(X[idx, i]) for i in range(len(keys))]
        for k, m in zip(keys, ms): res[k].append(m)
        if len(keys) == 2: diffs.append(ms[0] - ms[1])
    out = {}
    for k in keys:
        a = np.array(res[k]); pt = metrics(series_dict[k]); lo, hi = np.nanpercentile(a, [2.5, 97.5], axis=0)
        out[k] = dict(point=pt.tolist(), lo=lo.tolist(), hi=hi.tolist())
    if diffs:
        d = np.array(diffs); lo, hi = np.nanpercentile(d, [2.5, 97.5], axis=0); out['diff'] = dict(lo=lo.tolist(), hi=hi.tolist(), p_le0=np.nanmean(d <= 0, axis=0).tolist())
    return out
cv = pd.read_csv('results_analysis/curves.csv', index_col=0, parse_dates=True)
fin = cv['MOM20+TS20 + low-vol (FINAL)'].pct_change().dropna(); ew = cv['Equal-weight (benchmark)'].pct_change().dropna().reindex(fin.index).fillna(0)
R = boot({'final': fin.values, 'ew': ew.values})
json.dump(R, open('results_analysis/ci_full.json', 'w'), indent=1)
names = ['CAGR %', 'Sharpe', 'MaxDD %', 'Calmar']
for k in ('final', 'ew'):
    print(k, {n: f"{R[k]['point'][i]:.2f} [{R[k]['lo'][i]:.2f}, {R[k]['hi'][i]:.2f}]" for i, n in enumerate(names)})
print('diff final-ew', {n: f"[{R['diff']['lo'][i]:+.2f}, {R['diff']['hi'][i]:+.2f}] P<=0 {R['diff']['p_le0'][i]:.2f}" for i, n in enumerate(names)})

# ---- out-of-sample (walk-forward) intervals; run analysis/walk_forward.py first
d = pd.read_csv('results_analysis/wf_oos.csv', index_col=0, parse_dates=True)
Ro = boot({'sel': d['selected'].values, 'ew': d['ew'].values})
json.dump(Ro, open('results_analysis/ci_oos.json', 'w'), indent=1)
for k in ('sel', 'ew'):
    print('OOS', k, {n: f"{Ro[k]['point'][i]:.2f} [{Ro[k]['lo'][i]:.2f}, {Ro[k]['hi'][i]:.2f}]" for i, n in enumerate(names)})
print('OOS diff', {n: f"[{Ro['diff']['lo'][i]:+.2f}, {Ro['diff']['hi'][i]:+.2f}] P<=0 {Ro['diff']['p_le0'][i]:.2f}" for i, n in enumerate(names)})

# ---- paired comparison of the final strategy with the simpler MOM20 + TS20 variant (no low-volatility term)
cvs = pd.read_csv('results_analysis/curves.csv', index_col=0, parse_dates=True)
f_ = cvs['MOM20+TS20 + low-vol (FINAL)'].pct_change().dropna().values; m_ = cvs['MOM20+TS20'].pct_change().dropna().values
Rm = boot({'final': f_, 'momts': m_})
json.dump(Rm, open('results_analysis/ci_momts.json', 'w'), indent=1)
print('final minus MOM20+TS20:', {n: f"{Rm['final']['point'][i] - Rm['momts']['point'][i]:+.2f} [{Rm['diff']['lo'][i]:+.2f}, {Rm['diff']['hi'][i]:+.2f}] P<=0 {Rm['diff']['p_le0'][i]:.2f}" for i, n in enumerate(names)})
