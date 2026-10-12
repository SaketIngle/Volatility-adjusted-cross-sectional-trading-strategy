import os, sys
if os.environ.get('PYTHONHASHSEED') != '0':   # fixed hash seed: the backtester fills orders in set-iteration order, so this makes runs bit-identical
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)
sys.path.insert(0, os.getcwd())          # repository root: backtester.py, strategy_base.py, src/
os.makedirs('results_analysis', exist_ok=True)
import sys, json, numpy as np, pandas as pd
import research_lib as t
from scipy import stats as st
w = t.load_wide('data/Dataset_PS-A.csv'); c = w['close']; ret = np.log(c).diff()
OUT = {}
# ---- (1) Diebold-Mariano on QLIKE loss differentials, 5-day horizon, NW with 5 lags
H = 5
real = (ret ** 2).rolling(H).mean().shift(-H)
def qlike_series(fv):
    f = fv ** 2; m = f.notna() & real.notna() & (f > 0) & (real > 0); x = (real / f).where(m)
    return (x - np.log(x) - 1).mean(axis=1).dropna()
vols = {'EWMA hl10': t.vol_ewma(w, 10), 'Rolling 20d': t.vol_rolling(w, 20), 'Rolling 60d': t.vol_rolling(w, 60), 'EWMA hl30': t.vol_ewma(w, 30), 'Garman-Klass 20d': t.vol_garman_klass(w, 20)}
L = {k: qlike_series(v) for k, v in vols.items()}
dm = {}
for k in ('Rolling 20d', 'Rolling 60d', 'Garman-Klass 20d', 'EWMA hl30'):
    d = (L[k] - L['EWMA hl10']).dropna()          # positive => EWMA hl10 has lower loss
    tt = t.nw_tstat(d, lags=H); dm[k] = dict(mean_diff=float(d.mean()), t=float(tt), p=float(2 * (1 - st.norm.cdf(abs(tt)))))
OUT['dm_vs_ewma10'] = dm
# ---- (2) Ljung-Box on squared returns vs squared vol-standardised returns (lag 10)
def lb(x, Lg=10):
    x = np.asarray(x.dropna()); x = x - x.mean(); n = len(x); acf = np.array([(x[k:] @ x[:n - k]) / (x @ x) for k in range(1, Lg + 1)])
    Q = n * (n + 2) * np.sum(acf ** 2 / (n - np.arange(1, Lg + 1))); return Q, 1 - st.chi2.cdf(Q, Lg)
sig = vols['EWMA hl10'].shift(1)                # forecast made the day before
zst = ret / sig
rows = []
for k in ret.columns:
    q_raw, p_raw = lb(ret[k] ** 2); q_z, p_z = lb(zst[k] ** 2)
    rows.append((q_raw, p_raw, q_z, p_z))
r = np.array(rows)
OUT['lb'] = dict(sig_raw=int((r[:, 1] < 0.01).sum()), sig_std=int((r[:, 3] < 0.01).sum()), Q_raw_median=float(np.median(r[:, 0])), Q_std_median=float(np.median(r[:, 2])))
zz = zst.stack(); zz = zz[np.isfinite(zz)]; rr = ret.stack(); rr = rr[np.isfinite(rr)]
OUT['kurt'] = dict(raw=float(st.kurtosis(rr)), std=float(st.kurtosis(zz)), std_sd=float(zz.std()))
# ---- (3) ADF if available
try:
    from statsmodels.tsa.stattools import adfuller
    pl = [adfuller(np.log(c[k].dropna()), autolag='AIC')[1] for k in c.columns]; pr = [adfuller(ret[k].dropna(), autolag='AIC')[1] for k in c.columns]
    OUT['adf'] = dict(levels_reject_5pct=int(sum(p < 0.05 for p in pl)), returns_reject_5pct=int(sum(p < 0.05 for p in pr)), n=len(pl))
except Exception as e:
    OUT['adf'] = str(e)
print(json.dumps(OUT, indent=1))
json.dump(OUT, open('results_analysis/audit_stats.json', 'w'), indent=1)
