import os, sys
if os.environ.get('PYTHONHASHSEED') != '0':   # fixed hash seed: the backtester fills orders in set-iteration order, so this makes runs bit-identical
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)
sys.path.insert(0, os.getcwd())          # repository root: backtester.py, strategy_base.py, src/
os.makedirs('results_analysis', exist_ok=True)
import sys, json, numpy as np, pandas as pd
import research_lib as t
from scipy import stats as st
OUT = {}
w = t.load_wide('data/Dataset_PS-A.csv'); c = w['close']; ret = np.log(c).diff(); sret = c.pct_change()
n_t = c.shape[1]
OUT['rows'] = int(sum(w['close'].count())); OUT['tickers'] = n_t; OUT['days'] = int(c.shape[0])
OUT['start'] = str(c.index.min().date()); OUT['end'] = str(c.index.max().date())
OUT['missing_close'] = int(c.isna().sum().sum())
pooled = ret.stack(); pooled = pooled[np.isfinite(pooled)]
OUT['pooled_mean_pct'] = pooled.mean()*100; OUT['pooled_std_pct'] = pooled.std()*100
OUT['skew'] = float(st.skew(pooled)); OUT['xkurt'] = float(st.kurtosis(pooled))
av = ret.std()*np.sqrt(252); OUT['ann_vol_min'] = av.min(); OUT['ann_vol_max'] = av.max(); OUT['ann_vol_mean'] = av.mean()
C = ret.corr().values; off = C[~np.eye(n_t, dtype=bool)]
OUT['corr_min'] = off.min(); OUT['corr_max'] = off.max(); OUT['corr_mean'] = off.mean()
OUT['ac1_ret'] = float(ret.apply(lambda s: s.autocorr(1)).mean()); OUT['ac1_abs'] = float(ret.abs().apply(lambda s: s.autocorr(1)).mean())
# Ljung-Box on squared returns lag 10
def lb(x, L=10):
    x = x.dropna().values; x = x - x.mean(); n = len(x); acf = np.array([ (x[k:]@x[:n-k])/(x@x) for k in range(1, L+1)])
    Q = n*(n+2)*np.sum(acf**2/(n-np.arange(1, L+1))); return Q, 1-st.chi2.cdf(Q, L)
res = [lb(ret[k]**2) for k in ret.columns]; OUT['lb_sq_sig_count'] = int(sum(p < 0.01 for _, p in res)); OUT['lb_sq_Q_median'] = float(np.median([q for q,_ in res]))
res_r = [lb(ret[k]) for k in ret.columns]; OUT['lb_ret_sig_count'] = int(sum(p < 0.01 for _, p in res_r))
jb = st.jarque_bera(pooled); OUT['jb_p'] = float(jb.pvalue)
OUT['weekend_dates'] = [str(d.date()) for d in c.index if d.weekday() >= 5]
big = sret.stack(); OUT['n_moves_gt15'] = int((big.abs() > 0.15).sum())
mx = ret.abs().max().sort_values(ascending=False); OUT['max_abs_logret'] = float(mx.iloc[0])
# --- 20-day horizon signal tests (daily Spearman IC, Newey-West 20 lags)
H = 20
fwd = c.shift(-H)/c - 1
mom = c/c.shift(20) - 1
sig20 = sret.rolling(20).std()
ts = mom/sig20
d = c.diff(); g = d.clip(lower=0).rolling(14).mean(); l = (-d.clip(upper=0)).rolling(14).mean()
rsi = 100 - 100/(1 + g/l.replace(0, np.nan))
ewv = t.vol_ewma(w, 10)
hl = (np.log(w['high']/w['low'])); atr = (w['high']-w['low']).rolling(20).mean()/c
z = t.xs_zscore
cands = {'MOM20': -mom, 'TS20': -ts, 'RSI14': -rsi, 'MA-dist(20)': -(c/c.rolling(20).mean()-1), 'Breakout(20)': -(c/c.rolling(20).max()-1),
         'Rel. volume': -(w['volume']/w['volume'].rolling(20).mean()), 'Low vol (EWMA)': -ewv, 'ATR/Close': -atr}
cands['MOM20+TS20'] = z(z(-mom)+z(-ts))/1; cands['MOM20+TS20+RSI14 (RAVEN)'] = z(z(-mom)+z(-ts)+z(-rsi))
cands['MOM20+TS20 + low-vol'] = z(z(z(-mom)+z(-ts)) + z(-ewv))
tab = {}
for k, s in cands.items():
    ic = t.rank_ic(s, fwd).dropna()
    q = t.quantile_spread(s, fwd, 5)
    tab[k] = dict(mean_ic=float(ic.mean()), pos=float((ic>0).mean()), hac_t=float(t.nw_tstat(ic, 20)), q1=float(q.iloc[0]*100), q5=float(q.iloc[-1]*100), spread=float((q.iloc[-1]-q.iloc[0])*100), n=int(len(ic)))
    se = ic.mean()/tab[k]['hac_t'] if tab[k]['hac_t'] else np.nan
    tab[k]['hac_p'] = float(2*(1-st.norm.cdf(abs(tab[k]['hac_t']))))
OUT['ic_table'] = tab
# Fama-MacBeth for composite
for nm in ['MOM20+TS20', 'MOM20+TS20 + low-vol']:
    fm = t.fama_macbeth(cands[nm], fwd).dropna(); OUT['fm_'+nm] = dict(mean=float(fm.mean()), hac_t=float(t.nw_tstat(fm, 20)))
# IC by year and by vol regime for MOM20+TS20
s = cands['MOM20+TS20']; ic = t.rank_ic(s, fwd).dropna()
OUT['ic_by_year'] = {int(y): float(v) for y, v in ic.groupby(ic.index.year).mean().items()}
OUT['ic_by_year_hac'] = {int(y): float(t.nw_tstat(v, 20)) for y, v in ic.groupby(ic.index.year)}
medvol = ewv.median(axis=1).reindex(ic.index); terc = pd.qcut(medvol, 3, labels=['Low', 'Mid', 'High'])
OUT['ic_by_vol_regime'] = {k: float(v) for k, v in ic.groupby(terc).mean().items()}
OUT['ic_by_vol_regime_pos'] = {k: float(v) for k, v in ic.groupby(terc).apply(lambda x: (x>0).mean()).items()}
# lag-1 daily rev signal (5d) horizon 1 for reference
f1 = t.forward_returns(c, 1, 1); r5 = z(-np.log(c/c.shift(5))/(ewv*np.sqrt(5)))
ic5 = t.rank_ic(r5, f1).dropna(); OUT['rev5_h1'] = dict(ic=float(ic5.mean()), t=float(t.nw_tstat(ic5)))
# --- vol estimator comparison (QLIKE vs next-5d realized var)
vols = {'Rolling 20d': t.vol_rolling(w, 20), 'Rolling 60d': t.vol_rolling(w, 60), 'EWMA (hl 10)': t.vol_ewma(w, 10), 'EWMA (hl 30)': t.vol_ewma(w, 30), 'Garman-Klass 20d': t.vol_garman_klass(w, 20)}
OUT['qlike'] = {k: t.qlike(v, ret, 5) for k, v in vols.items()}
OUT['qlike20'] = {k: t.qlike(v, ret, 20) for k, v in vols.items()}
# --- concentration of top-5 by composite: avg pairwise corr (rolling 60d) vs universe
sc = cands['MOM20+TS20 + low-vol']; top5 = sc.rank(axis=1, ascending=False) <= 5
cm = []; um = []
rr = sret
for i in range(120, len(sc), 20):
    win = rr.iloc[i-60:i]; Cm = win.corr().values; names = np.where(top5.iloc[i].values)[0]
    if len(names) >= 5:
        sub = Cm[np.ix_(names, names)]; cm.append((sub.sum()-len(names))/(len(names)**2-len(names)))
        um.append((Cm.sum()-n_t)/(n_t**2-n_t))
OUT['top5_pair_corr'] = float(np.mean(cm)); OUT['univ_pair_corr_60d'] = float(np.mean(um))
freq = top5.mean(); OUT['top5_hhi'] = float(((freq/freq.sum())**2).sum()); OUT['top5_hhi_uniform'] = 1/n_t; OUT['top5_freq_min'] = float(freq.min()); OUT['top5_freq_max'] = float(freq.max())
json.dump(OUT, open('results_analysis/analysis1.json', 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in OUT.items() if k not in ('ic_table',)}, indent=1, default=float)[:3500])
print()
for k, v in tab.items(): print(f"{k:28s} IC={v['mean_ic']:+.4f} pos={v['pos']:.2%} HACt={v['hac_t']:+.2f} p={v['hac_p']:.4f} Q1={v['q1']:.2f} Q5={v['q5']:.2f} spr={v['spread']:+.2f}")
