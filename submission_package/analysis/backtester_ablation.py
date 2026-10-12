import os, sys
if os.environ.get('PYTHONHASHSEED') != '0':   # fixed hash seed: the backtester fills orders in set-iteration order, so this makes runs bit-identical
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)
sys.path.insert(0, os.getcwd())          # repository root: backtester.py, strategy_base.py, src/
os.makedirs('results_analysis', exist_ok=True)
import io, contextlib, json, sys, numpy as np, pandas as pd
import backtester as B
from backtester import Backtester, BacktestConfig
import src.engine_ladder as L
from bench_engine import BenchEW
CAP = {}
def patched(self, result, equity_curve, leverage_curve, timestamps=None, ledger=None):
    CAP['eq'] = np.array(equity_curve, float); CAP['lev'] = np.array(leverage_curve, float); CAP['ts'] = timestamps
B.Backtester._save_output = patched
def run(cls, path='data/Dataset_PS-A.csv'):
    cfg = BacktestConfig(track='A', data_path=path, initial_cash=100000.0, slippage_rate=0.0, commission_rate=0.0001,
                         symbol='ALL', max_ticks=None, save_results=True, output_dir='x', generate_plot=False)
    with contextlib.redirect_stdout(io.StringIO()):
        r = Backtester(cfg).run(cls())
    m = r.metrics_summary
    dates = pd.to_datetime(pd.Series(list(CAP['ts'])))
    eq = pd.Series(CAP['eq'], index=dates)
    return m, eq, pd.Series(CAP['lev'], index=dates)
OUT = {}
def S(m): return dict(cagr=m['CAGR (%)'], sharpe=m['Sharpe Ratio'], ci=m['Sharpe 95% CI'], sortino=m['Sortino Ratio'], dd=m['Max Drawdown (%)'], calmar=m['Calmar Ratio'],
                      turn=m.get('Total Turnover (%)'), lev=m.get('Mean Gross Leverage'), comm=m.get('Total Commission Paid ($)'))
curves = {}
# ---- ladder
rows = [('Equal-weight (benchmark)', BenchEW, {}), ('Inverse-vol, no vol target', L.LadderStrategy, dict(SIGNAL='none', VOL_TARGET=1e9)),
        ('Inverse-vol + vol target', L.LadderStrategy, dict(SIGNAL='none')), ('5d reversal + low-vol (earlier)', L.LadderStrategy, dict(SIGNAL='rev5lv')),
        ('MOM20 reversal', L.LadderStrategy, dict(SIGNAL='mom20')), ('MOM20+TS20', L.LadderStrategy, dict(SIGNAL='mom_ts')),
        ('RAVEN (MOM+TS+RSI)', L.LadderStrategy, dict(SIGNAL='raven')), ('MOM20+TS20 + low-vol (FINAL)', L.LadderStrategy, dict(SIGNAL='mom_ts_lv'))]
OUT['ladder'] = {}
for name, base, ov in rows:
    cls = base if not ov else type('S', (base,), ov)
    m, eq, lev = run(cls); OUT['ladder'][name] = S(m); curves[name] = eq
    if name.startswith('MOM20+TS20 + low'): final_lev = lev
# no vol target versions of signals (for the 'alpha vs risk' split)
OUT['novt'] = {}
for name, sg in (('MOM20+TS20', 'mom_ts'), ('MOM20+TS20 + low-vol', 'mom_ts_lv'), ('RAVEN', 'raven')):
    m, eq, lev = run(type('S', (L.LadderStrategy,), dict(SIGNAL=sg, VOL_TARGET=1e9))); OUT['novt'][name] = S(m)
# ---- sub-periods
OUT['sub'] = {}
for nm, sg in (('EW', None), ('FINAL', 'mom_ts_lv'), ('InvVol+VT', 'none')):
    for pth in ('data/pre2020.csv', 'data/post2020.csv'):
        cls = BenchEW if sg is None else type('S', (L.LadderStrategy,), dict(SIGNAL=sg))
        m, eq, lev = run(cls, pth); OUT['sub'][f'{nm}|{pth.split("/")[1][:-4]}'] = S(m)
# ---- sensitivity one-at-a-time around FINAL
base_m = OUT['ladder']['MOM20+TS20 + low-vol (FINAL)']
grid = {'VOL_TARGET': [1e9, 0.10, 0.12, 0.18, 0.22], 'TILT': [0.0, 0.5, 1.0, 3.0], 'REBAL': [1, 3, 10, 20], 'SMOOTH': [0.1, 0.5, 1.0],
        'VOL_HL': [5, 20, 30, 40], 'CAP': [0.10, 0.20], 'BOOK_HL': [5, 20, 40]}
OUT['sens'] = []
for p, vals in grid.items():
    for v in vals:
        m, _, _ = run(type('S', (L.LadderStrategy,), dict(SIGNAL='mom_ts_lv', **{p: v})))
        OUT['sens'].append(dict(param=p, value=('none' if v == 1e9 else v), cagr=m['CAGR (%)'], sharpe=m['Sharpe Ratio'], dd=m['Max Drawdown (%)'], calmar=m['Calmar Ratio']))
# ---- record weights of FINAL via wrapper
LOG = []
class Rec(L.LadderStrategy):
    SIGNAL = 'mom_ts_lv'
    def on_bar(self, context, bars):
        if not getattr(self, '_wrapped', False):
            orig = context.set_target_weights
            def rec(w):
                LOG.append(dict(w)); orig(w)
            context.set_target_weights = rec
            self._wrapped = True
        super().on_bar(context, bars)
m, eq, lev = run(Rec)
OUT['rec_check'] = dict(sharpe=m['Sharpe Ratio'], n_logs=len(LOG))
if LOG:
    W = pd.DataFrame(LOG).fillna(0.0); gross = W.sum(axis=1); nw = W.div(gross.replace(0, np.nan), axis=0)
    effN = 1 / (nw ** 2).sum(axis=1)
    OUT['weights'] = dict(mean_gross=float(gross.mean()), min_gross=float(gross.min()), mean_effN=float(effN.mean()), min_effN=float(effN.min()),
                          max_w_mean=float(nw.max(axis=1).mean()), max_w_max=float(nw.max().max()), mean_w_by_ticker_max=float(nw.mean().max()), mean_w_by_ticker_min=float(nw.mean().min()),
                          avg_cash_pct=float((1 - gross).mean() * 100))
# ---- yearly returns
def yearly(eq): r = eq.pct_change().dropna(); return {int(y): float(((1 + x).prod() - 1) * 100) for y, x in r.groupby(r.index.year)}
OUT['yearly'] = {k: yearly(curves[k]) for k in ('Equal-weight (benchmark)', 'MOM20+TS20 + low-vol (FINAL)', 'Inverse-vol + vol target')}
# ---- deflated Sharpe (Bailey & Lopez de Prado) on FINAL daily returns
from scipy import stats as st
r = curves['MOM20+TS20 + low-vol (FINAL)'].pct_change().dropna(); T = len(r)
sr_d = r.mean() / r.std(); g3 = st.skew(r); g4 = st.kurtosis(r, fisher=False)
grid_sh = pd.read_csv('results_analysis/wf_grid_sharpe.csv', index_col=0)['sharpe_full'].values / np.sqrt(252)
V = grid_sh.var(ddof=1); gam = 0.5772156649
res = {}
for N in (72, 320):
    e_max = np.sqrt(V) * ((1 - gam) * st.norm.ppf(1 - 1 / N) + gam * st.norm.ppf(1 - 1 / (N * np.e)))
    z = (sr_d - e_max) * np.sqrt(T - 1) / np.sqrt(1 - g3 * sr_d + (g4 - 1) / 4 * sr_d ** 2)
    res[N] = dict(sr0_ann=float(e_max * np.sqrt(252)), dsr=float(st.norm.cdf(z)))
OUT['dsr'] = dict(sr_ann=float(sr_d * np.sqrt(252)), var_trial_sr_daily=float(V), T=T, skew=float(g3), kurt=float(g4), by_N=res)
# paired bootstrap vs EW
ew = curves['Equal-weight (benchmark)'].pct_change().dropna().reindex(r.index).fillna(0).values; x = r.values
rng = np.random.default_rng(42); d = []
for _ in range(3000):
    st_ = rng.integers(0, T - 20, int(np.ceil(T / 20))); ii = np.concatenate([np.arange(j, j + 20) for j in st_])[:T]
    a, b = x[ii], ew[ii]; d.append(a.mean() / a.std() * np.sqrt(252) - b.mean() / b.std() * np.sqrt(252))
OUT['paired_vs_ew_full'] = dict(lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)), p_le0=float(np.mean(np.array(d) <= 0)))
# beta / alpha vs EW (daily OLS)
b_ = np.polyfit(ew, x, 1); OUT['ols_vs_ew'] = dict(beta=float(b_[0]), alpha_ann_pct=float(b_[1] * 252 * 100), corr=float(np.corrcoef(ew, x)[0, 1]))
# save curves for plotting
pd.DataFrame({k: v for k, v in curves.items()}).to_csv('results_analysis/curves.csv')
final_lev.to_csv('results_analysis/final_lev.csv')
json.dump(OUT, open('results_analysis/analysis2.json', 'w'), indent=1, default=float)
for k, v in OUT['ladder'].items(): print(f"{k:34s} CAGR={v['cagr']:5.1f} Sh={v['sharpe']:.2f} ci={v['ci']} DD={v['dd']:.1f} Cal={v['calmar']:.2f} turn={v['turn']}")
print('NOVT', {k: (round(v['cagr'],1), round(v['sharpe'],2), round(v['dd'],1)) for k, v in OUT['novt'].items()})
print('SUB', {k: (round(v['cagr'],1), round(v['sharpe'],2), round(v['dd'],1)) for k, v in OUT['sub'].items()})
print('WEIGHTS', OUT.get('weights'), OUT['rec_check'])
print('YEARLY', OUT['yearly'])
print('DSR', OUT['dsr']); print('PAIRED', OUT['paired_vs_ew_full'], OUT['ols_vs_ew'])
for s in OUT['sens']: print(f"{s['param']:10s} {str(s['value']):>6s} CAGR={s['cagr']:5.1f} Sh={s['sharpe']:.2f} DD={s['dd']:.1f} Cal={s['calmar']:.2f}")
