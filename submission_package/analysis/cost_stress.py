import os, sys
if os.environ.get('PYTHONHASHSEED') != '0':   # fixed hash seed: the backtester fills orders in set-iteration order, so this makes runs bit-identical
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)
sys.path.insert(0, os.getcwd())          # repository root: backtester.py, strategy_base.py, src/
os.makedirs('results_analysis', exist_ok=True)
import io, contextlib, json, sys, numpy as np
from backtester import Backtester, BacktestConfig
import src.engine_ladder as L
from bench_engine import BenchEW
def run(cls, comm, slip):
    cfg = BacktestConfig(track='A', data_path='data/Dataset_PS-A.csv', initial_cash=100000.0, slippage_rate=slip/1e4, commission_rate=comm/1e4,
                         symbol='ALL', max_ticks=None, save_results=False, output_dir='x', generate_plot=False)
    with contextlib.redirect_stdout(io.StringIO()):
        r = Backtester(cfg).run(cls())
    m = r.metrics_summary
    return dict(cagr=m['CAGR (%)'], sharpe=m['Sharpe Ratio'], ci=m['Sharpe 95% CI'], dd=m['Max Drawdown (%)'], calmar=m['Calmar Ratio'], comm=m['Total Commission Paid ($)'], slip=m['Total Slippage Paid ($)'])
strats = {'Equal-weight': BenchEW,
          'Inverse-vol + vol target': type('S', (L.LadderStrategy,), dict(SIGNAL='none')),
          'MOM20+TS20': type('S', (L.LadderStrategy,), dict(SIGNAL='mom_ts')),
          'RAVEN': type('S', (L.LadderStrategy,), dict(SIGNAL='raven')),
          'FINAL': type('S', (L.LadderStrategy,), dict(SIGNAL='mom_ts_lv'))}
scen = [(1, 0), (1, 5), (1, 10), (5, 5), (10, 10)]
OUT = {}
for sn, cls in strats.items():
    for c, s in scen:
        OUT[f'{sn}|{c}|{s}'] = run(cls, c, s)
        v = OUT[f'{sn}|{c}|{s}']; print(f"{sn:26s} comm={c:2d} slip={s:2d}  CAGR={v['cagr']:5.1f} Sharpe={v['sharpe']:.2f} {v['ci']} DD={v['dd']:.1f} Calmar={v['calmar']:.2f} comm$={v['comm']:.0f} slip$={v['slip']:.0f}")
json.dump(OUT, open('results_analysis/costs.json', 'w'), indent=1, default=float)
