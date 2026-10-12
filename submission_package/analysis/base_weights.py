import os, sys
if os.environ.get('PYTHONHASHSEED') != '0':   # fixed hash seed: the backtester fills orders in set-iteration order, so this makes runs bit-identical
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)
sys.path.insert(0, os.getcwd())          # repository root: backtester.py, strategy_base.py, src/
os.makedirs('results_analysis', exist_ok=True)
import io, contextlib, json, sys, time, numpy as np
from backtester import Backtester, BacktestConfig
from src.engine_base import BaseCmpStrategy
def run(**ov):
    cls = type('S', (BaseCmpStrategy,), ov)
    cfg = BacktestConfig(track='A', data_path='data/Dataset_PS-A.csv', initial_cash=100000.0, slippage_rate=0.0, commission_rate=0.0001,
                         symbol='ALL', max_ticks=None, save_results=False, output_dir='x', generate_plot=False)
    with contextlib.redirect_stdout(io.StringIO()):
        r = Backtester(cfg).run(cls())
    m = r.metrics_summary; return dict(cagr=m['CAGR (%)'], sharpe=m['Sharpe Ratio'], ci=m['Sharpe 95% CI'], dd=m['Max Drawdown (%)'], calmar=m['Calmar Ratio'], turn=m['Total Turnover (%)'])
OUT = {}
for b in ('ew', 'invvol', 'erc', 'minvar'):
    t0 = time.time(); OUT[b] = run(BASE=b); OUT[b]['secs'] = round(time.time() - t0, 1)
    OUT[b + '_notilt'] = run(BASE=b, TILT=0.0)
    print(b, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in OUT[b].items()}, '| notilt', {k: round(v, 2) for k, v in OUT[b + '_notilt'].items() if k in ('cagr', 'sharpe', 'dd')})
json.dump(OUT, open('results_analysis/analysis3.json', 'w'), indent=1, default=float)
