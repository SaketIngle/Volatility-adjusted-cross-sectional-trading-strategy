import os, sys
if os.environ.get('PYTHONHASHSEED') != '0':   # fixed hash seed: the backtester fills orders in set-iteration order, so this makes runs bit-identical
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)
sys.path.insert(0, os.getcwd())          # repository root: backtester.py, strategy_base.py, src/
os.makedirs('results_analysis', exist_ok=True)
from strategy_base import BaseStrategy
class BenchEW(BaseStrategy):
    def initialize(self, context): self.t = list(context.universe)
    def on_bar(self, context, bars):
        context.set_target_weights({k: 1.0/len(self.t) for k in self.t})
