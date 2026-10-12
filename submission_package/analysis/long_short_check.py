"""Dollar-neutral long-short variant of the final score in the official backtester (supports the long-only choice)."""
import os, sys
if os.environ.get('PYTHONHASHSEED') != '0':   # fixed hash seed: the backtester fills orders in set-iteration order, so this makes runs bit-identical
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)
sys.path.insert(0, os.getcwd())
import io, contextlib, math, numpy as np
from collections import deque
from backtester import Backtester, BacktestConfig
from strategy_base import BaseStrategy
import src.engine_ladder as L

class LongShort(L.LadderStrategy):
    """Dollar-neutral variant: long the high-score names, short the low-score names, gross exposure <= 1."""
    SIGNAL = 'mom_ts_lv'
    def _target(self, vol, C):
        s = self._signal(vol, C); x = s / vol
        out = np.zeros(self.n)
        for side, part in ((1, np.clip(x, 0, None)), (-1, np.clip(-x, 0, None))):
            if part.sum() <= 0: continue
            p = part / part.sum() * 0.5
            for _ in range(10):
                p = np.minimum(p, self.CAP); p = p / p.sum() * 0.5
            out += side * p
        return out
    def on_bar(self, context, bars):
        c = np.array([bars[t].close if t in bars else np.nan for t in self.tickers], float)
        if np.isnan(c).any(): return
        if self.prev is not None:
            rs, rl = c / self.prev - 1, np.log(c / self.prev)
            rb = float(self.w_book @ rs)
            self.b_num = (1 - self.book_alpha) * self.b_num + rb * rb; self.b_den = (1 - self.book_alpha) * self.b_den + 1.0; self.b_obs += 1
            self.num = (1 - self.alpha) * self.num + rl ** 2; self.den = (1 - self.alpha) * self.den + 1.0; self.obs += 1
        self.prev = c; self.closes.append(c)
        vol = np.sqrt(self.num / self.den) if self.obs >= self.VOL_HL else None
        if context.step % self.REBAL == 0 and vol is not None:
            v = np.maximum(vol, 1e-6)
            tgt = self._target(v, np.array(self.closes)) if (context.step >= self.WARMUP and len(self.closes) == 21) else np.zeros(self.n)
            self.w_book = (1 - self.SMOOTH) * self.w_book + self.SMOOTH * tgt
        if self.w_book is not None and np.all(self.w_book == 1.0 / self.n): self.w_book = np.zeros(self.n)
        expo = 1.0
        if self.b_obs >= self.BOOK_HL and self.b_den > 0:
            pv = math.sqrt(self.b_num / self.b_den * 252)
            if pv > 1e-9: expo = min(1.0, self.VOL_TARGET / pv)
        g = np.abs(self.w_book).sum()
        w = self.w_book * (min(1.0, 1.0 / g) if g > 1 else 1.0) * expo
        context.set_target_weights({t: float(x) for t, x in zip(self.tickers, w)})

def run(cls, vt=0.15):
    c = type('S', (cls,), dict(VOL_TARGET=vt))
    cfg = BacktestConfig(track='A', data_path='data/Dataset_PS-A.csv', initial_cash=100000.0, slippage_rate=0.0, commission_rate=0.0001, symbol='ALL', max_ticks=None, save_results=False, output_dir='x', generate_plot=False)
    with contextlib.redirect_stdout(io.StringIO()):
        r = Backtester(cfg).run(c())
    m = r.metrics_summary
    return {k: m[k] for k in ('CAGR (%)', 'Sharpe Ratio', 'Sharpe 95% CI', 'Max Drawdown (%)', 'Calmar Ratio', 'Mean Gross Leverage', 'Max Gross Leverage', 'Gross Leverage Violations')}
for name, cls, vt in (('long-short, 15% vol target', LongShort, 0.15), ('long-short, no vol target', LongShort, 1e9), ('long-only final (mom_ts_lv)', type('F', (L.LadderStrategy,), dict(SIGNAL='mom_ts_lv')), 0.15)):
    print(name, run(cls, vt))
