"""Base-weight comparison under identical tilt/smoothing/vol-target: BASE in {invvol, ew, erc, minvar}.
Covariance for erc/minvar: Ledoit-Wolf shrinkage on the last COV_WIN daily returns."""
from __future__ import annotations
from collections import deque
import numpy as np
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from src.engine_ladder import LadderStrategy, _z


class BaseCmpStrategy(LadderStrategy):
    SIGNAL = "mom_ts_lv"; BASE = "invvol"; COV_WIN = 120

    def initialize(self, context):
        super().initialize(context)
        self.rets = deque(maxlen=self.COV_WIN)

    def on_bar(self, context, bars):
        c = np.array([bars[t].close if t in bars else np.nan for t in self.tickers], float)
        if not np.isnan(c).any() and self.prev is not None:
            self.rets.append(c / self.prev - 1.0)
        super().on_bar(context, bars)

    def _base(self, vol):
        n = self.n
        if self.BASE == "ew": return np.full(n, 1.0 / n)
        if self.BASE == "invvol" or len(self.rets) < self.COV_WIN:
            inv = 1.0 / vol; return inv / inv.sum()
        S = LedoitWolf().fit(np.array(self.rets)).covariance_
        if self.BASE == "erc":
            f = lambda w: 0.5 * w @ S @ w - np.log(w).sum() / n
            g = lambda w: S @ w - 1.0 / (n * w)
            r = minimize(f, np.full(n, 1.0 / n), jac=g, method="L-BFGS-B", bounds=[(1e-6, None)] * n)
            w = r.x / r.x.sum(); return w
        if self.BASE == "minvar":
            cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
            r = minimize(lambda w: w @ S @ w, np.full(n, 1.0 / n), jac=lambda w: 2 * S @ w, method="SLSQP",
                         bounds=[(0.0, self.CAP)] * n, constraints=cons)
            w = np.clip(r.x, 0, None); return w / w.sum()
        raise ValueError(self.BASE)

    def _target(self, vol, C):
        base = self._base(vol)
        tgt = np.clip(base * (1.0 + self.TILT * self._signal(vol, C)), 0.0, None)
        if tgt.sum() <= 0: tgt = base
        tgt /= tgt.sum()
        for _ in range(10):
            tgt = np.minimum(tgt, self.CAP); tgt /= tgt.sum()
        return tgt
