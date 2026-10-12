"""Signal ladder under identical execution: same inverse-vol base, tilt, smoothing, cap, optional vol-target.
SIGNAL: none | rev5lv (current) | mom20 | mom_ts | raven | raven_lv"""
from __future__ import annotations
import math
from collections import deque
from typing import Dict, List
import numpy as np
from strategy_base import BaseStrategy, Context, Bar


def _z(x):
    s = x.std(ddof=1)
    return (x - x.mean()) / s if s > 1e-12 else np.zeros_like(x)


class LadderStrategy(BaseStrategy):
    SIGNAL = "raven"
    VOL_HL = 10; TILT = 2.0; REBAL = 5; SMOOTH = 0.3; CAP = 0.15
    WARMUP = 60; VOL_TARGET = 0.15; BOOK_HL = 10

    def initialize(self, context):
        self.tickers: List[str] = list(context.universe); n = self.n = len(self.tickers)
        self.alpha = 1 - math.exp(-math.log(2) / self.VOL_HL)
        self.book_alpha = 1 - math.exp(-math.log(2) / self.BOOK_HL)
        self.closes = deque(maxlen=21); self.prev = None
        self.num = np.zeros(n); self.den = 0.0; self.obs = 0
        self.w_book = np.full(n, 1.0 / n); self.b_num = 0.0; self.b_den = 0.0; self.b_obs = 0

    def _signal(self, vol, C):
        # C: (21, n) closes, last row = today
        r5 = -np.log(C[-1] / C[-6]) / (vol * math.sqrt(5))
        mom = C[-1] / C[0] - 1.0
        rets = C[1:] / C[:-1] - 1.0
        sig20 = rets.std(axis=0, ddof=1)
        ts = mom / np.maximum(sig20, 1e-8)
        d = np.diff(C[-15:], axis=0)
        g, l = np.maximum(d, 0).mean(axis=0), np.maximum(-d, 0).mean(axis=0)
        rsi = np.where(l < 1e-12, 100.0, 100 - 100 / (1 + g / np.maximum(l, 1e-12)))
        s = self.SIGNAL
        if s == "rev5lv":   raw = _z(_z(r5) + _z(-vol))
        elif s == "mom20":  raw = -_z(mom)
        elif s == "mom_ts": raw = -(_z(mom) + _z(ts)) / 2
        elif s == "raven":  raw = -(_z(mom) + _z(ts) + _z(rsi)) / 3
        elif s == "mom_ts_lv": raw = _z(-(_z(mom) + _z(ts)) / 2 + _z(-vol))
        elif s == "raven_lv": raw = _z(-(_z(mom) + _z(ts) + _z(rsi)) / 3 + _z(-vol))
        else: return np.zeros(self.n)
        return _z(raw)

    def _target(self, vol, C):
        inv = 1 / vol; base = inv / inv.sum()
        tgt = np.clip(base * (1 + self.TILT * self._signal(vol, C)), 0, None)
        if tgt.sum() <= 0: tgt = base
        tgt /= tgt.sum()
        for _ in range(10): tgt = np.minimum(tgt, self.CAP); tgt /= tgt.sum()
        return tgt

    def on_bar(self, context, bars):
        c = np.array([bars[t].close if t in bars else np.nan for t in self.tickers], float)
        if np.isnan(c).any(): return
        if self.prev is not None:
            rs, rl = c / self.prev - 1, np.log(c / self.prev)
            rb = float(self.w_book @ rs)
            self.b_num = (1 - self.book_alpha) * self.b_num + rb * rb
            self.b_den = (1 - self.book_alpha) * self.b_den + 1.0; self.b_obs += 1
            self.num = (1 - self.alpha) * self.num + rl ** 2
            self.den = (1 - self.alpha) * self.den + 1.0; self.obs += 1
        self.prev = c; self.closes.append(c)
        vol_ok = self.obs >= self.VOL_HL
        vol = np.sqrt(self.num / self.den) if vol_ok else None
        if context.step % self.REBAL == 0 and vol_ok:
            v = np.maximum(vol, 1e-6)
            if context.step >= self.WARMUP and len(self.closes) == 21:
                tgt = self._target(v, np.array(self.closes))
            else:
                inv = 1 / v; tgt = inv / inv.sum()
            w = (1 - self.SMOOTH) * self.w_book + self.SMOOTH * tgt; self.w_book = w / w.sum()
        expo = 1.0
        if self.b_obs >= self.BOOK_HL and self.b_den > 0:
            pv = math.sqrt(self.b_num / self.b_den * 252)
            if pv > 1e-9: expo = min(1.0, self.VOL_TARGET / pv)
        context.set_target_weights({t: float(x) for t, x in zip(self.tickers, self.w_book * expo)})
