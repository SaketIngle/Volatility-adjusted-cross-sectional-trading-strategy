"""
src/engine.py - Track A: Volatility-Adjusted Cross-Sectional Reversal (long-only, no leverage)

Pipeline (all state built online from bars already seen; backtester fills at next open)
  1. Per-asset risk   : EWMA volatility of daily log returns (half-life VOL_HL days)
  2. Base sizing      : inverse volatility weights
  3. Ranking signal   : 20-day cross-sectional REVERSAL = -avg( z(MOM20), z(TS20) ) where
                        MOM20 = C_t / C_{t-20} - 1,  TS20 = MOM20 / std(last 20 daily returns)
                        (+ optional low-volatility risk tilt z(-sigma), USE_LOWVOL)
  4. Tilt + smoothing : w = base * (1 + TILT * signal), clipped >= 0, capped at CAP per name;
                        every REBAL days close SMOOTH of the gap to target (turnover control)
  5. Vol targeting    : scale total exposure by min(1, VOL_TARGET / forecast ann. vol of the
                        unscaled book); remainder held in cash. Gross exposure never exceeds 1.
"""
from __future__ import annotations
import math
from collections import deque
from typing import Dict, List

import numpy as np

from strategy_base import BaseStrategy, Context, Bar


def _z(x: np.ndarray) -> np.ndarray:
    s = x.std(ddof=1)
    return (x - x.mean()) / s if s > 1e-12 else np.zeros_like(x)


class ParticipantStrategy(BaseStrategy):
    # ---------------- parameters ----------------
    USE_LOWVOL = True    # False -> pure MOM20+TS20 reversal (simpler variant)
    MOM_WIN = 20         # momentum / reversal lookback (days)
    VOL_HL = 10          # per-asset EWMA vol half-life (days)
    TILT = 2.0           # signal tilt strength
    REBAL = 5            # rebalance cadence (days)
    SMOOTH = 0.3         # fraction of gap to target closed each rebalance
    CAP = 0.15           # max weight per name
    WARMUP = 60          # steps before the signal is used (inverse-vol only before)
    VOL_TARGET = 0.15    # annualised portfolio vol target (exposure capped at 1)
    BOOK_HL = 10         # half-life for portfolio vol forecast

    def initialize(self, context: Context) -> None:
        self.tickers: List[str] = list(context.universe)
        n = self.n = len(self.tickers)
        self.alpha = 1.0 - math.exp(-math.log(2.0) / self.VOL_HL)
        self.book_alpha = 1.0 - math.exp(-math.log(2.0) / self.BOOK_HL)
        self.closes = deque(maxlen=self.MOM_WIN + 1)
        self.prev_close = None
        self.num = np.zeros(n); self.den = 0.0; self.obs = 0
        self.w_book = np.full(n, 1.0 / n)
        self.b_num = 0.0; self.b_den = 0.0; self.b_obs = 0

    def _signal(self, vol: np.ndarray, C: np.ndarray) -> np.ndarray:
        mom = C[-1] / C[0] - 1.0
        rets = C[1:] / C[:-1] - 1.0
        ts = mom / np.maximum(rets.std(axis=0, ddof=1), 1e-8)
        raw = -(_z(mom) + _z(ts)) / 2.0                 # reversal: losers score high
        if self.USE_LOWVOL:
            raw = _z(raw) + _z(-vol)                    # calm names score higher (risk tilt)
        return _z(raw)

    def _target(self, vol: np.ndarray, C: np.ndarray) -> np.ndarray:
        inv = 1.0 / vol
        base = inv / inv.sum()
        tgt = np.clip(base * (1.0 + self.TILT * self._signal(vol, C)), 0.0, None)
        if tgt.sum() <= 0:
            tgt = base
        tgt /= tgt.sum()
        for _ in range(10):
            tgt = np.minimum(tgt, self.CAP)
            tgt /= tgt.sum()
        return tgt

    def on_bar(self, context: Context, bars: Dict[str, Bar]) -> None:
        c = np.array([bars[t].close if t in bars else np.nan for t in self.tickers], dtype=float)
        if np.isnan(c).any():                           # missing bar: keep previous targets
            return
        if self.prev_close is not None:
            r_simple = c / self.prev_close - 1.0
            r_log = np.log(c / self.prev_close)
            rb = float(np.dot(self.w_book, r_simple))   # shadow unscaled-book return
            self.b_num = (1 - self.book_alpha) * self.b_num + rb * rb
            self.b_den = (1 - self.book_alpha) * self.b_den + 1.0
            self.b_obs += 1
            self.num = (1 - self.alpha) * self.num + r_log ** 2
            self.den = (1 - self.alpha) * self.den + 1.0
            self.obs += 1
        self.prev_close = c
        self.closes.append(c)

        vol = np.sqrt(self.num / self.den) if self.obs >= self.VOL_HL else None

        if context.step % self.REBAL == 0 and vol is not None:
            v = np.maximum(vol, 1e-6)
            if context.step >= self.WARMUP and len(self.closes) == self.MOM_WIN + 1:
                tgt = self._target(v, np.array(self.closes))
            else:
                inv = 1.0 / v
                tgt = inv / inv.sum()
            w = (1 - self.SMOOTH) * self.w_book + self.SMOOTH * tgt
            self.w_book = w / w.sum()

        expo = 1.0
        if self.b_obs >= self.BOOK_HL and self.b_den > 0:
            pv = math.sqrt(self.b_num / self.b_den * 252.0)
            if pv > 1e-9:
                expo = min(1.0, self.VOL_TARGET / pv)

        weights = self.w_book * expo
        context.set_target_weights({t: float(w) for t, w in zip(self.tickers, weights)})
