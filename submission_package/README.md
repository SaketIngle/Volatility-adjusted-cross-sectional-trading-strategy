# Track A submission: volatility-adjusted cross-sectional reversal

Long-only, unlevered strategy for the Inter IIT Tech Meet 15.0 Quant Selection, Track A. It runs inside the organizer backtester
(`Abhiraj212024/inter-iit-backtester`) and uses only the supplied `Dataset_PS-A.csv`.

## Contents
| Path | Purpose |
|---|---|
| `src/engine.py` | **The submitted strategy** (`ParticipantStrategy`). Drop-in replacement for the repository's `src/engine.py`. |
| `src/engine_ladder.py`, `src/engine_base.py` | Variants used only for the report's ablation ladder and base-weight comparison. |
| `analysis/` | Scripts that reproduce every table in the report (see below). All paths are relative to the repository root. |
| `requirements.txt` | Pinned dependencies (tested on Python 3.12). |

## Setup
1. Copy the contents of this folder over a clone of the organizer repository (`src/engine.py` is replaced, `analysis/` is added).
2. Place the dataset at `data/Dataset_PS-A.csv`.
3. `python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt`

## Run the strategy
```
python test_submission.py                      # organizer's pre-submission validator
PYTHONHASHSEED=0 python run_backtest.py        # official run with config.yaml (1 bp commission, 0 slippage)
```
The strategy is deterministic: it uses no random numbers. `USE_LOWVOL = False` in `src/engine.py` gives the simpler MOM20 + TS20 variant.

## Reproduce the report (run from the repository root, in this order)
```
python analysis/make_splits.py            # data/pre2020.csv, data/post2020.csv
python analysis/signal_tests.py           # Tables 2 and 4, Figure 2: data statistics, rank IC, Newey-West, quintiles, concentration
python analysis/vol_tests.py              # Diebold-Mariano tests, clustering test on standardised returns, ADF tests
python analysis/walk_forward.py           # Table 9: expanding-window walk-forward on the official engine (~3 min)
python analysis/backtester_ablation.py    # Tables 8 and 10, Figure 4b, Section 5.3: ladder, sub-periods, sensitivity, weights, deflated Sharpe
python analysis/base_weights.py           # Table 7: equal weight / inverse vol / risk parity / minimum variance
python analysis/cost_stress.py            # Table 12: commission and slippage scenarios
python analysis/bootstrap_ci.py           # Table 11: bootstrap intervals for CAGR, Sharpe, drawdown, Calmar
python analysis/long_short_check.py       # Section 5.2: dollar-neutral long-short variant (why the strategy is long-only)
```
Outputs go to `results_analysis/`. Bootstrap procedures use 20-day blocks and seed 42. Set `SLIP_BPS` and `COMM_BPS` in the
environment to rerun `walk_forward.py` with other cost settings.

## Design in one paragraph
Per-asset EWMA volatility (half-life 10 days) gives inverse-volatility base weights. These are tilted by a cross-sectional
z-score of a 20-day reversal score (negative momentum and trend strength) plus a low-volatility term, capped at 15% per name,
smoothed toward the target every 5 days, and scaled so that forecast portfolio volatility stays near 15% (exposure never
above 1, remainder in cash). See Section 5.1 of the report for the formulas.

## Determinism and seeds
- The strategy (`src/engine.py`) uses no random numbers, time or file access.
- The organizer backtester fills orders in the iteration order of a Python `set` of tickers. When cash is tight (for example after an overnight
  gap down in a fully invested book) the last orders in that sequence are only partly filled, so the result depends on Python's hash seed.
  Set `PYTHONHASHSEED=0` for bit-identical output (three runs gave the same `trades.csv`). The analysis scripts set it for themselves.
- The seed only moves the headline numbers in the second decimal. Across ten hash seeds the official run gave CAGR 21.53% to 21.56%,
  Sharpe 1.57 and its interval [0.80, 2.46] every time, maximum drawdown 22.42% to 22.46% and Calmar 0.96.
- All bootstrap procedures use `numpy.random.default_rng(42)`; `research_lib.py` also calls `numpy.random.seed(42)`.
