# Track A Submission: Volatility-Adjusted Cross-Sectional Reversal

**Competition:** Inter IIT Tech Meet 15.0 — Quant Selection  
**Track:** A — Volatility-Adjusted Cross-Sectional Strategy

This project implements a long-only, unlevered cross-sectional strategy using the organizer-provided `Dataset_PS-A.csv`. It combines a 20-day reversal framework with trend-strength information and a low-volatility preference, alongside volatility-aware portfolio construction.

The participant strategy is intended to run inside the organizer's backtesting framework: [`Abhiraj212024/inter-iit-backtester`](https://github.com/Abhiraj212024/inter-iit-backtester).

## 1. Actual Package Structure

```text
submission_package/
├── README.md
├── requirements.txt
├── src/
│   ├── engine.py
│   ├── engine_base.py
│   └── engine_ladder.py
├── analysis/
│   ├── backtester_ablation.py
│   ├── base_weights.py
│   ├── bench_engine.py
│   ├── bootstrap_ci.py
│   ├── cost_stress.py
│   ├── long_short_check.py
│   ├── make_splits.py
│   ├── research_lib.py
│   ├── signal_tests.py
│   ├── vol_tests.py
│   └── walk_forward.py
└── References/
    ├── Annualized volatality.png
    ├── ATR hypothesis.pdf
    ├── Breakout Hypothesis.pdf
    ├── Corelation of 20-20 Momentum and result.png
    ├── Corelation of stocks.png
    ├── Dataset_PS-A.csv
    ├── Momentum + Volitility.pdf
    ├── Momentum Hypothesis.pdf
    ├── Momentum+Trend Strength.pdf
    ├── Moving average hypothesis.pdf
    ├── Relative volume.pdf
    ├── Roadmap.png
    ├── RSI hypothesis.pdf
    ├── Trend strength Hypothesis.pdf
    ├── Volitality Hypothesis.pdf
    └── volume hypothesis.pdf
```

### File and folder descriptions

| Path | Purpose |
|---|---|
| `src/engine.py` | Main submitted strategy (`ParticipantStrategy`). Intended as a replacement for the organizer repository's `src/engine.py`. |
| `src/engine_base.py` | Alternative implementation used for base-weight comparisons. |
| `src/engine_ladder.py` | Strategy variant used in the report's ablation analysis. |
| `analysis/` | Scripts for signal tests, volatility tests, walk-forward evaluation, ablations, portfolio-weight comparisons, cost stress, bootstrap intervals, and long-short comparison. |
| `analysis/bench_engine.py` | Supporting benchmark-engine code. |
| `analysis/research_lib.py` | Shared utilities used by research scripts. |
| `requirements.txt` | Python dependencies; the original project notes specify Python 3.12. |
| `References/` | Supporting hypothesis reports, exploratory plots, a roadmap image, and the dataset file listed above. |
| `README.md` | Project setup, strategy design, execution, and research-reproduction instructions. |

## 2. Strategy Design

Per-asset exponentially weighted volatility estimates, using a 10-day half-life, determine inverse-volatility base weights. These are tilted using cross-sectional standardized values of a 20-day reversal score based on negative momentum and trend strength, with a low-volatility term included in the ranking.

The project description specifies the following portfolio controls:

- Maximum individual asset weight: 15%.
- Rebalancing interval: every five trading bars.
- Smoothing toward target weights.
- Forecast portfolio-volatility target: approximately 15%.
- Long-only exposure capped at 100%, with unused capital held as cash.

See Section 5.1 of the technical report and `src/engine.py` for the exact formulas and implementation. Setting `USE_LOWVOL = False` in `src/engine.py` selects the simpler MOM20 + TS20 variant, according to the original project notes.

## 3. Setup

### Prerequisites

- Python 3.12, as specified in the original project notes.
- The organizer's backtesting repository.
- The organizer-provided `Dataset_PS-A.csv`.

Clone the organizer's repository and copy the submission package's files into the corresponding locations. The submitted `src/engine.py` replaces the organizer repository's `src/engine.py`; the `analysis/` directory is added for research reproduction.

Place the dataset at the path expected by the backtester, described in the original instructions as:

```text
data/Dataset_PS-A.csv
```

Create and activate a virtual environment.

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux or macOS**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Run the Strategy

Run commands from the root of the organizer's backtesting repository. Confirm that the command names and configuration match the version supplied by the organizers.

### Pre-submission validation

```bash
python test_submission.py
```

### Backtest

```bash
python run_backtest.py
```

The original project notes describe the run as using `config.yaml` with 1 basis point commission and zero slippage. Verify the actual configuration before interpreting results.

## 5. Reproduce the Research

The original README gives the following analysis order. Run these commands from the root of the combined organizer-backtester project, after confirming the expected data paths and dependencies:

```bash
python analysis/make_splits.py
python analysis/signal_tests.py
python analysis/vol_tests.py
python analysis/walk_forward.py
python analysis/backtester_ablation.py
python analysis/base_weights.py
python analysis/cost_stress.py
python analysis/bootstrap_ci.py
python analysis/long_short_check.py
```

| Script | Purpose described in the original project notes |
|---|---|
| `make_splits.py` | Creates `data/pre2020.csv` and `data/post2020.csv`. |
| `signal_tests.py` | Data statistics, rank IC, Newey–West inference, quintile returns, and concentration. |
| `vol_tests.py` | Diebold–Mariano tests, standardized-return clustering analysis, and ADF tests. |
| `walk_forward.py` | Expanding-window walk-forward evaluation. |
| `backtester_ablation.py` | Ablation ladder, sub-periods, sensitivity analysis, portfolio weights, and deflated Sharpe analysis. |
| `base_weights.py` | Equal-weight, inverse-volatility, risk-parity, and minimum-variance comparisons. |
| `cost_stress.py` | Commission and slippage scenarios. |
| `bootstrap_ci.py` | Bootstrap intervals for CAGR, Sharpe, drawdown, and Calmar. |
| `long_short_check.py` | Dollar-neutral long-short comparison. |

The original notes say outputs are written to `results_analysis/`, bootstrap procedures use 20-day blocks and seed 42, and `SLIP_BPS` / `COMM_BPS` can be set to vary costs for `walk_forward.py`. These behaviors should be confirmed against the scripts before relying on them.

## 6. Determinism and Reported Results

The original project notes state that `src/engine.py` does not itself use random numbers, time, or file access. They also report that ticker-order iteration in the organizer backtester can affect partial fills when cash is tight, making results sensitive to Python's hash seed. The notes recommend setting `PYTHONHASHSEED=0` for repeatability.

The original README reports the following results across ten hash-seed settings:

| Metric | Reported result |
|---|---:|
| CAGR | 21.53%–21.56% |
| Sharpe ratio | 1.57 |
| Sharpe interval | 0.80–2.46 |
| Maximum drawdown | 22.42%–22.46% |
| Calmar ratio | 0.96 |

These are figures reported by the project's existing notes, not independent verification. Treat them as validated only after the submitted code and configuration reproduce them using the organizer's backtester.

The original notes describe bootstrap procedures using `numpy.random.default_rng(42)` and state that `analysis/research_lib.py` also calls `numpy.random.seed(42)`.

## 7. References and Supporting Research

The `References/` folder contains supporting materials and exploratory work on volatility, ATR, breakouts, momentum, momentum plus volatility, momentum plus trend strength, moving averages, relative volume, RSI, trend strength, and volume. It also contains correlation plots, an annualized-volatility plot, a roadmap image, and `Dataset_PS-A.csv`.

These files are included as project materials; their presence alone does not establish that every hypothesis was accepted or used in the final strategy. Consult the technical report for which tests were retained, rejected, or used as supporting evidence.
