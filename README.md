# Track A Submission: Volatility-Adjusted Cross-Sectional Reversal

**Competition:** Inter IIT Tech Meet 15.0 — Quant Selection  
**Track:** A — Volatility-Adjusted Cross-Sectional Strategy

This project implements a long-only, unlevered cross-sectional strategy using the organizer-provided `Dataset_PS-A.csv`. It combines a 20-day reversal framework with trend-strength information and a low-volatility preference, alongside volatility-aware portfolio construction.

The participant strategy is intended to run inside the organizer's backtesting framework: [Abhiraj212024/inter-iit-backtester](https://github.com/Abhiraj212024/inter-iit-backtester).

## 1. Project Structure

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

### File Descriptions

| File or Directory | Description |
|---|---|
| `src/engine.py` | Main submitted strategy implementation (`ParticipantStrategy`). |
| `src/engine_base.py` | Alternative implementation used for base-weight comparisons. |
| `src/engine_ladder.py` | Strategy variant used for ablation analysis. |
| `analysis/` | Scripts for signal testing, volatility analysis, walk-forward evaluation, ablations, weight comparisons, cost stress tests, bootstrap intervals, and long-short comparisons. |
| `analysis/research_lib.py` | Shared research utilities. |
| `analysis/bench_engine.py` | Supporting benchmark-engine code. |
| `requirements.txt` | Python dependencies. |
| `References/` | Supporting hypothesis reports, plots, roadmap, and dataset file. |

## 2. Strategy Overview

The strategy combines cross-sectional reversal signals with trend-strength information and a low-volatility preference.

Its portfolio construction uses exponentially weighted volatility estimates with a 10-day half-life to calculate inverse-volatility base weights. These weights are adjusted using standardized cross-sectional signals.

The project description specifies the following portfolio controls:

- Maximum individual asset weight: 15%.
- Rebalancing interval: every five trading bars.
- Weight smoothing toward target allocations.
- Target portfolio volatility: approximately 15%.
- Long-only exposure, capped at 100%, with unused capital held as cash.

The main implementation is located in `src/engine.py`. Refer to the technical report for the detailed mathematical formulation and justification of the strategy.

The original project notes also describe a simpler MOM20 + TS20 variant selected by setting `USE_LOWVOL = False` in `src/engine.py`.

## 3. Setup

### Prerequisites

- Python 3.12, as specified in the original project notes.
- The organizer's backtesting repository.
- The organizer-provided `Dataset_PS-A.csv`.

Clone the organizer's repository:

```bash
git clone https://github.com/Abhiraj212024/inter-iit-backtester.git
```

Copy the submission files into the corresponding locations in the backtester repository. The submitted `src/engine.py` is intended to replace the corresponding engine file in the organizer's repository.

Place the dataset at the location expected by the backtester:

```text
data/Dataset_PS-A.csv
```

### Install Dependencies

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

## 4. Running the Strategy

Run commands from the root directory of the combined organizer-backtester project. Confirm the command names and configuration against the version provided by the organizers.

### Validate the Submission

```bash
python test_submission.py
```

### Run the Backtest

```bash
python run_backtest.py
```

The original project notes describe a configuration using 1 basis point commission and zero slippage. Verify the actual configuration before interpreting or reporting results.

## 5. Research and Analysis

The `analysis/` directory contains scripts for evaluating the signals and testing alternative portfolio-construction choices.

Run the following commands from the root of the combined project after confirming the expected data paths and dependencies:

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

| Script | Purpose |
|---|---|
| `make_splits.py` | Creates pre-2020 and post-2020 data splits. |
| `signal_tests.py` | Examines data statistics, rank information coefficients, statistical significance, quintile returns, and concentration. |
| `vol_tests.py` | Performs volatility-related statistical tests and time-series diagnostics. |
| `walk_forward.py` | Evaluates performance using expanding-window walk-forward analysis. |
| `backtester_ablation.py` | Compares strategy variants and examines sub-periods, sensitivity, portfolio weights, and Sharpe-related diagnostics. |
| `base_weights.py` | Compares equal-weight, inverse-volatility, risk-parity, and minimum-variance approaches. |
| `cost_stress.py` | Evaluates alternative commission and slippage assumptions. |
| `bootstrap_ci.py` | Estimates bootstrap intervals for CAGR, Sharpe ratio, drawdown, and Calmar ratio. |
| `long_short_check.py` | Compares the strategy with a dollar-neutral long-short approach. |

The original project notes describe analysis outputs under `results_analysis/` and bootstrap procedures using 20-day blocks with seed 42. Confirm these details against the actual scripts before relying on them.

## 6. Reported Performance

The original project notes report the following performance figures across ten hash-seed settings:

| Metric | Reported Result |
|---|---:|
| CAGR | 21.53%–21.56% |
| Sharpe Ratio | 1.57 |
| Sharpe Interval | 0.80–2.46 |
| Maximum Drawdown | 22.42%–22.46% |
| Calmar Ratio | 0.96 |

These figures are reported by the existing project notes and should not be interpreted as independently verified results. Final performance should be confirmed by running the submitted implementation with the organizer's backtester and the prescribed configuration.

The original project notes also describe using `PYTHONHASHSEED=0` to improve repeatability when ticker-order iteration affects partial fills.

## 7. References and Supporting Research

The `References/` directory contains supporting materials and exploratory research on:

- Momentum and momentum combined with volatility.
- Trend strength and moving averages.
- Volatility and average true range (ATR).
- Breakout signals.
- Relative volume and volume-based signals.
- Relative strength index (RSI).
- Correlation analysis and annualized volatility.
- Research roadmap and supporting plots.

These files document the exploratory research associated with the strategy. Their inclusion does not imply that every tested hypothesis was accepted or incorporated into the final implementation. Refer to the technical report for the corresponding findings and rationale.

## Competition Information

**Competition:** Inter IIT Tech Meet 15.0 — Quant Selection  
**Track:** A — Volatility-Adjusted Cross-Sectional Strategy
