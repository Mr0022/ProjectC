# The full benchmark — 10 models + HAR-RV × 10 datasets × h = 1, 5, 22

Trains every deep model on every realized-variance series at every horizon,
fits HAR-RV on the same rows, and writes the metric tables **and** the
per-observation loss series that Diebold–Mariano and the Model Confidence Set
need.

```bash
python orchestrate/run_benchmark.py                    # the whole grid (3000 trainings + 10 HAR fits)
python orchestrate/run_benchmark.py --dry_run          # print the plan, train nothing
python orchestrate/run_benchmark.py --validate         # build every config, forward one batch, stop
python orchestrate/run_benchmark.py --assets crypto    # or --datasets btcusdt --models FITS --horizons 5
python orchestrate/aggregate_results.py                # re-score what is on disk, no retraining
```

Run from the repository root (`exp_basic` discovers models by scanning the
relative path `models/`). Everything lands in `orchestrate/results/`, which is
git-ignored — `results/` matches at any depth.

---

## 1. The grid

| | |
|---|---|
| **Models** | DLinear, PatchTST, iTransformer, TimesNet, MSGNet, TimeMixer, FITS, TSLANet, ModernTCN, AdaWaveNet, **HAR-RV** |
| **Forex** | EURUSD, AUDUSD, EURGBP, USDCHF, USDJPY — train ..2022-12 / val 2023-01..2024-12 / test 2025-01.. |
| **Crypto** | btcusdt, ethusdt, adausdt, bnbusdt, xrpusdt — train 2018-06..2024-06 / val 2024-07..2025-06 / test 2025-07..2026-06 |
| **Horizons** | h = 1, 5, 22 |

10 × 10 × 3 = **300 deep-model cells**, each repeated over 10 seeds, plus
**10 HAR-RV fits** (one fit covers all three horizons). Splits come from `data_provider/splits.py`, so both model
families read the same calendar.

Forecasts per test window, after the h−1 embargo at each edge:

| | h=1 | h=5 | h=22 |
|---|---|---|---|
| forex | 387 | 383 | 366 |
| crypto | 365 | 361 | 344 |

## 2. The protocol (identical for every cell)

Every deep-model cell runs under the flags the tuning study used, at a fixed
training budget:

```
--aggregate_mean --features S --target RV --seq_len 96 --label_len 48
--train_epochs 50 --patience 10
```

The budget is the one part of the protocol the orchestrator sets itself rather
than inheriting: `train_epochs` and `patience` are in `ORCHESTRATOR_FLAGS`, so
whatever an anchor's tuned command line said about them is dropped and these
values are used instead. The tuning study searched at 30 / 7, so anchors are
being reused at a longer budget than they were selected under — deliberate, but
worth stating, and worth re-tuning if the budget moves again. Note also that
`--lradj cosine` anchors divide by `train_epochs` to shape the learning-rate
schedule, so for those the cap is not merely a stopping point.

so the model emits **one** number per window — the mean of RV over the next h
days — which is exactly HAR-RV's target `Y^(h)`. HAR-RV is fitted by
`HAR-RV_RUN.PY`, unmodified. Same target, same rows, same actuals, so the two
families' losses can be set side by side and fed to the same tests.

Each cell is repeated **`--itr 10`** times, seeds 2021–2030 — run.py's rule
that repeat *i* uses `--seed + i`, so `--itr N --seed S` here and in `run.py`
are the same N runs, one subprocess each. That is 3 000 trainings; `--itr 3`
is the cheaper setting.

### Scale: `--log` or raw

The scale is a property of the **anchor**, not a flag of the sweep — a model is
scored on the scale it was tuned and trained on:

| anchor's command line | scale | target | QLIKE computed on |
|---|---|---|---|
| has `--log` | `ln_RV` | `ln( mean RV )` | the Jensen back-transform `exp(pred + bias + σ²/2)` |
| no `--log` | `raw_RV` | `mean RV` | the forecast itself |

The tuned anchors in `tuning/ProjectC_tuning` all carry `--log`, so today every
sweep is `ln_RV`. Tune a study without `--log`, point `--anchor_dir` at it, and
the whole pipeline follows: the target, HAR-RV's own invocation, the losses and
the tables. Nothing needs a flag — `run_benchmark.py` prints the scale it
derived, and refuses a sweep whose anchors disagree, because one table cannot
carry MSE in two different units. Give the raw sweep its own `--results_dir`.

What changes in raw mode:

* **No Jensen correction** — a raw model forecasts the variance directly, so
  the extra training pass is skipped and `bias`/`resid_var` are 0.
* **`MSE_ln`/`MAE_ln` are empty**, and no `se_ln`/`ae_ln` loss matrices are
  written: a raw forecast can be ≤ 0, where the log does not exist.
* **Non-positive forecasts become possible.** MSE/MAE are measured against
  `max(forecast, 0)` and QLIKE floors them at `1e-4 × mean training RV`, the
  same two rules `HAR-RV_RUN.PY` and `Exp_Long_Term_Forecast` apply;
  `n_floored` in `metrics.csv` counts them.
* **`MSE_RV`, `MAE_RV` and `QLIKE` stay comparable to a `--log` sweep** — `exp()`
  of the log target is the raw target exactly, at every horizon, so the two
  runs are scored against identical actuals on that scale.

`pred_len` is the only thing the horizon changes. Under `--aggregate_mean` the
forecast head is always built with `pred_len = 1`
(`Exp_Long_Term_Forecast._build_model`), so h changes the aggregation window of
the target and nothing about any architecture — no model's shape constraints
are a function of h, which is why the h=1 anchors transfer to h=5 and h=22
without a single re-derived flag.

## 3. Hyper-parameters

Taken verbatim from `tuning/ProjectC_tuning/<Model>_best.json` — not the
`best_params` block alone but the whole winning **command line**, so every flag
the search fixed (architecture, learning rate, batch size, `lradj`, dropout,
regularisers) is reproduced exactly. The orchestrator overrides only what a
benchmark cell must own:

```
--data --root_path --data_path --pred_len --model_id --des
--itr --seed --checkpoints --num_workers --train_epochs --patience
```

Each cell's `.npz` stores the full `python -u run.py …` line that produced it,
so any number in the table can be reproduced on its own.

⚠️ **The anchors were tuned on EUR/USD at h = 1.** Applying them to the other
nine datasets and to h = 5 / 22 is a *transfer*, not a per-cell search. It is
the fair-comparison choice — every model gets the same treatment on every
series — but it should be stated wherever the table is reported, since a model
whose optimum moves with the series is penalised relative to one whose optimum
does not.

Per-dataset tuning is a drop-in when you want it: put
`tuning/ProjectC_tuning/<dataset>/<Model>_best.json` in place and that dataset
picks it up automatically. Nothing else changes.

## 4. What gets written

```
orchestrate/results/
├── runs/<dataset>/h<hh>/<Model>_seed<seed>.npz    per-observation forecasts (the raw record)
├── har/<dataset>/har_rv[_log]_*.csv|png|pdf      HAR-RV_RUN.PY's own output, untouched
├── tables/metrics.csv                             one row per (dataset, horizon, model, seed)
├── tables/metrics_mean.csv                        seed mean ± std
├── tables/metrics_seedmean.csv                    metrics of the seed-AVERAGED forecast
├── tables/pivot_<metric>_h<hh>.csv                models × datasets, one metric, one horizon
├── forecasts/<dataset>_h<hh>__seed<S>.csv         y_pred_scale, y_rv + <Model>__pred, <Model>__rv
├── losses/<dataset>_h<hh>__<loss>__seed<S>.csv    ← the DM / MCS input, per seed
├── losses/<dataset>_h<hh>__<loss>__seedmean.csv   ← and for the seed-averaged forecast
├── failures.csv                                   cells that failed, and why
└── manifest.json                                  coverage, dates, QLIKE floors, checks
```

A cell stores **forecasts**, not losses: a loss can be changed afterwards
without retraining anything, and `aggregate_results.py` can be re-run on its own
at any time.

### Metrics

`metrics.csv` carries, per cell:

| column | meaning |
|---|---|
| `scale` | `ln_RV` or `raw_RV` — which scale this cell was fitted and is reported on |
| `MSE_ln`, `MAE_ln` | errors in **ln(RV)**. `--log` sweeps only; empty for a raw cell |
| `MSE_RV`, `MAE_RV` | the same errors after the lognormal back-transform, on the **variance** scale |
| `QLIKE` | `mean( RV/RV̂ − ln(RV/RV̂) − 1 )` (Patton, 2011), a variance loss, hence computed on the back-transformed forecast |
| `bias`, `resid_var` | the Jensen terms: `RV̂ = exp(pred + bias + σ²/2)`, both measured on **training** residuals |
| `n_floored` | non-positive variance forecasts floored for QLIKE — 0 under `--log` by construction, possible on the raw scale |
| `target_dev` | how far this cell's own actuals sit from `Y^(h)` rebuilt from the CSV (see §6) |
| `val_loss` | MSE on the full validation split for the checkpoint early stopping kept |

`ln`-scale and `RV`-scale losses are **not** comparable to each other; either is
comparable across models and horizons, because both families predict the same
arithmetic forward mean. The `RV` columns are comparable across scales too, so
a raw sweep and a `--log` one can be set side by side there.

## 5. Running DM and MCS

`losses/<dataset>_h<hh>__<loss>__seed<S>.csv` is a date-indexed matrix, one row
per forecast and one column per model — one file per seed, so `--itr 10` gives
ten of them per loss.

**Run the tests on `__seedmean`.** It is the same matrix for the forecast each
model's ten repeats average to: one series per model, so one DM statistic and
one MCS instead of ten that cannot be pooled. The repeats are averaged on both
scales separately — the ln-scale losses see the mean ln forecast, QLIKE and the
variance-scale losses see the mean variance forecast — and HAR-RV, being
deterministic, is carried through unchanged. Its metrics are in
`tables/metrics_seedmean.csv`, and they are **not** the seed means in
`metrics_mean.csv`: the loss of an average sits below the average of the losses
whenever the loss is convex, so the ensemble reads better than any single
repeat. It is a different forecast, not a smoothed report of the same one — use
`metrics_mean.csv` (mean ± std) when the table is about one model, and the
`seedmean` files when the question is which model wins.

The losses, for each of:

| file suffix | loss |
|---|---|
| `se_ln` | squared error, ln(RV) — `--log` sweeps only |
| `ae_ln` | absolute error, ln(RV) — `--log` sweeps only |
| `qlike` | QLIKE, variance scale |
| `se_rv` | squared error, variance scale |
| `ae_rv` | absolute error, variance scale |

The mean of a column **is** the corresponding cell in `metrics.csv`, so the
table and the tests cannot disagree.

```python
import pandas as pd
L = pd.read_csv('orchestrate/results/losses/EURUSD_h05__qlike__seedmean.csv',
                index_col=0, parse_dates=True)
d = L['TimeMixer'] - L['HAR-RV']          # DM loss differential
M = L.values                              # MCS: T × 11 loss matrix
```

Two things the h > 1 rows require:

* **Overlapping targets.** At horizon h the target windows of consecutive rows
  share h−1 days, so `d_t` is autocorrelated *by construction*. The DM
  long-run variance must be HAC-estimated with at least h−1 lags (Newey–West);
  the Harvey–Leybourne–Newbold small-sample correction is the usual companion
  at T ≈ 350–390. `HAR-RV_RUN.PY` uses `max(2(h−1), Newey-West 1994 plug-in)`
  for the same reason — 0, 8 and 42 at h = 1, 5, 22.
* **The MCS resamples blocks**, so its block length has to respect the same
  overlap; the same h−1 is the floor.

Both tests need the whole model set present for a given (dataset, horizon) — a
MCS over a subset is a different test, not a partial one. `manifest.json` lists
which models each block holds, and `aggregate_results.py` says so on stdout
when a block is short.

The tests themselves are **not** implemented here: this produces their inputs.

## 6. What is checked, so the tests are not run on a misalignment

A DM statistic computed across misaligned rows looks perfectly healthy. So:

* **Dates.** Every forecast is indexed by the date its target window opens.
  `benchmark_config.test_target_dates` derives that index from the split
  calendar alone; a cell whose forecast count disagrees fails at its own cell,
  and HAR-RV's exported rows are compared against it before conversion.
* **Actuals.** `benchmark_config.horizon_target` rebuilds `Y^(h)` from the CSV
  in float64 — a third computation, independent of both families — and every
  model is *scored against that*, with the gap to its own stored copy recorded
  as `target_dev`. Expect ~1e-7 for the networks (a float32 round trip through
  the loader's `StandardScaler`) and ~1e-16 for HAR-RV. Anything larger means a
  cell is not predicting the same object as the rest of the grid, and
  `--strict` turns it into an error.

## 7. Cost, resuming and failures

300 trainings of up to 50 epochs. On a T4 the cheap models (FITS, DLinear,
TSLANet) are seconds each and TimesNet/MSGNet dominate; the whole grid is an
overnight job. On CPU it is only realistic as a smoke test — use `--quick`
(5 epochs, patience 3) and `--limit`.

* **Resumable.** A cell whose `.npz` exists is skipped, so re-running the same
  command continues where it stopped. A failed cell is recorded in
  `failures.csv` and skipped too (`--retry_failed` re-runs those).
* **One subprocess per cell**, so an OOM kill or a CUDA fault costs one cell
  rather than the sweep.
* **Ordered dataset → horizon → model**, so an interrupted run leaves complete
  (dataset, horizon) blocks behind — the unit the MCS needs.
* `--quick` results are indistinguishable from real ones on the resume path
  (only `train_epochs` in the metadata says otherwise), so point a smoke test at
  a separate `--results_dir`.
* Checkpoints go to `--checkpoint_dir` (default `./checkpoints/benchmark`) and
  are deleted after each cell; point it at local disk when `--results_dir` is on
  a Drive mount.

## 8. Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mr0022/ProjectC/blob/claude/orchestrate-model-eval-forex-crypto-rir1dh/orchestrate/colab_benchmark.ipynb)

`orchestrate/colab_benchmark.ipynb` — mounts Drive, clones the repo, installs
what Colab lacks, runs the sweep with `--results_dir` on Drive and
`--checkpoint_dir` on local disk, and prints the summary tables. A disconnected
session is resumed by re-running the setup cells and the sweep cell.

## 9. Two changes this made to `exp/exp_long_term_forecasting.py`

Both are in the `--log` branch of `_report_rv_metrics`, i.e. they affect what
`run.py` prints for a `--log --aggregate_mean` run. Neither touches training,
model selection or any `ln`-scale number.

1. **QLIKE argument order.** `utils.metrics.QLIKE(pred, true, floor)` forms the
   ratio `true/pred`; the log branch was calling it `QLIKE(actual, forecast)`,
   which evaluates `1/r + ln r − 1` instead of `r − ln r − 1`. The two agree to
   second order — the reported QLIKE was out by ~0.05 % on EUR/USD, not by an
   order of magnitude — but they are different losses, and only the corrected
   one matches `HAR-RV_RUN.PY`'s `qlike()`. Visible fingerprint: `QLIKE_naive`
   used to print *below* the Jensen-corrected QLIKE, i.e. the correction looked
   harmful; it now prints above it, which is the direction the theory predicts.
2. **The Jensen terms are now measured over a sequential pass of the whole
   training split.** They were measured over the training *loader*, which
   shuffles and drops its last partial batch — so `bias` and `σ²`, and every
   variance-scale metric built on them, depended on the RNG state and moved
   between identical runs. They are part of the forecast, so they have to be
   reproducible.
