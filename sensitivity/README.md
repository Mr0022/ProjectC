# OFAT hyper-parameter sensitivity — the tuned models, one knob at a time

How much does each hyper-parameter actually matter, once the model is tuned?
This directory answers that for every model that has a `<Model>_best.json` in
`tuning/ProjectC_tuning/`: it anchors on the configuration Optuna selected,
sweeps **one hyper-parameter at a time** over the range that was searched,
trains every swept point with **five seeds**, and turns the result into the
figures and tables a paper needs.

```bash
python sensitivity/ofat_sensitivity.py --dry_run          # the plan and every command
python sensitivity/ofat_sensitivity.py --validate         # build every point, train none
python sensitivity/ofat_sensitivity.py                    # every model with an anchor
python sensitivity/ofat_sensitivity.py --models ModernTCN # one model
python sensitivity/ofat_plots.py                          # figures + summary tables
```

On Colab, open `sensitivity/colab_ofat.ipynb` (§9). Run everything from the
repository root — `exp_basic` discovers models by scanning the relative path
`models/`.

| File | What it is |
|---|---|
| `ofat_grids.py` | the sweep grids and the architectural clamps — the OFAT twin of `tuning/search_spaces.py` |
| `ofat_sensitivity.py` | the runner: builds each point's `run.py` command, trains it, parses the metrics |
| `ofat_plots.py` | panels, tornado, heatmap, summary CSVs |
| `colab_ofat.ipynb` | the same sweep on a Colab GPU, writing to Drive |
| `ofat_results.csv` | long format, one row per (model, knob, value, seed) |

---

## 1. What OFAT shows, and what it does not

Each panel holds **every other hyper-parameter at the tuned optimum** and moves
one. So a curve answers:

> given that the rest of the configuration is tuned, how much does this knob
> move the test metric, and is the tuned value the best of the ones searched?

It does **not** answer "what is the best value of this knob in general".
A knob can look inert here because a partner knob has already absorbed its
effect — a small `d_model` with a large `d_ff` is not the same experiment as a
small `d_model` alone. **Interactions are invisible to OFAT by construction**,
which is the price of the method: full-factorial sensitivity over 14 knobs is
thousands of configurations, OFAT is dozens. The figures state this on the
subtitle so a reader cannot take a curve for a global one.

The second thing that must be read off the figures is **seed noise**. Five
seeds per point exist so that the movement of a curve can be compared with the
spread of the anchor. A knob whose whole sweep sits inside the anchor's ±1 sd
band has not been shown to matter, and the tornado plot marks it as such
rather than ranking it as a small effect.

## 2. The anchor

The sweep centre is the `best_params` block of `tuning/ProjectC_tuning/<Model>_best.json`
— the winner of that model's 50-trial study. Everything else in that file's
`command` (dataset, `seq_len 96`, `pred_len 1`, `--aggregate_mean --log`,
epochs, patience) is reused **verbatim** as the base of every OFAT command, so
a swept point differs from the tuned run in exactly one flag and nothing else.
`optuna.db` is never opened; the JSON is the interface.

The anchor is trained **once per model** and reused as the centre of every
panel, rather than retrained for each knob — it is one configuration, and
training it a dozen times would only measure seed noise a dozen times.

Models without a `<Model>_best.json` are skipped with a warning. Drop the file
in and re-run; nothing else needs changing (§10).

## 3. The grids

Grids live in **`run.py` flag space**, because that is the space the anchor
lives in: `<Model>_best.json` records `d_ff: 512`, not the `d_ff_mult: 4` that
Optuna actually drew. Each grid is the value set of the corresponding
`suggest_*` call in `tuning/search_spaces.py`; continuous knobs get an evenly
spaced grid across the searched range — logarithmic where the draw was
logarithmic.

Where a range was expressed **relative** to another knob, the grid is rebuilt
from the anchor's value of that other knob, so the sweep still covers the
searched ratios:

| Knob | Grid | Because the study drew |
|---|---|---|
| `d_ff` | `{1, 2, 4} × anchor d_model` | `d_ff_mult ∈ {1,2,4}` |
| PatchTST `stride` | `{patch_len/2, patch_len}` at the anchor's patch length | `patch_overlap ∈ {half, none}` |
| ModernTCN `patch_size` | `{1, 2} × anchor patch_stride` | `patch_size_mult ∈ {1,2}` |

### 3.1 The optimisation block

Swept by every model, in the same panel position in every figure — but no
longer over the same range everywhere, because the first round of these very
sweeps narrowed five of the spaces. The defaults:

| Knob | Grid | Searched range |
|---|---|---|
| `learning_rate` | 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2 | log-uniform 1e-4 … 5e-2 |
| `batch_size` | 16, 32, 64, 128 | the same four |
| `lradj` | `type1`, `type3`, `cosine` | the same three |

and the overrides, which mirror the bounds each model passes to
`search_spaces._optimisation` (`_OPTIMISATION` in `ofat_grids.py`):

| Model | Knob | Grid | Searched range |
|---|---|---|---|
| DLinear, AdaWaveNet | `learning_rate` | 1e-3, 3e-3, 1e-2, 3e-2, 7e-2 | log-uniform 1e-3 … 7e-2 |
| MSGNet, ModernTCN | `learning_rate` | 1e-3, 2e-3, 3e-3, 5e-3, 1e-2 | log-uniform 1e-3 … 1e-2 |
| MSGNet | `batch_size` | 16, 32, 64 | the same three |
| iTransformer | `batch_size` | 32, 64, 128 | the same three |

`lradj` is never overridden — no model narrows it.

`dropout` sweeps 0.0, 0.1, 0.2, 0.3 — the U(0, 0.3) the study drew from — with
three exceptions that again follow their spaces: TSLANet starts at 0.05, MSGNet
stops at 0.2 (0.0, 0.05, 0.1, 0.15, 0.2), and ModernTCN sweeps 0.2, 0.3, 0.4,
0.5, 0.6, its range having moved *up*. ModernTCN's `head_dropout` keeps the
shared 0.0 – 0.3, matching the one-sided narrowing in its space.

⚠️ **Anchors predating the narrowing.** §3.2 folds the anchor's own value into
every grid, so an anchor from a study run under the older, wider ranges still
appears on its panel while sitting outside the current grid — MSGNet's
`skip_channel 8`, ModernTCN's `patch_stride 2`. That is intended: a curve that
skipped its own centre could not be read against the anchor run. Re-tune before
reading such a panel as a statement about the current space.

### 3.2 The anchor is always on the grid

Each knob's grid gets the anchor's own value folded in, so every curve passes
through the tuned configuration. A grid point that lands **almost** on the
anchor — Optuna's 0.00114 beside the grid's 0.001 — is replaced by it rather
than trained separately: within a fifth of a decade on a log knob, or 5% of the
range on a linear one, the two configurations are indistinguishable and would
plot on top of each other. That trims 10 of 279 points.

### 3.3 Points per model

| Model | Knobs swept | Points (incl. anchor) | Trainings at `--itr 5` |
|---|---|---|---|
| DLinear | 4 | 15 | 75 |
| FITS | 4 | 17 | 85 |
| TSLANet | 7 | 21 | 105 |
| iTransformer | 9 | 24 | 120 |
| TimesNet | 9 | 26 | 130 |
| PatchTST | 11 | 28 | 140 |
| TimeMixer | 14 | 31 | 155 |
| ModernTCN | 12 | 33 | 165 |
| AdaWaveNet | 13 | 37 | 185 |
| MSGNet | 14 | 37 | 185 |
| **total (10 tuned models)** | | **269** | **1 345** |

Counted against the anchors currently in `tuning/ProjectC_tuning/` and the
narrowed spaces of §3.1. The narrowing is close to cost-neutral overall —
`iTransformer` and `ModernTCN` lose a point or two with their dropped
`batch_size`/`patch_stride` choices, `DLinear` and `FITS` gain a couple from
the longer `moving_avg` and evenly spaced `cut_freq` grids.

`--dry_run` prints the exact plan and every command without training anything.
`WFTNet` joins the table as soon as its `_best.json` file exists.

## 4. Coupled knobs and clamps

Some flags cannot move alone. `resolve()` re-applies exactly the rules
`tuning/search_spaces.py` applies, and **whatever else it had to move is
recorded in the `coupled` column of the results CSV** and printed under the
panel, so no curve hides a second change.

| Model | Rule |
|---|---|
| all with attention | `n_heads` is halved until it divides `d_model` |
| PatchTST | sweeping `patch_len` carries the stride with it, holding the anchor's overlap *regime* — otherwise `patch_len 8` with the anchor's `stride 24` would skip two thirds of the window, a point no trial ever saw |
| TimeMixer | `down_sampling_layers` is reduced until `window**layers` divides `seq_len`; sweeping `down_sampling_window` to 3 therefore also drops the pyramid to one level |
| TimeMixer | `decomp_method` switches which sub-knob is live, so the swept point hands the other one run.py's default (`top_k 5` / `moving_avg 25`) — both inside the searched grids |
| ModernTCN | sweeping `patch_stride` carries `patch_size` at the anchor's multiple; `large_size` is clipped to twice the patch count; `small_size ≤ large_size` |

A grid value that **clamps back onto the anchor** is dropped from the plan
rather than trained: it would retrain the anchor under a label claiming a value
the model never received. `--dry_run` shows which ones went.

## 5. What is recorded

Every point is trained by `run.py --itr 5`, and each seed's metrics are parsed
from its stdout into one CSV row:

| Column | Meaning |
|---|---|
| `val_loss` | the lowest validation loss of that run — the number the Optuna study was ranked on |
| `mse`, `mae` | the HAR-comparable test errors; on the `ln(RV)` scale under `--log` |
| `qlike` | QLIKE on the variance scale, with the lognormal back-transform |
| `mse_rv`, `mae_rv` | test errors after the back-transform, i.e. in RV units |
| `scale` | `ln_RV` or `raw_RV`, read off the metric block header |
| `coupled` | flags the clamps had to move alongside the swept one |

So the same sweep can be read on the criterion the search optimised
(`val_loss`) and on the ones the paper reports (`--metrics mse qlike …`).

## 6. Cost, resuming, failures

At `--itr 5` and 30 epochs the full sweep is ~1 345 trainings — the same order
as the tuning study itself (50 trials × 3 seeds per model). On a T4:
DLinear/FITS are minutes, PatchTST/TSLANet/iTransformer/ModernTCN/AdaWaveNet/
TimeMixer are a few hours each, MSGNet the better part of a day. Plan on more
than one session.

* **Check the plan before committing a session.** `--validate` builds every
  planned configuration through run.py's own parser and pushes one batch
  through it on CPU — about a minute for the whole plan — so a corner the
  architecture rejects surfaces before the sweep starts rather than at hour
  six. All 269 points of the current plan build and run.
* **Resumable.** Every completed point is appended to the CSV, and any
  (model, knob, value) already there is skipped. Re-running the same command
  after a disconnect continues where it stopped.
* **Cheaper passes.** `--itr 3` costs 40% less and still gives a spread;
  `--params learning_rate batch_size lradj` sweeps only the shared block;
  `--quick` (2 seeds, 5 epochs) is a smoke test, not a result.
* **Failures** are recorded in `<out>.failures.csv` and skipped on the next
  run, so one impossible corner does not block the sweep. `--retry_failed`
  re-runs them.
* **Checkpoints** go to `--checkpoint_dir` and are deleted after each point.
  Point it at local disk when `--out` is on Drive: they are rewritten every
  improving epoch.
* Keep one sweep to **one machine**. A CSV half-trained on CPU and half on GPU
  compares configurations across hardware, not against each other.

## 7. The figures

`python sensitivity/ofat_plots.py` writes into `sensitivity/figures/`, for each
metric in `--metrics` (default `mse qlike`):

**`ofat_<Model>_<metric>.png`** — one panel per knob.
* The **dashed line and grey band** are the anchor and its ±1 sd over seeds.
  A curve inside the band has not been shown to matter.
* **All panels share a y axis**, so a flat knob looks flat. A point too far off
  the scale to draw — a diverging learning rate, usually — is marked on the
  boundary with a hollow triangle and its value, instead of being allowed to
  flatten every other panel.
* The **star** is the anchor's own value of that knob; the small grey dots are
  the individual seeds.
* Ordered discrete choices sit at even spacing (16/32/64/128 on a linear axis
  would pile three points into the left quarter); continuous knobs keep a real
  numeric axis, logarithmic where the study drew logarithmically. Unordered
  choices (`lradj`, `activation`, …) are drawn as dots with intervals, not
  bars — the metric axis does not start at zero, which is precisely where a
  bar's length misleads.

**`ofat_tornado_<metric>.png`** — one panel per model, knobs ranked by how much
they move the metric, with the seed-noise level marked. Bars that do not clear
it are hatched: *no effect demonstrated*, which is a different statement from
*small effect*.

**`ofat_heatmap_<metric>.png`** — the same numbers as one models × knobs map,
so a knob that matters everywhere is visible at a glance.

## 8. The tables

`ofat_summary_<metric>.csv`, one row per model and knob:

| Column | Meaning |
|---|---|
| `span_pct` | (max − min) over the knob's grid, as % of the anchor — the size of the effect the sweep demonstrates |
| `gain_pct` | how much better than the anchor the best grid value is, as % of the anchor. **0 means the tuned value won its own sweep**; a positive value that clears `noise_pct` means the search left something on the table along that axis |
| `noise_pct` | 1 sd of the anchor over its seeds, same units — the reference the other two must be read against |
| `above_noise` | `span_pct > noise_pct` |
| `best_value`, `anchor_value` | the argmin of the sweep, and where the study put it |

`ofat_points_<metric>.csv` is every swept point with its mean, sd and seed
count — the appendix table.

## 9. Google Colab

`sensitivity/colab_ofat.ipynb` mirrors `tuning/colab_tune.ipynb`: mount Drive,
clone, install, smoke test, then one call per model so progress is saved as it
goes. Results and figures land in the same Drive folder as the tuning study, so
the anchors and the sweeps stay together.

## 10. Adding a model

Drop its `<Model>_best.json` (the file `tuning/optuna_tune.py` writes) into
`--results_dir` and re-run. The runner picks it up, reads the anchor and the
base command out of it, and builds the grid from `ofat_grids.py` — where all
eleven models already have one, including `TimesNet` and `WFTNet`. Nothing else
needs editing.
