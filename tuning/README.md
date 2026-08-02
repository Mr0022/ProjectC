# Optuna hyper-parameter search — 11 deep models, `seq_len = 96`, `pred_len = 1`

Search spaces and an Optuna driver for the eleven long-term forecasters in
`models/`, calibrated for the shipped EUR/USD realized-variance series.

```bash
python tuning/optuna_tune.py --model DLinear                       # one model, 50 trials
python tuning/optuna_tune.py --model all --retrain_best            # all eleven, then --itr 5
```

On Colab, open `tuning/colab_tune.ipynb` (see §7).

Run from the repository root (`exp_basic` discovers models by scanning the
relative path `models/`). Studies live in `tuning/results/optuna.db`; re-running
the same command resumes rather than restarts.

---

## 1. The protocol (fixed, never searched)

| Setting | Value | Note |
|---|---|---|
| `seq_len` | **96** | look-back window, ~4.5 months of business days |
| `pred_len` | **1** | one-step-ahead |
| `label_len` | 48 | decoder start token; none of the eleven has a real decoder, so it only sizes the unused `dec_inp` |
| `features` / `enc_in` | `S` / 1 | univariate `RV` |
| `train_epochs` / `patience` | 30 / 5 | early stopping, not epoch count, controls capacity |
| loss / selection | MSE on validation | test split is never read during a study |

With `--aggregate_mean` (on by default in the driver) at `h = 1` the target is
`Y^(1) = RV_{t+1}`, i.e. the HAR-RV target at horizon 1, and the loss is
computed on the `ln(RV)` scale rather than in z-space — so the validation
numbers are comparable across models *and* against `HAR-RV_RUN.PY --log`.

Splits are chronological and fixed by the loader: train ≤ 2022, validation
2023–2024 (519 windows), test 2025+ (402 windows).

## 2. Shared across all eleven

Identical for all eleven — no model gets a range another one lacks, so the
resulting table compares architectures rather than search effort.

| Parameter | Range | Why |
|---|---|---|
| `learning_rate` | log-uniform `1e-4 … 5e-2` | the UNION of what the family needs. The top of the range exists for DLinear/FITS (a few hundred parameters, much larger steps); the deep models learn within a few trials that it diverges and TPE stops proposing it |
| `batch_size` | {16, 32, 64, 128} | capped at 128: the train/val loaders use `drop_last=True` and validation holds only 519 windows |
| `lradj` | {`type1`, `type3`, `cosine`} | not cosmetic — `type1` *halves* the LR every epoch (≈8 useful epochs whatever the budget), `type3` holds 3 epochs then decays 0.9×, `cosine` anneals over the full budget. Strongly coupled to `learning_rate`, so searched jointly |

`d_ff` is always sampled as a **multiple** of `d_model` (`d_ff_mult ∈ {1,2,4}`)
rather than independently — otherwise a large part of the grid is spent on
`d_ff < d_model`, and the ratio is the quantity that transfers across widths.

## 3. Per-model spaces

Only knobs each model actually reads are included. Everything below maps to a
`run.py` flag, so any result is reproducible verbatim.

### 1. DLinear — 1 architectural knob
| Parameter | Range | Note |
|---|---|---|
| `moving_avg` | {5, 13, 25, 49} | trend/seasonal split. **Must be odd** — the block pads `(k-1)//2` per side, an even kernel returns `L-1` and the residual fails to broadcast. Values ≈ week / half-month / month / quarter |

No `d_model`, no `dropout`: DLinear reads neither, and `individual` is a
constructor argument it never takes from the config.

### 2. PatchTST
| Parameter | Range |
|---|---|
| `d_model` | {16, 32, 64, 128} |
| `d_ff_mult` | {1, 2, 4} |
| `n_heads` | {2, 4, 8} |
| `e_layers` | {1, 2, 3} |
| `patch_len` | {8, 16, 24} |
| `patch_overlap` | {half, none} → `stride = patch_len/2` or `patch_len` |
| `dropout` | 0.0 – 0.3 |
| `activation` | {gelu, relu} |

Patch count is `(96 − patch_len)/stride + 2` and is also the flatten-head
width: 24 patches at `patch_len 8 / stride 4`, 5 at `24 / 24`. `factor` is
excluded — it only parameterises `ProbAttention`, and PatchTST uses
`FullAttention`.

### 3. iTransformer
| Parameter | Range |
|---|---|
| `d_model` | {32, 64, 128, 256} |
| `d_ff_mult` | {1, 2, 4} |
| `n_heads` | {2, 4, 8} |
| `e_layers` | {1, 2, 3} |
| `dropout` | 0.0 – 0.3 |
| `activation` | {gelu, relu} |

⚠️ **Univariate caveat.** iTransformer's tokens *are* the variates, so with
`enc_in = 1` the encoder holds one token and self-attention is a no-op up to
the projections. All capacity sits in `Linear(96 → d_model)` and the FFN —
hence the widest `d_model` range of the group. `n_heads` will look inert here.

### 4. TimesNet
| Parameter | Range | Note |
|---|---|---|
| `d_model` | {16, 32, 64} | tightest ranges in the file — cost is `e_layers × top_k × num_kernels × d_model × d_ff` |
| `d_ff_mult` | {1, 2, 4} | |
| `e_layers` | {1, 2, 3} | |
| `top_k` | {2, 3, 5} | dominant FFT periods of the 97-step series; beyond ~5 they are spectral noise, each costing a full Inception pass |
| `num_kernels` | {3, 4, 6} | parallel Inception kernel sizes |
| `dropout` | 0.0 – 0.3 | |

### 5. MSGNet
| Parameter | Range |
|---|---|
| `d_model` | {16, 32, 64} |
| `d_ff_mult` | {1, 2, 4} |
| `e_layers` | {1, 2} |
| `n_heads` | {2, 4, 8} |
| `top_k` | {2, 3, 5} |
| `node_dim` | {5, 10, 20} |
| `gcn_depth` | {1, 2, 3} |
| `propalpha` | 0.05 – 0.5 |
| `conv_channel` | {8, 16, 32} |
| `skip_channel` | {8, 16, 32} |
| `dropout` | 0.0 – 0.3 |

⚠️ The graph half (`node_dim`, `propalpha`, and to a degree `gcn_depth`) builds
an adjacency over `c_out` = **1** node, whose softmax is identically 1 — it
carries no information on a univariate series. `gcn_depth` and the channel
widths still size the `mixprop` MLP, so they stay in the space with small
ranges and no expectation of a large effect. The scale half (`top_k`,
`d_model`, `e_layers`) is what works here.

### 6. TimeMixer
| Parameter | Range |
|---|---|
| `d_model` | {16, 32, 64} |
| `d_ff_mult` | {1, 2, 4} |
| `e_layers` | {1, 2, 3} |
| `down_sampling_window` | {2, 3, 4} |
| `down_sampling_layers` | {1, 2, 3} → clamped, see below |
| `down_sampling_method` | {avg, max, conv} |
| `channel_independence` | {0, 1} |
| `use_norm` | {0, 1} |
| `decomp_method` | {moving_avg, dft_decomp} |
| ↳ `moving_avg` | {13, 25, 49} — only when `decomp_method = moving_avg` |
| ↳ `top_k` | {2, 3, 5} — only when `decomp_method = dft_decomp` |
| `dropout` | 0.0 – 0.3 |

**Hard constraint (verified by construction over all 54 combinations):**
`down_sampling_window ** down_sampling_layers` must **divide** `seq_len`. The
model sizes its inter-level maps as `seq_len // window**i` while pooling
produces the floor of the previous length, and the two diverge on any
remainder — window 3 at 2 layers wants 10 steps but gets 10 (pool) or 11
(conv, which pads), and mixing dies on the mismatch. Surviving pyramids:

```
window 2 → 96, 48, 24, 12   (1–3 layers)
window 3 → 96, 32           (1 layer;   9 ∤ 96)
window 4 → 96, 24, 6        (1–2 layers; 64 ∤ 96)
```

`run.py` separately rejects 0 layers (a one-level pyramid has nothing to mix).

### 7. FITS — 1 architectural knob
| Parameter | Range | Note |
|---|---|---|
| `cut_freq` | int 3 – 49 | retained rFFT bins of the 96-step window; 49 = `96/2+1` is the hard ceiling (the model clamps beyond it) |

Bin `b` ↔ period `96/b` days: `cut_freq 5` keeps everything slower than ~19
days, `24` everything slower than 4 days, `49` keeps all of it. Since the model
*is* `Linear(cut_freq → cut_freq·97/96)` in complex space, `cut_freq` is both
the bandwidth and the parameter count — hence a dense integer range rather
than a coarse grid.

### 8. WFTNet
| Parameter | Range | Note |
|---|---|---|
| `d_model` | {16, 32, 64} | TimesNet's Inception backbone plus a CWT branch |
| `d_ff_mult` | {1, 2, 4} | |
| `e_layers` | {1, 2} | |
| `top_k` | {2, 3, 5} | |
| `num_kernels` | {3, 4, 6} | |
| `wavelet_scale` | {3, 4, 5, 6} | Morlet scales are `2**linspace(-1, scale, 8)`: scale 3 spans ~0.7–8 days, scale 6 spans ~0.7–64. The *count* (8) is hard-wired by the `(8,1)` convolution; only the span is tunable |
| `period_coeff` | 0.1 – 0.9 | wavelet-vs-Fourier weight — the knob most worth trials. Volatility is bursty rather than globally periodic, so expect an optimum below 0.5 |
| `dropout` | 0.0 – 0.3 | |

### 9. TSLANet — reads exactly four fields
| Parameter | Range | Note |
|---|---|---|
| `d_model` | {32, 64, 128} | patch embedding width |
| `e_layers` | {1, 2, 3} | spectral + ICB blocks |
| `patch_size` | {8, 16, 24, 32} | stride is fixed internally at `patch_size/2` → 23 patches at 8, 5 at 32 |
| `dropout` | 0.05 – 0.3 | doubles as the ceiling of the drop-path schedule, so the floor is kept above 0 |

The head is `Linear(d_model × num_patches → 1)`, so `d_model` and `patch_size`
jointly dominate the parameter count.

### 10. ModernTCN
| Parameter | Range | Note |
|---|---|---|
| `d_model` | {16, 32, 64} | |
| `ffn_ratio` | {1, 2, 4} | |
| `num_blocks` | {1, 2, 3} | depth *within* the single stage |
| `large_size` | {13, 21, 31, 51} | large depth-wise kernel — the point of the model |
| `small_size` | {3, 5, 7} | re-param kernel |
| `patch_stride` | {2, 4, 8, 16} | |
| `patch_size` | `patch_stride × {1, 2}` | |
| `dropout`, `head_dropout` | 0.0 – 0.3 each | |

Three constraints, all handled by clamping:
* `patch_stride` must divide 96 — the head is sized `d_model × (96/stride)`
  while the stem produces `floor((96−stride)/stride)+1` patches.
* `large_size`/`small_size` must be **odd** (kernel//2 padding, otherwise the
  `[B,M,D,N]` reshape breaks) and `small_size ≤ large_size` (asserted by
  `ReparamLargeKernelConv`).
* `large_size` is clipped to `2 × patch_count`: a 51-tap kernel over the 6
  patches that `stride 16` produces is 90 % padding. Full range at `stride 2`.

`num_blocks` stays one element long: with `use_multi_scale` at its default the
head is built for the pre-downsampling patch count, so a second stage would
halve the feature axis and mismatch it.

### 11. AdaWaveNet
| Parameter | Range | Note |
|---|---|---|
| `d_model` | {16, 32, 64} | |
| `d_ff_mult` | {1, 2, 4} | |
| `e_layers` | {1, 2} | |
| `n_heads` | {2, 4, 8} | |
| `lifting_levels` | {1, 2, 3} | each level halves the window: 96 → 48 → 24 → 12. 96 = 2⁵·3 tolerates 5, but 3 is the deepest leaving usable resolution |
| `lifting_kernel_size` | {3, 5, 7, 9} | *may* be even — the scheme pads asymmetrically `(k//2, k−1−k//2)`, unlike every other conv here |
| `regu_details` | log 1e-3 – 1e-1 | penalises detail-coefficient magnitude |
| `regu_approx` | log 1e-3 – 1e-1 | penalises approximation-branch drift; together they stop the learned wavelet collapsing to an arbitrary invertible map. Log scale because the effect is multiplicative |
| `dropout` | 0.0 – 0.3 | |
| `activation` | {gelu, relu} | |

`n_clusters` is excluded — the model clamps it to `min(n_clusters, enc_in) = 1`.
`sr_ratio` (super-resolution only) and `factor` (ProbAttention only) likewise.

## 4. Trial budget

**50 trials for every model** — an equal budget, so the table compares
architectures rather than how long each search ran. TPE spends the first 10 on
random startup, leaving 40 model-guided ones.

| Model | Discrete grid | Continuous | Trials |
|---|---|---|---|
| DLinear | 48 | lr | 50 |
| TSLANet | 432 | lr, dropout | 50 |
| FITS | 564 | lr | 50 |
| iTransformer | 2,592 | lr, dropout | 50 |
| TimesNet | 2,916 | lr, dropout | 50 |
| WFTNet | 7,776 | lr, dropout, period_coeff | 50 |
| PatchTST | 15,552 | lr, dropout | 50 |
| AdaWaveNet | 15,552 | lr, dropout, regu_details, regu_approx | 50 |
| ModernTCN | 31,104 | lr, dropout, head_dropout | 50 |
| MSGNet | 157,464 | lr, dropout, propalpha | 50 |
| TimeMixer | 209,952 (139,968 after the divisibility clamp) | lr, dropout | 50 |

Grid sizes include the shared `batch_size` × `lradj` factor of 12. The coverage
50 trials buys is therefore very uneven — near-exhaustive for DLinear, a thin
sample for MSGNet and TimeMixer. That is the price of an equal protocol, and
it is the right price to pay for a comparison; the alternative biases the
table towards whichever model was searched hardest.

## 4b. The final run: `--itr 5`

`--retrain_best` re-runs the winning configuration through `run.py` with
`--itr 5`, i.e. five independent seeds (2021–2025). `run.py` reseeds *before*
each model is built, so a repeat is defined entirely by its own seed, and
`summarize_runs` then prints mean ± std, min and max for every HAR-comparable
metric:

```
  MEAN OVER 5 RUNS   seeds 2021-2025
  MSE [ln]   : 0.282285 +/- 0.003725   [min 0.277683, max 0.285736]
  QLIKE [RV] : 0.164766 +/- 0.003517   [min 0.159301, max 0.168441]
  MSE_RV     : 0.042002 +/- 0.000862   [min 0.041038, max 0.043336]
```

That spread is the initialisation noise of the configuration and belongs in
the benchmark table — a single run sits closer to a best case than to a mean.

Order of cost per trial (cheapest first): FITS ≈ DLinear ≪ TSLANet <
PatchTST ≈ iTransformer ≈ ModernTCN < TimeMixer ≈ AdaWaveNet < MSGNet <
WFTNet ≈ TimesNet.

## 5. Driver behaviour worth knowing

* **Selection metric** — the *minimum* validation loss over epochs, i.e. the
  loss of the checkpoint `EarlyStopping` keeps. Not the last epoch's, which
  would reward configurations that happen to stop cleanly.
* **Pruning** — `MedianPruner` (10 startup trials, 5 warm-up epochs). The
  driver observes the epoch curve by swapping the `EarlyStopping` name that
  `exp_long_term_forecasting` resolves for a wrapper around the real class;
  nothing under `exp/` is modified, and the stopping rule is unchanged.
* **Reproducibility** — every trial is built through `run.py`'s own parser, and
  the winning command line is written into `tuning/results/<Model>_best.json`.
  Verified: re-running an emitted command reproduces the trial's validation
  loss to the last digit.
* **Noise** — 519 validation windows is a small sample; differences below
  ~1 % of the loss are seed noise. Use `--n_seeds 2` or `3` to average each
  configuration over seeds when the ranking matters (cost scales linearly).
* **Failures are pruned, not fatal** — a config that hits a shape or assertion
  error is recorded (`trial.user_attrs['error']`) and pruned so the study
  continues. All eleven spaces were validated by building and forward-passing
  40 sampled configurations each (440 total) — none currently fail — so a
  populated `error` attribute means something changed in `models/`.
* **Disk** — trial checkpoints go to `tuning/results/_checkpoints/` and are
  deleted after each trial. `tuning/results/` is already git-ignored.

## 6. Google Colab

`tuning/colab_tune.ipynb` — open it from Colab (File → Open notebook → GitHub,
or upload it). It mounts Drive, clones this private repo with a token you
enter at the prompt, installs the handful of packages Colab lacks (`ptwt`,
`fast_pytorch_kmeans`, `reformer-pytorch`, `local-attention`, `optuna`), runs
the 50-trial search per model, then the `--itr 5` final runs, and prints a
ranked summary table.

Studies live in `optuna.db` on Drive, so a disconnected session is resumed by
re-running the setup cells and the search cell — a model that already has its
50 trials returns immediately. Checkpoints are pointed at local disk with
`--checkpoint_dir /content/_ckpt`; they are rewritten every improving epoch,
and on a Drive mount that would dominate the runtime.

## 7. Extending to multivariate (`--features M`)

Three knobs deliberately excluded because `enc_in = 1` makes them no-ops:
`individual` (DLinear takes it as a constructor argument, FITS and ModernTCN
from the config), AdaWaveNet's `n_clusters`, and MSGNet's graph half becomes
meaningful. Add them to the relevant spaces in `search_spaces.py` and pass
`--features M --enc_in <n>` to the driver.
