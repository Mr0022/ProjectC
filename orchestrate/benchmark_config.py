#!/usr/bin/env python3
"""The benchmark grid, the tuned anchors and the loss definitions.

Single source of truth for ``run_benchmark.py`` (which trains and forecasts)
and ``aggregate_results.py`` (which scores and tabulates). Nothing here trains
anything, and nothing here imports torch, so the scoring half runs in a bare
pandas/numpy environment.

The grid
--------
    10 datasets  x  3 horizons  x  (10 deep models + HAR-RV)

    forex   EURUSD  AUDUSD  EURGBP  USDCHF  USDJPY     (data/<NAME>-RV.csv)
    crypto  btcusdt ethusdt adausdt bnbusdt xrpusdt    (data/<name>-RV.csv)
    h       1  5  22

Every cell is run under the SAME protocol the hyper-parameter search used:
``--aggregate_mean``, ``seq_len 96``, univariate ``RV``. The model forecasts
one number, the h-day forward mean of RV, which is exactly HAR-RV's target
Y^(h) -- so a deep model and the baseline are scored on identical rows against
identical actuals.

The SCALE is whatever the anchor was tuned on: ``--log`` in its command line
means the cell models ln(mean RV), its absence means the mean RV itself. Both
are supported end to end -- the target, the Jensen correction (log only), the
losses and HAR-RV's own invocation all follow the anchor -- and a sweep must
be on one scale, which run_benchmark checks. exp() of the log target is the raw
target exactly, so the two are still comparable on the variance-scale losses.

Hyper-parameters
----------------
Taken verbatim from the Optuna winners in ``tuning/ProjectC_tuning`` --
``<Model>_best.json`` carries both the winning parameters and the full command
line that produced them, and the command line is what is reused. Everything the
orchestrator does not own (architecture, learning rate, batch size, schedule,
``--aggregate_mean``, ``--log``) is passed through unchanged; the owned flags
are listed in ``ORCHESTRATOR_FLAGS`` below.

Those studies were run on EUR/USD at h = 1 only, so applying them to the other
nine datasets and to h = 5 / 22 is a transfer, not a per-cell tuning. It is the
honest protocol for a cross-dataset comparison -- every model gets the same
treatment -- but the transfer should be stated wherever the table is reported. Nothing about it is baked in: drop a
``<anchor_dir>/<dataset>/<Model>_best.json`` in place and that dataset picks up
its own anchor automatically (see ``anchor_path``).

Horizon transfer is safe by construction. Under ``--aggregate_mean`` the head
is always built with ``pred_len = 1`` (Exp_Long_Term_Forecast._build_model), so
changing h changes the target's aggregation window and nothing about the
architecture -- no shape constraint of any model is a function of h.
"""

import glob
import json
import os
import sys
from collections import OrderedDict, namedtuple

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_provider.splits import (SPLITS, har_bounds, month_mask,  # noqa: E402
                                  month_split_borders)

# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------

DatasetSpec = namedtuple('DatasetSpec', ['name', 'data_path', 'asset'])

FOREX_DATASETS = ('EURUSD', 'AUDUSD', 'EURGBP', 'USDCHF', 'USDJPY')
CRYPTO_DATASETS = ('btcusdt', 'ethusdt', 'adausdt', 'bnbusdt', 'xrpusdt')

DATASETS = OrderedDict()
for _name in FOREX_DATASETS:
    DATASETS[_name] = DatasetSpec(_name, f'{_name}-RV.csv', 'forex')
for _name in CRYPTO_DATASETS:
    DATASETS[_name] = DatasetSpec(_name, f'{_name}-RV.csv', 'crypto')

# The ten deep models, in the order the tuning README lists them.
DEEP_MODELS = ('DLinear', 'PatchTST', 'iTransformer', 'TimesNet', 'MSGNet',
               'TimeMixer', 'FITS', 'TSLANet', 'ModernTCN', 'AdaWaveNet')

# The baseline. Not a torch model and not trained through run.py -- it is
# fitted by HAR-RV_RUN.PY, whose forecasts are folded into the same per-run
# files so the scoring code never has to special-case it.
HAR_MODEL = 'HAR-RV'
MODELS = DEEP_MODELS + (HAR_MODEL,)

HORIZONS = (1, 5, 22)

# The protocol. seq_len is fixed by the tuning study; it is read back from each
# anchor and checked rather than assumed, because the test row count depends on
# it (see test_target_dates).
SEQ_LEN = 96
DEFAULT_SEED = 2021
# Repeats per cell. Ten is what a benchmark table needs: the initialisation
# spread on these splits is comparable to the gap between neighbouring models,
# so a single run reports a draw rather than a model. Seeds follow run.py's
# rule -- repeat i uses (--seed + i) -- so 10 x DEFAULT_SEED means 2021..2030.
DEFAULT_N_SEEDS = 10
DEFAULT_TRAIN_EPOCHS = 30
DEFAULT_PATIENCE = 7

# Same fraction HAR-RV_RUN.PY and Exp_Long_Term_Forecast use. Under --log every
# forecast is exp(.) > 0, so the floor never binds; it is applied anyway so the
# QLIKE written here is the same function of (forecast, actual) as theirs. On
# the raw scale it does bind -- an unconstrained head can forecast a negative
# variance -- and metrics.csv counts how often (`n_floored`).
QLIKE_FLOOR_FRAC = 1e-4

# The modelling scale of a cell, taken from whether its tuned command line
# carries --log. It is a property of the ANCHOR, not a flag of the sweep: a
# model has to be scored on the scale it was tuned and trained on, and mixing
# the two within one comparison is what the check in run_benchmark prevents.
LN_SCALE = 'ln_RV'
RAW_SCALE = 'raw_RV'


def is_log(scale):
    return scale == LN_SCALE


TARGET_COL = 'RV'
DATE_COL = 'date'


def seed_list(base=DEFAULT_SEED, n_seeds=DEFAULT_N_SEEDS):
    """The seeds of an --itr n_seeds sweep: base, base+1, ... base+n-1.

    run.py's own rule, so `--itr N --seed S` there and the N cells this sweep
    trains are the same N runs -- it simply shards them into one process each,
    which is what makes an individual repeat resumable and separately scorable.
    """
    return [base + i for i in range(n_seeds)]


def dataset_names(assets=None):
    """Dataset names, optionally restricted to one or more asset classes."""
    if not assets:
        return list(DATASETS)
    return [n for n, s in DATASETS.items() if s.asset in set(assets)]


def dataset_spec(name):
    if name not in DATASETS:
        raise KeyError(f"unknown dataset '{name}'; known: {', '.join(DATASETS)}")
    return DATASETS[name]


# ---------------------------------------------------------------------------
# The row grid, shared by both model families
# ---------------------------------------------------------------------------

def read_rv_series(spec, root_path=None):
    """The dataset as both families see it: sorted, complete, strictly positive.

    Dataset_Custom drops NaN and (under --log) non-positive rows BEFORE it
    computes the split borders, and HAR-RV_RUN.PY drops exactly the same rows
    before it builds its regressors. Reproducing that here -- once -- is what
    lets this module state which dates a cell forecasts without loading either.
    """
    root_path = root_path or os.path.join(REPO_ROOT, 'data')
    df = pd.read_csv(os.path.join(root_path, spec.data_path))
    df = df.dropna(subset=[DATE_COL, TARGET_COL]).reset_index(drop=True)
    df = df[df[TARGET_COL] > 0].reset_index(drop=True)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    if not df[DATE_COL].is_monotonic_increasing:
        raise ValueError(
            f'{spec.data_path}: rows are not sorted by date. Both the deep '
            f'models and HAR-RV assume chronological order.')
    return df


def test_target_dates(spec, h, seq_len=SEQ_LEN, root_path=None):
    """The dates a cell forecasts, in the order the test loader emits them.

    Window i of the test split reads rows [b1+i, b1+i+seq_len) and targets rows
    [b1+i+seq_len, b1+i+seq_len+h), so the forecast is indexed by the FIRST row
    of its target window -- the same index HAR-RV gives Y^(h), whose window
    spans t .. t+h-1 from regressors that stop at t-1. That shared convention
    is what makes the two families' forecasts comparable row for row, and this
    function is where it is written down.

    The count is (rows in the test window) - h + 1: the loader cannot enumerate
    a window whose target runs past the end of the split, and HAR-RV's embargo
    (horizon_month_mask) removes exactly the same rows.

    Returned as a DatetimeIndex so every forecast, actual and loss this
    benchmark writes carries a date -- the alignment key the DM and MCS tests
    need across models.
    """
    df = read_rv_series(spec, root_path)
    border1s, border2s = month_split_borders(
        df[DATE_COL], seq_len, len(df), SPLITS[spec.asset])
    b1, b2 = border1s[2], border2s[2]
    n_windows = (b2 - b1) - seq_len - h + 1
    if n_windows < 1:
        raise ValueError(
            f'{spec.name}: the test split holds {b2 - b1} row(s), too few for '
            f'one seq_len={seq_len} + h={h} window.')
    start = b1 + seq_len
    return pd.DatetimeIndex(df[DATE_COL].iloc[start:start + n_windows].values,
                            name=DATE_COL)


def horizon_target(spec, h, log_mode=True, seq_len=SEQ_LEN, root_path=None):
    """The actuals a cell is scored against.

        --log     Y^(h) = ln( mean RV over t..t+h-1 )
        raw       Y^(h) =     mean RV over t..t+h-1

    exp() of the first is the second exactly -- the log sits outside the mean
    on both sides -- which is what lets a raw-scale run and a --log run be
    compared on the variance-scale losses at every horizon.

    Derived here, in float64, from the CSV -- independently of both families.
    HAR-RV builds it as ``rolling(h).mean().shift(-(h-1))``, logged under
    ``--log``; Exp_Long_Term_Forecast builds it as the mean over the future
    window, or ``logsumexp(ln RV) - ln(h)``, after undoing the loader's
    standardisation. Those are the same quantity by algebra, and this is the
    third computation that says so: ``aggregate_results.py`` checks every
    stored cell against it, which is what turns "the two families forecast the
    same target" from a claim into a test.

    Scoring against one reference series rather than each cell's own copy also
    keeps the loss differentials clean: a deep model's actuals have been
    through a float32 standardise/unstandardise round trip and agree with the
    CSV only to ~1e-7, which is harmless but need not be carried into a DM
    statistic.

    Indexed, like everything else here, by the date the target window opens.
    """
    df = read_rv_series(spec, root_path)
    forward = df[TARGET_COL].rolling(h).mean().shift(-(h - 1))
    values = np.log(forward.values) if log_mode else forward.values
    series = pd.Series(values, name='y_ln' if log_mode else 'y_rv',
                       index=pd.DatetimeIndex(df[DATE_COL].values, name=DATE_COL))
    return series.reindex(test_target_dates(spec, h, seq_len, root_path))


# How far a stored cell's own actuals may sit from horizon_target before it is
# reported. The deep models' copy round-trips through a float32 StandardScaler,
# which costs ~1e-7 in ln(RV); a cell scored on the wrong rows would be off by
# orders of magnitude more.
TARGET_TOL = 1e-4


def qlike_floor(spec, root_path=None):
    """1e-4 * mean training RV -- the floor for non-positive forecasts.

    Measured over HAR-RV's estimation window (train + val), which is what
    HAR-RV_RUN.PY does. Exp_Long_Term_Forecast measures it over the train split
    alone, so the two differ slightly; under --log neither ever binds, because
    a log-scale forecast back-transforms to exp(.) > 0. Reported so that stays
    checkable rather than assumed.
    """
    df = read_rv_series(spec, root_path)
    train_start, train_end, _, _ = har_bounds(SPLITS[spec.asset])
    inside = month_mask(df[DATE_COL], train_start, train_end)
    return QLIKE_FLOOR_FRAC * float(df.loc[inside, TARGET_COL].mean())


# ---------------------------------------------------------------------------
# Losses -- one definition, used for every model
# ---------------------------------------------------------------------------

def per_obs_losses(pred, pred_rv, target, floor, log_mode=True):
    """Per-observation losses for one cell, as flat arrays of equal length.

    Reproduces utils.metrics.QLIKE and HAR-RV_RUN.PY's mse/mae/qlike exactly,
    but keeps the terms instead of averaging them -- the per-observation series
    is what Diebold-Mariano and the MCS consume, and the mean of each column is
    the headline metric, so the table and the tests cannot disagree.

    `pred` and `target` are on the MODELLING scale, `pred_rv` is the forecast
    as a variance -- the Jensen back-transform under --log, the forecast itself
    in raw mode.

        se_ln / ae_ln  squared / absolute error in ln(RV). LOG MODE ONLY: in
                       raw mode a forecast can be <= 0, where the log does not
                       exist, so these keys are simply absent
        se_rv / ae_rv  the same errors on the variance scale. Comparable
                       between a raw run and a --log one -- exp() of the log
                       target is the raw target exactly, at every horizon
        qlike          RV/RV_hat - ln(RV/RV_hat) - 1  (Patton, 2011)

    Two rules for the non-positive forecasts only a raw-scale model can
    produce, both taken from HAR-RV_RUN.PY's raw branch and
    Exp_Long_Term_Forecast.test() so the three agree number for number:

      * MSE/MAE are measured against max(forecast, 0) -- a negative variance is
        not admissible, so zero is the best feasible prediction the model could
        have issued;
      * QLIKE is infinite at zero and floors them at `floor` instead.

    Both are inert under --log, where the forecast is exp(.) > 0.

    QLIKE is asymmetric in its argument order: the ratio is actual/forecast,
    and inverting it is a different loss, not a rescaling of the same one.
    """
    pred = np.asarray(pred, dtype=float).reshape(-1)
    pred_rv = np.asarray(pred_rv, dtype=float).reshape(-1)
    target = np.asarray(target, dtype=float).reshape(-1)
    true_rv = np.exp(target) if log_mode else target

    losses = {}
    if log_mode:
        err = target - pred
        losses['se_ln'] = err ** 2
        losses['ae_ln'] = np.abs(err)

    err_rv = true_rv - np.maximum(pred_rv, 0.0)
    ratio = true_rv / np.where(pred_rv <= 0, floor, pred_rv)
    losses['qlike'] = ratio - np.log(ratio) - 1.0
    losses['se_rv'] = err_rv ** 2
    losses['ae_rv'] = np.abs(err_rv)
    return losses


# Loss column -> the metric its mean is reported as. A raw-scale cell has no
# ln-scale losses, so those two metrics stay empty for it; the columns are kept
# in the table either way, so a raw run and a --log run share one schema and can
# be compared on the RV rows.
LOSS_TO_METRIC = OrderedDict([('se_ln', 'MSE_ln'), ('ae_ln', 'MAE_ln'),
                              ('qlike', 'QLIKE'),
                              ('se_rv', 'MSE_RV'), ('ae_rv', 'MAE_RV')])


def summarize_losses(losses):
    """{loss column: array} -> {metric name: mean}, skipping absent losses."""
    return {metric: float(np.mean(losses[key]))
            for key, metric in LOSS_TO_METRIC.items() if key in losses}


# ---------------------------------------------------------------------------
# Tuned anchors
# ---------------------------------------------------------------------------

# Flags the orchestrator sets itself. Whatever the tuning command said about
# them is dropped, because they are what a benchmark cell varies (dataset,
# horizon) or what it standardises (naming, seeds, budget, checkpoints).
# Everything else in that command line survives verbatim.
ORCHESTRATOR_FLAGS = frozenset({
    'model_id', 'des', 'itr', 'seed', 'checkpoints', 'num_workers',
    'data', 'root_path', 'data_path', 'pred_len',
    'train_epochs', 'patience',
})

DEFAULT_ANCHOR_DIR = os.path.join(REPO_ROOT, 'tuning', 'ProjectC_tuning')


def _is_number(text):
    try:
        float(text)
    except ValueError:
        return False
    return True


def parse_command(command):
    """``python -u run.py --a 1 --b`` -> ``{'a': ['1'], 'b': []}``.

    Bare switches map to an empty list and ``nargs='+'`` flags to several
    values, so the dict round-trips back to argv unchanged. A token counts as a
    flag only when it starts with ``--`` and is not a negative number, so a
    value of ``-1`` is never mistaken for one.
    """
    tokens = command.split()
    while tokens and not tokens[0].startswith('--'):
        tokens = tokens[1:]

    flags, current = OrderedDict(), None
    for token in tokens:
        if token.startswith('--') and not _is_number(token[2:]):
            current = token[2:]
            flags[current] = []
        elif current is not None:
            flags[current].append(token)
    return flags


def anchor_path(model, anchor_dir=None, dataset=None):
    """Where this model's tuned configuration lives.

    A per-dataset anchor wins over the shared one, so a later per-dataset
    study is picked up by dropping ``<anchor_dir>/<dataset>/<Model>_best.json``
    in place -- no change here and no flag to remember.
    """
    anchor_dir = anchor_dir or DEFAULT_ANCHOR_DIR
    if dataset:
        per_dataset = os.path.join(anchor_dir, dataset, f'{model}_best.json')
        if os.path.isfile(per_dataset):
            return per_dataset
    return os.path.join(anchor_dir, f'{model}_best.json')


def load_anchor(model, anchor_dir=None, dataset=None):
    """(flags, meta) for one model: the tuned command line minus the owned flags."""
    path = anchor_path(model, anchor_dir, dataset)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"no tuned configuration for {model}: expected {path}. Run "
            f"tuning/optuna_tune.py --model {model} first, or point "
            f"--anchor_dir at the directory holding <Model>_best.json.")
    with open(path) as fh:
        best = json.load(fh)

    flags = parse_command(best['command'])
    for flag in ORCHESTRATOR_FLAGS:
        flags.pop(flag, None)

    seq_len = int(flags.get('seq_len', [SEQ_LEN])[0])
    if seq_len != SEQ_LEN:
        raise ValueError(
            f'{path}: seq_len is {seq_len}, but the grid runs at '
            f'{SEQ_LEN}. One look-back for every model is what makes the table '
            f'a comparison of architectures; if the protocol really changed, '
            f'change SEQ_LEN in orchestrate/benchmark_config.py so the test '
            f'window is derived for the same value.')
    if 'aggregate_mean' not in flags:
        raise ValueError(
            f'{path}: the tuned command line is missing --aggregate_mean, so '
            f'its target is a pred_len-step window rather than the h-day mean '
            f'HAR-RV forecasts. The benchmark has nothing to compare it to.')
    return flags, {'path': path,
                   'scale': LN_SCALE if 'log' in flags else RAW_SCALE,
                   'best_val_loss': best.get('best_val_loss'),
                   'best_params': best.get('best_params', {})}


def discover_anchors(anchor_dir=None, models=None):
    """Deep models that have a tuned configuration, in DEEP_MODELS order."""
    anchor_dir = anchor_dir or DEFAULT_ANCHOR_DIR
    found = set()
    for path in glob.glob(os.path.join(anchor_dir, '*_best.json')):
        found.add(os.path.basename(path)[:-len('_best.json')])
    # `models=[]` means "no deep models", not "all of them" -- it is what
    # --models HAR-RV produces, and defaulting it to everything would train the
    # entire grid when only the baseline was asked for.
    wanted = set(DEEP_MODELS) if models is None else set(models)
    return [m for m in DEEP_MODELS if m in found and m in wanted]


def anchor_argv(model, spec, h, seed, anchor_dir=None, checkpoints=None,
                train_epochs=DEFAULT_TRAIN_EPOCHS, patience=DEFAULT_PATIENCE,
                num_workers=0, use_gpu=True, root_path='./data/'):
    """The run.py argv for one benchmark cell.

    The tuned flags come first and the owned ones last, so a cell is the tuned
    configuration plus exactly one dataset, one horizon and one seed.
    """
    flags, meta = load_anchor(model, anchor_dir, spec.name)
    argv = []
    for flag, values in flags.items():
        argv.append(f'--{flag}')
        argv.extend(values)
    argv += ['--data', spec.asset,
             '--root_path', root_path,
             '--data_path', spec.data_path,
             '--pred_len', str(h),
             '--model_id', cell_id(spec.name, h, model),
             '--des', 'bench',
             '--itr', '1',
             '--seed', str(seed),
             '--train_epochs', str(train_epochs),
             '--patience', str(patience),
             '--num_workers', str(num_workers)]
    if checkpoints:
        argv += ['--checkpoints', checkpoints]
    if not use_gpu:
        # Exp_Basic reads use_gpu directly, so a CPU-only box needs the flag
        # even though run.py's own device pick would have fallen back.
        argv.append('--no_use_gpu')
    return argv, meta


def cell_id(dataset, h, model):
    """Stable name for one cell; goes into --model_id and the checkpoint path."""
    return f'bench_{dataset}_h{h:02d}_{model}'


# ---------------------------------------------------------------------------
# Where a cell's forecasts are stored
# ---------------------------------------------------------------------------

DEFAULT_RESULTS_DIR = os.path.join(REPO_ROOT, 'orchestrate', 'results')


def cell_path(results_dir, dataset, h, model, seed=None):
    """Per-cell forecast file. HAR-RV is deterministic, so it carries no seed."""
    stem = model if seed is None else f'{model}_seed{seed}'
    return os.path.join(results_dir, 'runs', dataset, f'h{h:02d}', f'{stem}.npz')


def save_cell(path, pred, pred_rv, true, dates, meta):
    """Write one cell's forecasts.

    `pred` and `true` are on the modelling scale (ln(RV) under --log, RV
    otherwise); `pred_rv` is the forecast as a variance -- the Jensen
    back-transform under --log, and `pred` itself in raw mode.

    Both scales are stored rather than one plus a rule for recovering the
    other: the modelling-scale forecast is what the model produced, the
    variance forecast is what QLIKE is defined on, and the Jensen terms that
    connect them are part of the fitted model. Storing all three means the
    scoring code applies no model-specific correction of its own -- and
    `meta['scale']` says which is which.

    Raw-scale forecasts are stored UNCLIPPED, exactly as the model issued them.
    The clip at zero is a property of the loss, not of the forecast, and
    per_obs_losses applies it; storing clipped values would hide how far
    negative a model went.

    Written to a temporary file and renamed, so the cell either exists in full
    or does not exist at all. A sweep is expected to be interrupted, and the
    resume rule is "the file is there" -- a half-written .npz from a kill
    mid-write would be skipped forever and then fail the scoring pass.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.tmp'
    np.savez(tmp,
             pred=np.asarray(pred, dtype=float),
             pred_rv=np.asarray(pred_rv, dtype=float),
             true=np.asarray(true, dtype=float),
             dates=np.asarray([str(d) for d in
                               pd.DatetimeIndex(dates).strftime('%Y-%m-%d')]),
             meta=json.dumps(meta, sort_keys=True))
    # np.savez appends .npz unless the name already ends in it.
    os.replace(tmp if tmp.endswith('.npz') else f'{tmp}.npz', path)


def load_cell(path):
    """Read one cell back as (DataFrame indexed by date, meta dict).

    Cells written before the raw scale existed name their columns pred_ln /
    true_ln and carry no 'scale'; they were all ln(RV), so they are read under
    the current names with that scale filled in rather than being re-run.
    """
    with np.load(path, allow_pickle=False) as data:
        legacy = 'pred_ln' in data
        frame = pd.DataFrame(
            {'pred': data['pred_ln' if legacy else 'pred'],
             'pred_rv': data['pred_rv'],
             'true': data['true_ln' if legacy else 'true']},
            index=pd.DatetimeIndex(pd.to_datetime(data['dates']), name=DATE_COL))
        meta = json.loads(str(data['meta']))
    meta.setdefault('scale', LN_SCALE)
    return frame, meta
