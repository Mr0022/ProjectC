"""
Train / validation / test calendars -- the single source of truth.

Both the deep models (data_provider/data_loader.py) and the HAR-RV baseline
(HAR-RV_RUN.PY) import their split boundaries from here, so the two can never
drift apart. Everything in this module is deliberately pandas/numpy only --
no torch, no sklearn -- so HAR-RV keeps running in a bare environment.

The two asset classes in data/ do not share a calendar:

    forex   train 2012-01 .. 2022-12 | val 2023-01 .. 2024-12 | test 2025-01 .. end
    crypto  train 2018-06 .. 2024-06 | val 2024-07 .. 2025-06 | test 2025-07 .. 2026-06

Boundaries are (year, month) pairs, INCLUSIVE of the whole month named, so a
split may end mid-year. `train_start` and `test_end` may be None, meaning "run
to the start / end of the sample" -- the open-ended forex behaviour.

HAR-RV is OLS and has no hyperparameters to tune, so a held-out validation set
would serve no purpose there. It folds the validation months back into the
estimation sample (see `har_bounds`) and keeps the TEST window identical, which
is what makes the two families' out-of-sample losses comparable.
"""

from collections import namedtuple

import numpy as np
import pandas as pd

SplitSpec = namedtuple(
    'SplitSpec', ['name', 'train_start', 'train_end', 'val_end', 'test_end'])


# ==============================================================================
# Month arithmetic. Months are compared as a single sortable integer,
# year * 12 + month, which makes "is this row inside the window?" one integer
# comparison and sidesteps every day-of-month and calendar edge case.
# ==============================================================================

def to_ordinal(ym):
    """(year, month) -> month ordinal. None passes through."""
    return None if ym is None else ym[0] * 12 + ym[1]


def month_ordinal(dates):
    """Calendar month of each row as a sortable int, aligned 1:1 with `dates`."""
    d = pd.to_datetime(pd.Series(np.asarray(dates)))
    return d.dt.year * 12 + d.dt.month


def next_month(ym):
    """The month after `ym`, rolling the year over at December."""
    year, month = ym
    return (year + 1, 1) if month == 12 else (year, month + 1)


def fmt_month(ym):
    """(2024, 6) -> '2024-06'. None -> '?' so a label never reads as blank."""
    return '?' if ym is None else f"{ym[0]:04d}-{ym[1]:02d}"


def month_of(date):
    """A single timestamp -> (year, month)."""
    ts = pd.Timestamp(date)
    return (int(ts.year), int(ts.month))


def month_mask(dates, lo=None, hi=None):
    """Boolean mask selecting rows whose calendar month lies in [lo, hi].

    Both bounds are inclusive (year, month) pairs; None means unbounded on that
    side. Used by HAR-RV, which selects rows by date rather than by row index.
    """
    months = month_ordinal(dates)
    mask = pd.Series(True, index=months.index)
    if lo is not None:
        mask &= months >= to_ordinal(lo)
    if hi is not None:
        mask &= months <= to_ordinal(hi)
    return mask.values


# ==============================================================================
# Row borders for the sequence models
# ==============================================================================

def month_split_borders(dates, seq_len, n_rows, spec):
    """
    Row borders for the chronological split, indexed by set_type
    (0 = train, 1 = val, 2 = test).

    `dates` must be datetime-like, sorted, and aligned 1:1 with the data rows.
    `spec` is the SplitSpec of the asset class (FOREX_SPLIT / CRYPTO_SPLIT).

    Boundaries are resolved at MONTH granularity, so a split can end mid-year
    (crypto ends its training sample at 2024-06). Rows outside the spec -- before
    `train_start` or after `test_end` -- are excluded from every split rather
    than absorbed into the nearest one, so extending a CSV past `test_end` never
    silently grows the test set.

    The val and test windows start seq_len rows early so the first forecast
    origin in each split has a complete look-back window. Those extra rows are
    context only -- they are never used as targets, so no target from an
    earlier split is scored twice.

    Returns (border1s, border2s); border2s[0] - border1s[0] is the train row count.
    """
    months = month_ordinal(dates)
    if not months.is_monotonic_increasing:
        raise ValueError(
            "Rows must be sorted by date before the split borders are computed; "
            "the borders are row counts and assume chronological order.")

    start = 0 if spec.train_start is None \
        else int((months < to_ordinal(spec.train_start)).sum())
    train_end = int((months <= to_ordinal(spec.train_end)).sum())
    val_end = int((months <= to_ordinal(spec.val_end)).sum())
    test_end = n_rows if spec.test_end is None \
        else int((months <= to_ordinal(spec.test_end)).sum())

    # Clamp the seq_len look-back so a short train/val split cannot produce a
    # negative index (which would silently slice from the end of the array).
    border1s = [start,
                max(start, train_end - seq_len),
                max(start, val_end - seq_len)]
    border2s = [train_end, val_end, test_end]
    return border1s, border2s


# ==============================================================================
# Bounds for HAR-RV
# ==============================================================================

def har_bounds(spec):
    """
    (train_start, train_end, test_start, test_end) for the HAR-RV baseline on
    this calendar, as inclusive (year, month) pairs.

    HAR-RV is OLS: no hyperparameters, so no validation set is needed and the
    validation months are folded into the estimation sample --

        train : spec.train_start .. spec.val_end     [= DL train + val]
        test  : spec.val_end + 1 .. spec.test_end    [= DL test, unchanged]

    The test window is byte-identical to the deep models', which is the whole
    point: it keeps the out-of-sample losses directly comparable. Note the deep
    models never FIT on the validation months, they only use them for early
    stopping and model selection.
    """
    return (spec.train_start, spec.val_end,
            next_month(spec.val_end), spec.test_end)


# ==============================================================================
# SECTION 1 -- FOREX     data/EURUSD-RV.csv, AUDUSD, EURGBP, USDCHF, USDJPY
#
# Sample runs 2012-01 .. 2026-06, split on year boundaries:
#
#     train : start of sample      ..  end of TRAIN_END_YEAR   (2012 - 2022)
#     val   : TRAIN_END_YEAR + 1   ..  end of VAL_END_YEAR     (2023 - 2024)
#     test  : VAL_END_YEAR + 1     ..  end of sample           (2025 - end)
#
# Both ends stay open: training starts at whatever the file starts at, and the
# test window runs to the last row, so appending FX data extends the test set.
# ==============================================================================
TRAIN_END_YEAR = 2022
VAL_END_YEAR = 2024

FOREX_SPLIT = SplitSpec(
    name='forex',
    train_start=None,                  # open: start of sample
    train_end=(TRAIN_END_YEAR, 12),
    val_end=(VAL_END_YEAR, 12),
    test_end=None,                     # open: end of sample
)


# ==============================================================================
# SECTION 2 -- CRYPTO    data/btcusdt-RV.csv, ethusdt, adausdt, bnbusdt, xrpusdt
#
# Sample runs 2018-06 .. 2026-06. Crypto starts mid-2018 and trades every
# calendar day, so the windows are set at mid-year boundaries instead:
#
#     train : 2018-06  ..  2024-06     (CRYPTO_TRAIN_START .. CRYPTO_TRAIN_END)
#     val   : 2024-07  ..  2025-06     (CRYPTO_VAL_END)
#     test  : 2025-07  ..  2026-06     (CRYPTO_TEST_END)
#
# Unlike forex, BOTH ends are closed. Rows before 2018-06 or after 2026-06 are
# excluded from every split rather than folded into train or test, so extending
# a crypto CSV does not silently change what the test metrics cover.
# ==============================================================================
CRYPTO_TRAIN_START = (2018, 6)
CRYPTO_TRAIN_END = (2024, 6)
CRYPTO_VAL_END = (2025, 6)
CRYPTO_TEST_END = (2026, 6)

CRYPTO_SPLIT = SplitSpec(
    name='crypto',
    train_start=CRYPTO_TRAIN_START,
    train_end=CRYPTO_TRAIN_END,
    val_end=CRYPTO_VAL_END,
    test_end=CRYPTO_TEST_END,
)


# Asset class -> calendar. Keys are what '--data' (run.py) and '--asset'
# (HAR-RV_RUN.PY) accept.
SPLITS = {
    'forex': FOREX_SPLIT,
    'crypto': CRYPTO_SPLIT,
}
