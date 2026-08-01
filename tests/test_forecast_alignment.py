#!/usr/bin/env python3
"""
The join between a deep model and HAR-RV has to be exact, or every DM and MCS
number downstream compares the two on different days without saying so. This
checks it directly, on the real data file:

  1. Dataset_Custom.forecast_dates() returns exactly the dates HAR-RV_RUN.PY
     builds Y_h for on the test split -- same count, same days, same order.
  2. The target VALUES agree, so the two models are scored against one series
     rather than two constructions that merely share an index.
  3. Both hold at every horizon, on both scales, and independently of seq_len
     (a longer look-back must not shift the first forecast).

Run:  python tests/test_forecast_alignment.py

torch and scikit-learn are stubbed if absent, so the check runs without a
training environment -- neither is reached by the code paths under test.
"""

import os
import sys
import types

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(ROOT, 'data')
DATA_FILE = 'EURUSD-RV.csv'
TEST_START_YEAR = 2025
LAG_W, LAG_M = 5, 22


def _stub_missing_deps():
    """Stand in for torch / sklearn when they are not installed. Dataset_Custom
    touches neither with scale=False and timeenc=0."""
    if 'torch' not in sys.modules:
        try:
            import torch  # noqa: F401
        except ImportError:
            torch = types.ModuleType('torch')
            utils = types.ModuleType('torch.utils')
            data = types.ModuleType('torch.utils.data')

            class Dataset:
                pass

            class DataLoader:
                def __init__(self, *a, **k):
                    pass

            data.Dataset, data.DataLoader = Dataset, DataLoader
            utils.data = data
            torch.utils = utils
            sys.modules.update({'torch': torch, 'torch.utils': utils,
                                'torch.utils.data': data})
    try:
        import sklearn.preprocessing  # noqa: F401
    except ImportError:
        sk = types.ModuleType('sklearn')
        pre = types.ModuleType('sklearn.preprocessing')

        class StandardScaler:
            def fit(self, x):
                return self

            def transform(self, x):
                return x

            def inverse_transform(self, x):
                return x

        pre.StandardScaler = StandardScaler
        sk.preprocessing = pre
        sys.modules.update({'sklearn': sk, 'sklearn.preprocessing': pre})


def har_reference(horizon, log_mode):
    """
    Rebuild HAR-RV_RUN.PY's test-split target independently of that script, so
    the check is against the published construction and not a shared helper
    that could be wrong in both places.

    Mirrors load_base_features + build_horizon_target + split_by_year:
    non-positive rows dropped, regressors shifted one day, target the h-day
    forward MEAN with the log (if any) applied OUTSIDE the mean.
    """
    raw = pd.read_csv(os.path.join(DATA_DIR, DATA_FILE),
                      index_col=0, parse_dates=True)
    raw.index.name = 'date'
    s_raw = raw['RV'].sort_index().dropna().astype(float)
    s_raw = s_raw[s_raw > 0]
    s = np.log(s_raw) if log_mode else s_raw

    df = pd.DataFrame({'RV': s, 'RV_raw': s_raw})
    prev = s_raw.shift(1)
    if log_mode:
        df['RV_d'] = s.shift(1)
        df['RV_w'] = np.log(prev.rolling(LAG_W).mean())
        df['RV_m'] = np.log(prev.rolling(LAG_M).mean())
    else:
        df['RV_d'] = s.shift(1)
        df['RV_w'] = s.shift(1).rolling(LAG_W).mean()
        df['RV_m'] = s.shift(1).rolling(LAG_M).mean()

    if horizon == 1:
        df['Y_h'] = df['RV']
    else:
        fwd = df['RV_raw'].rolling(horizon).mean().shift(-(horizon - 1))
        df['Y_h'] = np.log(fwd) if log_mode else fwd

    df = df.dropna(subset=['Y_h', 'RV_d', 'RV_w', 'RV_m'])
    return df[df.index.year >= TEST_START_YEAR]['Y_h']


def dl_forecasts(horizon, log_mode, seq_len=96, label_len=48):
    """Dates and targets the deep-model test loader actually produces."""
    from data_provider.data_loader import Dataset_Custom

    ds = Dataset_Custom(
        root_path=DATA_DIR, data_path=DATA_FILE, flag='test',
        size=[seq_len, label_len, horizon], features='S', target='RV',
        timeenc=0, freq='d', log=log_mode, drop_nonpositive=True)

    dates = pd.to_datetime(ds.forecast_dates())

    # Reproduce Exp_Long_Term_Forecast._get_target: aggregate in VARIANCE space
    # and log the aggregate, never the terms.
    targets = []
    for i in range(len(ds)):
        window = np.asarray(ds.data_y[i + seq_len:i + seq_len + horizon, -1],
                            dtype=float)
        if log_mode:
            targets.append(np.log(np.mean(np.exp(window))))
        else:
            targets.append(np.mean(window))
    return pd.Series(np.asarray(targets), index=dates)


def check(horizon, log_mode, seq_len=96):
    scale = 'log' if log_mode else 'raw'
    har = har_reference(horizon, log_mode)
    dl = dl_forecasts(horizon, log_mode, seq_len=seq_len)

    label = f'h={horizon:<2} scale={scale:<3} seq_len={seq_len:<3}'
    problems = []

    if len(har) != len(dl):
        problems.append(f'count {len(dl)} (DL) vs {len(har)} (HAR)')
    elif not (har.index == dl.index).all():
        first = int(np.argmax((har.index == dl.index) == False))  # noqa: E712
        problems.append(f'dates diverge at row {first}: '
                        f'DL {dl.index[first].date()} vs '
                        f'HAR {har.index[first].date()}')
    else:
        gap = float(np.max(np.abs(har.to_numpy() - dl.to_numpy())))
        tol = 1e-9 * max(float(np.max(np.abs(har.to_numpy()))), 1.0)
        if gap > tol:
            problems.append(f'target values differ by up to {gap:.3e}')

    if problems:
        print(f'  FAIL  {label}  {"; ".join(problems)}')
        return False
    print(f'  ok    {label}  n={len(dl)}  '
          f'{dl.index[0].date()} .. {dl.index[-1].date()}')
    return True


def main():
    _stub_missing_deps()
    print('Deep-model test forecasts vs HAR-RV Y_h, on data/' + DATA_FILE)
    print('-' * 72)

    passed = True
    for log_mode in (False, True):
        for horizon in (1, 5, 22):
            passed &= check(horizon, log_mode)

    print('-' * 72)
    print('seq_len independence (the split border moves with the look-back,')
    print('so the first forecast must stay on the first test-year row):')
    for seq_len in (32, 64, 96, 128):
        passed &= check(5, True, seq_len=seq_len)

    print('-' * 72)
    print('PASS -- the two models key on identical dates' if passed else
          'FAIL -- do not run DM/MCS until this is fixed')
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
