#!/usr/bin/env python3
"""
Fold every run's per-date forecasts into the matrices Diebold-Mariano and the
Model Confidence Set consume.

    python aggregate_forecasts.py                       # forecasts/ + HAR CSVs -> analysis/
    python aggregate_forecasts.py --outdir analysis_v2

Inputs
------
forecasts/<tag>.csv        one per deep-model run, written by
                           Exp_Long_Term_Forecast._export_forecasts
har_rv[_log]_h??_fitted.csv  written by HAR-RV_RUN.PY's export_all

Outputs (in --outdir)
---------------------
forecasts_long.csv            every forecast from every model, one row each,
                              with per-observation losses attached
loss_rv_h<HH>_<loss>.csv      T x M matrix: rows = date, cols = model,
                              cells = that model's loss on that day. RAW
                              VARIANCE scale, so raw runs, --log runs and
                              HAR-RV all compete in one matrix.
loss_<scale>scale_h<HH>_se_model.csv
                              T x M squared error on the FITTED scale, within
                              one scale only (ln-scale and raw-scale errors are
                              different units and must not share a matrix).
mean_loss_h<HH>.csv           the aggregate table, for eyeballing before testing
alignment_report.txt          row counts, date coverage, integrity checks

The T x M layout is the native input of both tests: `arch.bootstrap.MCS` in
Python and `MCS::MCSprocedure` in R each take exactly this matrix, and a DM
test between two models is a two-column slice of it.

Why the losses are recomputed here rather than trusted from each run: QLIKE
needs one floor shared by every model (see utils.forecast_export.add_losses),
and a floor derived per run would let a model pick its own penalty.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.forecast_export import (FORECAST_COLUMNS, MODEL_SCALE_LOSSES,
                                   RV_LOSSES, add_losses)

# Must match data_provider/data_loader.py and HAR-RV_RUN.PY.
TRAIN_END_YEAR = 2022
QLIKE_FLOOR_FRAC = 1e-4

# y_true_rv is the same target for every model, so disagreement means a broken
# join. Relative tolerance, since RV spans orders of magnitude.
TARGET_RTOL = 1e-6


def qlike_floor_from_data(data_path, train_end_year=TRAIN_END_YEAR):
    """
    One QLIKE floor for the whole comparison: 1e-4 * mean training RV, read
    straight off the source series so it cannot drift between models.

    Non-positive rows are dropped first -- both producers drop them (they are
    exchange holidays, not zero-variance days), so the mean is taken over the
    same rows the models actually train on.
    """
    df = pd.read_csv(data_path, parse_dates=['date'])
    col = 'RV' if 'RV' in df.columns else df.select_dtypes('number').columns[0]
    rv = df.loc[df[col] > 0, [ 'date', col]]
    train_rv = rv.loc[rv['date'].dt.year <= train_end_year, col]
    if train_rv.empty:
        raise ValueError(f"no training rows (year <= {train_end_year}) in {data_path}")
    return QLIKE_FLOOR_FRAC * float(train_rv.mean())


def load_dl_forecasts(forecast_dir):
    """Read every per-run CSV the deep-model sweep produced."""
    paths = sorted(glob.glob(os.path.join(forecast_dir, '*.csv')))
    frames = []
    for p in paths:
        df = pd.read_csv(p, parse_dates=['date'])
        missing = set(FORECAST_COLUMNS) - set(df.columns)
        if missing:
            print(f"  SKIP {os.path.basename(p)}: missing columns {sorted(missing)}")
            continue
        frames.append(df[FORECAST_COLUMNS])
    return frames


def load_har_forecasts(har_dir, seed=0):
    """
    Convert HAR-RV_RUN.PY's *_h??_fitted.csv into the shared row format.

    Only test rows are kept -- HAR is fitted once on the training sample, so its
    training rows are in-sample fits and would not be a forecast comparison.

    Under --log the CSV already carries fitted_RV, the Jensen-corrected
    variance forecast; it is read rather than recomputed so the exported number
    is the one HAR's own metrics were built on. sigma^2 is backed out of it for
    the audit column (HAR's bias term is identically zero: an OLS fit with an
    intercept has mean residual zero by construction, which is exactly the
    guarantee a network lacks).
    """
    frames = []
    for p in sorted(glob.glob(os.path.join(har_dir, 'har_rv*_h??_fitted.csv'))):
        name = os.path.basename(p)
        scale = 'log' if 'har_rv_log' in name else 'raw'
        horizon = int(name.split('_h')[-1].split('_')[0])

        df = pd.read_csv(p, parse_dates=[0], index_col=0)
        df = df[df['split'] == 'test'].copy()
        if df.empty:
            print(f"  SKIP {name}: no test rows")
            continue

        y_true_model = df['Y_h'].to_numpy(dtype=float)
        y_pred_model = df['fitted'].to_numpy(dtype=float)
        if scale == 'log':
            y_true_rv = df['actual_RV'].to_numpy(dtype=float)
            y_pred_rv = df['fitted_RV'].to_numpy(dtype=float)
            resid_var = float(np.median(
                2.0 * (np.log(y_pred_rv) - y_pred_model)))
        else:
            y_true_rv, y_pred_rv = y_true_model, y_pred_model
            resid_var = 0.0

        frames.append(pd.DataFrame({
            'date': df.index,
            'model': 'HAR-RV',
            'scale': scale,
            'horizon': horizon,
            'seed': seed,
            'split': 'test',
            'y_true_rv': y_true_rv,
            'y_pred_rv': y_pred_rv,
            'y_true_model': y_true_model,
            'y_pred_model': y_pred_model,
            'bias': 0.0,
            'resid_var': resid_var,
        })[FORECAST_COLUMNS])
    return frames


def label_models(df):
    """
    Column name each model gets in the loss matrices.

    Scale is always in the label: a raw-scale and a --log fit of the same
    architecture are two different forecasts and must compete as two entries.
    The seed is appended only when a (model, scale, horizon) cell actually has
    more than one, so single-seed sweeps keep readable names.
    """
    out = df.copy()
    n_seeds = out.groupby(['model', 'scale', 'horizon'])['seed'].transform('nunique')
    base = out['model'] + '[' + out['scale'] + ']'
    out['label'] = np.where(n_seeds > 1,
                            base + 's' + out['seed'].astype(str),
                            base)
    return out


def loss_matrix(long_df, horizon, loss, scale=None):
    """
    Pivot to the T x M matrix both tests take: one row per date, one column per
    model, cells = that model's loss that day.

    Dates missing for any model are dropped, because DM and MCS both need a
    balanced panel -- a model with extra rows would otherwise be compared on a
    different sample. The count of dropped rows is returned so the caller can
    report it rather than discovering a silent trim later.
    """
    sub = long_df[long_df['horizon'] == horizon]
    if scale is not None:
        sub = sub[sub['scale'] == scale]
    if sub.empty:
        return None, 0

    mat = sub.pivot_table(index='date', columns='label', values=loss)
    n_before = len(mat)
    mat = mat.dropna(axis=0, how='any')
    return mat, n_before - len(mat)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='Build DM / MCS loss matrices from per-date forecasts.')
    ap.add_argument('--forecast-dir', default='./forecasts',
                    help='directory of per-run deep-model forecast CSVs')
    ap.add_argument('--har-dir', default='.',
                    help='directory holding har_rv*_h??_fitted.csv')
    ap.add_argument('--outdir', default='./analysis',
                    help='where the long table and loss matrices are written')
    ap.add_argument('--data', default='./data/EURUSD-RV.csv',
                    help='source RV series, used only for the shared QLIKE floor')
    ap.add_argument('--skip-har', action='store_true',
                    help='deep models only. Note DM then has no benchmark to '
                         'test against -- MCS still works.')
    args = ap.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    report = []

    def say(line=''):
        print(line)
        report.append(line)

    say('=' * 72)
    say('AGGREGATING PER-DATE FORECASTS FOR DM / MCS')
    say('=' * 72)

    frames = load_dl_forecasts(args.forecast_dir)
    say(f'  deep-model runs found : {len(frames)}  (in {args.forecast_dir})')
    if not args.skip_har:
        har = load_har_forecasts(args.har_dir)
        say(f'  HAR-RV horizons found : {len(har)}  (in {args.har_dir})')
        frames += har

    if not frames:
        say('  nothing to aggregate -- run the sweep first (orchestrate_rv.py)')
        return 1

    long_df = pd.concat(frames, ignore_index=True)

    floor = qlike_floor_from_data(args.data)
    say(f'  shared QLIKE floor    : {floor:.6e}  '
        f'(1e-4 x mean training RV, year <= {TRAIN_END_YEAR})')

    long_df = add_losses(long_df, floor)
    long_df = label_models(long_df)
    long_df = long_df.sort_values(['horizon', 'label', 'date'])

    long_path = os.path.join(args.outdir, 'forecasts_long.csv')
    long_df.to_csv(long_path, index=False)
    say(f'  -> {long_path}  ({len(long_df)} rows)')

    # -- integrity: every model must be scored against the same target -------
    say('')
    say('TARGET CONSISTENCY  (y_true_rv must agree across models)')
    for h, grp in long_df.groupby('horizon'):
        piv = grp.pivot_table(index='date', columns='label', values='y_true_rv')
        piv = piv.dropna(axis=0, how='any')
        if piv.empty or piv.shape[1] < 2:
            say(f'  h={h:>2}: not enough overlapping models to check')
            continue
        spread = (piv.max(axis=1) - piv.min(axis=1)).max()
        scale_ = float(np.nanmedian(np.abs(piv.to_numpy())))
        ok = spread <= TARGET_RTOL * max(scale_, 1e-12)
        say(f'  h={h:>2}: max spread {spread:.3e} across {piv.shape[1]} models '
            f'-> {"OK" if ok else "MISMATCH -- dates are misaligned"}')

    # -- floored forecasts ----------------------------------------------------
    floored = long_df.groupby(['horizon', 'label'])['n_floored'].sum()
    floored = floored[floored > 0]
    say('')
    if floored.empty:
        say('FLOORED FORECASTS: none -- no model predicted a non-positive variance')
    else:
        say('FLOORED FORECASTS  (non-positive variance forecasts, QLIKE floored)')
        for (h, lab), n in floored.items():
            say(f'  h={h:>2}  {lab:<28} {int(n)} day(s)')

    # -- the matrices ---------------------------------------------------------
    say('')
    say('LOSS MATRICES  (rows = date, cols = model; feed straight to DM / MCS)')
    horizons = sorted(long_df['horizon'].unique())

    for h in horizons:
        for loss in RV_LOSSES:
            mat, dropped = loss_matrix(long_df, h, loss)
            if mat is None or mat.empty:
                continue
            path = os.path.join(args.outdir, f'loss_rv_h{h:02d}_{loss}.csv')
            mat.to_csv(path)
            note = f'  ({dropped} unbalanced row(s) dropped)' if dropped else ''
            say(f'  -> {os.path.basename(path):<32} '
                f'{mat.shape[0]} x {mat.shape[1]}{note}')

        for scale in ('raw', 'log'):
            for loss in MODEL_SCALE_LOSSES:
                mat, dropped = loss_matrix(long_df, h, loss, scale=scale)
                if mat is None or mat.empty:
                    continue
                path = os.path.join(
                    args.outdir, f'loss_{scale}scale_h{h:02d}_{loss}.csv')
                mat.to_csv(path)
                note = f'  ({dropped} unbalanced row(s) dropped)' if dropped else ''
                say(f'  -> {os.path.basename(path):<32} '
                    f'{mat.shape[0]} x {mat.shape[1]}{note}')

        # -- mean-loss table, the thing to read before testing ---------------
        sub = long_df[long_df['horizon'] == h]
        summary = (sub.groupby('label')
                      .agg(n=('date', 'size'),
                           MSE_RV=('se_rv', 'mean'),
                           QLIKE=('qlike', 'mean'),
                           MSE_fitted_scale=('se_model', 'mean'),
                           MAE_RV=('ae_rv', 'mean'))
                      .sort_values('QLIKE'))
        spath = os.path.join(args.outdir, f'mean_loss_h{h:02d}.csv')
        summary.to_csv(spath)
        say(f'  -> {os.path.basename(spath):<32} {len(summary)} models')

    say('')
    say('Effective sample per horizon (overlapping targets induce MA(h-1) in')
    say('the loss differential, so these are NOT independent observations):')
    for h in horizons:
        n = long_df[long_df['horizon'] == h].groupby('label').size().max()
        say(f'  h={h:>2}: T = {n}, roughly {max(int(n // max(h, 1)), 1)} '
            f'independent blocks -- use HAC lag >= {max(h - 1, 1)} for DM and '
            f'block length >= {h} for the MCS bootstrap')

    rpath = os.path.join(args.outdir, 'alignment_report.txt')
    with open(rpath, 'w') as f:
        f.write('\n'.join(report) + '\n')
    print(f'\n  -> {rpath}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
