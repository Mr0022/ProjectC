#!/usr/bin/env python3
"""Score the stored forecasts: metric tables, and the loss matrices DM/MCS need.

    python orchestrate/aggregate_results.py                 # scores what is on disk
    python orchestrate/aggregate_results.py --strict        # fail on an incomplete block

``run_benchmark.py`` calls this at the end of a sweep, and it can be re-run on
its own at any time -- it reads only the per-cell ``.npz`` files, so a loss can
be added or changed without retraining anything.

What it writes, under ``<results_dir>``
---------------------------------------
``tables/metrics.csv``
    One row per (dataset, horizon, model, seed): MSE and MAE on both scales,
    QLIKE, and the fitted quantities behind them (Jensen bias and sigma^2,
    validation loss, parameter count, seconds).

``tables/metrics_mean.csv``
    The same averaged over seeds, with the standard deviation across them.
    With a single seed the std column is empty -- a reminder that one run sits
    closer to a best case than to a mean.

``tables/pivot_<metric>_h<hh>.csv``
    Models (rows) x datasets (columns) for one metric at one horizon -- the
    shape a results table in a paper has.

``forecasts/<dataset>_h<hh>__seed<S>.csv``
    Date-indexed actuals and every model's forecast, on both scales. Everything
    below is derived from this, so any other loss can be built from it without
    re-reading the .npz files.

``losses/<dataset>_h<hh>__<loss>__seed<S>.csv``
    THE FILES THE TESTS CONSUME. Date-indexed, one column per model, one row
    per forecast, for each of se_ln, ae_ln, qlike, se_rv, ae_rv.

      * Diebold-Mariano compares two columns: d_t = L_i,t - L_j,t, and the
        statistic is mean(d) / sqrt(HAC var(d) / T). The h-day targets overlap,
        so d_t is autocorrelated by construction and the long-run variance must
        be HAC-estimated with at least h-1 lags (Newey-West; Harvey-Leybourne-
        Newbold's small-sample correction is the usual companion).
      * The MCS takes the whole matrix and eliminates models until the survivors
        are statistically indistinguishable; the block bootstrap it resamples
        with needs a block length that respects the same overlap.

    Both are why the per-observation terms are stored rather than the means:
    neither test can be run from a table of averages.

Alignment
---------
Every column of a loss matrix is indexed by the date its target window opens,
and every model in a block is checked against the dates the split calendar
implies before the matrix is written. Models whose rows do not line up are not
comparable, and a DM statistic computed across a misalignment looks perfectly
healthy -- so the check is here, not left to the reader.
"""

import argparse
import glob
import json
import os
import sys
import time
from collections import OrderedDict, defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from orchestrate.benchmark_config import (  # noqa: E402
    DEFAULT_RESULTS_DIR, LOSS_TO_METRIC, MODELS, TARGET_TOL, dataset_spec,
    horizon_target, load_cell, per_obs_losses, qlike_floor, summarize_losses)

IDENTITY = ['dataset', 'asset', 'horizon', 'model', 'seed']
METRIC_COLS = list(LOSS_TO_METRIC.values())
EXTRA_COLS = ['n_obs', 'n_floored', 'target_dev', 'bias', 'resid_var',
              'val_loss', 'n_params', 'seconds']


# ---------------------------------------------------------------------------
# Reading the sweep back
# ---------------------------------------------------------------------------

def collect(results_dir):
    """Every stored cell as (rows, blocks, floors, targets).

    ``rows`` is the long-format metric table; ``blocks`` maps
    (dataset, horizon) to {(model, seed): (frame, losses)} and is what the
    matrices are built from.

    Every cell is scored against ``horizon_target`` -- the actuals rebuilt from
    the CSV in float64 -- rather than against the copy it stored, and how far
    the two sit apart is recorded per cell as ``target_dev``. That column is
    the check that a deep model and HAR-RV really are predicting the same
    Y^(h) on the same rows; it should read ~1e-7 for the networks (a float32
    round trip through the loader's scaler) and ~1e-16 for HAR-RV.
    """
    pattern = os.path.join(results_dir, 'runs', '*', 'h*', '*.npz')
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(
            f'no forecasts under {os.path.join(results_dir, "runs")}. Run '
            f'orchestrate/run_benchmark.py first.')

    floors, targets, rows, blocks = {}, {}, [], defaultdict(OrderedDict)
    for path in paths:
        frame, meta = load_cell(path)
        dataset, h = meta['dataset'], int(meta['horizon'])
        spec = dataset_spec(dataset)
        if dataset not in floors:
            floors[dataset] = qlike_floor(spec)
        if (dataset, h) not in targets:
            targets[(dataset, h)] = horizon_target(spec, h)
        floor, target = floors[dataset], targets[(dataset, h)]

        if not frame.index.equals(target.index):
            print(f'[WARN] {os.path.relpath(path, results_dir)}: forecast dates '
                  f'do not match the {spec.asset} calendar; cell skipped.')
            continue
        deviation = float(np.max(np.abs(frame['true_ln'].values
                                        - target.values)))

        losses = per_obs_losses(frame['pred_ln'], frame['pred_rv'],
                                target.values, floor)
        row = {'dataset': dataset, 'asset': meta['asset'], 'horizon': h,
               'model': meta['model'], 'seed': meta.get('seed'),
               'n_obs': len(frame),
               'n_floored': int((frame['pred_rv'] <= 0).sum()),
               'target_dev': deviation}
        row.update(summarize_losses(losses))
        for key in ('bias', 'resid_var', 'val_loss', 'n_params', 'seconds'):
            row[key] = meta.get(key)
        rows.append(row)
        blocks[(dataset, h)][(meta['model'], meta.get('seed'))] = (frame, losses)

    return rows, blocks, floors, targets


def block_seeds(cells):
    """Seeds present in one block; HAR-RV's None belongs to all of them."""
    seeds = sorted({seed for (_, seed) in cells if seed is not None})
    return seeds or [None]


def check_targets(rows, strict):
    """Report any cell whose own actuals drift from the rebuilt Y^(h).

    Dates are already checked when a cell is read; this checks the values. A
    cell that fails is predicting a different target -- a horizon aggregation
    or a scale that does not match the rest of the grid -- and no loss matrix
    built from it means anything, so it is worth stopping for.
    """
    bad = [r for r in rows if r['target_dev'] > TARGET_TOL]
    for row in bad:
        message = (f"{row['dataset']} h={row['horizon']} {row['model']}: its "
                   f"stored actuals differ from Y^(h) rebuilt from the CSV by "
                   f"{row['target_dev']:.3g} (tolerance {TARGET_TOL:g}). The "
                   f"cell is not scoring the same target as the rest of the "
                   f"grid.")
        if strict:
            raise SystemExit(message)
        print(f'[WARN] {message}')
    return bad


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def model_order(names):
    """MODELS order first, anything unrecognised appended alphabetically."""
    known = [m for m in MODELS if m in names]
    return known + sorted(n for n in names if n not in set(MODELS))


def seedmean_members(cells):
    """{model: (pred_ln, pred_rv, n_seeds)} — the repeats averaged into one forecast.

    Averaged on BOTH scales separately, because each loss is defined on one of
    them: the ln-scale losses see the mean ln forecast, and QLIKE / the
    variance-scale losses see the mean variance forecast, which is the
    conditional mean they are minimised by. Passing one through exp() to get
    the other would give the geometric mean of the repeats instead.

    HAR-RV is deterministic and has a single cell, so it passes through
    unchanged and stays comparable to the ensembles.
    """
    frames = defaultdict(list)
    for (model, _), (frame, _) in cells.items():
        frames[model].append(frame)
    return {model: (np.mean([f['pred_ln'].values for f in group], axis=0),
                    np.mean([f['pred_rv'].values for f in group], axis=0),
                    len(group))
            for model, group in frames.items()}


def write_seedmean(results_dir, dataset, asset, h, cells, target, floor):
    """The seed-averaged forecast and its loss matrices, plus its metric rows.

    With --itr 10 there are ten per-seed loss matrices per block and no
    principled way to combine ten DM statistics, so this is the single series
    to run the tests on: one forecast per model, the ensemble of its repeats.

    Its metrics are NOT the seed means in metrics_mean.csv -- the loss of an
    average is below the average of the losses whenever the loss is convex, so
    the ensemble reads better than any repeat. It is a different (and better)
    forecast, not a smoothed report of the same one.
    """
    members = seedmean_members(cells)
    order = model_order(members)
    dates = target.index
    stem = f'{dataset}_h{h:02d}'

    wide = pd.DataFrame(index=dates)
    wide['y_ln'] = target.values
    wide['y_rv'] = np.exp(target.values)
    losses, rows = {}, []
    for model in order:
        pred_ln, pred_rv, n_seeds = members[model]
        wide[f'{model}__ln'] = pred_ln
        wide[f'{model}__rv'] = pred_rv
        losses[model] = per_obs_losses(pred_ln, pred_rv, target.values, floor)
        row = {'dataset': dataset, 'asset': asset, 'horizon': h,
               'model': model, 'n_seeds': n_seeds, 'n_obs': len(dates)}
        row.update(summarize_losses(losses[model]))
        rows.append(row)
    wide.to_csv(os.path.join(results_dir, 'forecasts', f'{stem}__seedmean.csv'))

    for loss in LOSS_TO_METRIC:
        pd.DataFrame({model: losses[model][loss] for model in order},
                     index=dates, columns=order).to_csv(
            os.path.join(results_dir, 'losses', f'{stem}__{loss}__seedmean.csv'))
    return rows


def write_block(results_dir, dataset, h, cells, target, complete):
    """Forecast and loss files for one (dataset, horizon) block, per seed."""
    forecasts_dir = os.path.join(results_dir, 'forecasts')
    losses_dir = os.path.join(results_dir, 'losses')
    os.makedirs(forecasts_dir, exist_ok=True)
    os.makedirs(losses_dir, exist_ok=True)

    dates = target.index
    for seed in block_seeds(cells):
        # HAR-RV carries no seed and belongs in every seed's matrix: it is the
        # same baseline whatever the networks were initialised with.
        members = {model: payload for (model, s), payload in cells.items()
                   if s == seed or s is None}
        if not members:
            continue
        order = model_order(members)
        stem = f'{dataset}_h{h:02d}'
        # The seed goes last so a directory listing groups by loss, which is
        # how these are read: one loss, every dataset.
        suffix = '' if seed is None else f'__seed{seed}'

        wide = pd.DataFrame(index=dates)
        wide['y_ln'] = target.values
        wide['y_rv'] = np.exp(target.values)
        for model in order:
            frame, _ = members[model]
            wide[f'{model}__ln'] = frame['pred_ln'].values
            wide[f'{model}__rv'] = frame['pred_rv'].values
        wide.to_csv(os.path.join(forecasts_dir, f'{stem}{suffix}.csv'))

        for loss in LOSS_TO_METRIC:
            matrix = pd.DataFrame(
                {model: members[model][1][loss] for model in order},
                index=dates, columns=order)
            matrix.to_csv(os.path.join(losses_dir,
                                       f'{stem}__{loss}{suffix}.csv'))

    if not complete:
        print(f'[note] {dataset} h={h}: block holds {len(cells)} cell(s); the '
              f'MCS is defined over a fixed set of models, so re-run the '
              f'missing ones before using it.')


def write_tables(results_dir, rows, seedmean_rows=()):
    tables_dir = os.path.join(results_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)

    if len(seedmean_rows):
        ensemble = pd.DataFrame(list(seedmean_rows))
        ensemble['model'] = pd.Categorical(
            ensemble['model'], categories=model_order(ensemble['model'].unique()),
            ordered=True)
        ensemble = ensemble.sort_values(['dataset', 'horizon', 'model'])
        ensemble.to_csv(os.path.join(tables_dir, 'metrics_seedmean.csv'),
                        index=False)

    frame = pd.DataFrame(rows)
    # Nullable Int64: HAR-RV has no seed, and a plain int column with one
    # missing value would silently turn every seed into a float.
    frame['seed'] = frame['seed'].astype('Int64')
    frame['model'] = pd.Categorical(frame['model'],
                                    categories=model_order(frame['model'].unique()),
                                    ordered=True)
    frame = frame.sort_values(['dataset', 'horizon', 'model', 'seed'])
    frame = frame[IDENTITY + METRIC_COLS + EXTRA_COLS]
    frame.to_csv(os.path.join(tables_dir, 'metrics.csv'), index=False)

    # Seed-averaged. std is the initialisation spread of the configuration and
    # is empty with one seed, which is the honest way to show it is unmeasured.
    grouped = frame.groupby(['dataset', 'asset', 'horizon', 'model'],
                            observed=True)
    mean = grouped[METRIC_COLS].mean()
    std = grouped[METRIC_COLS].std(ddof=1)
    std.columns = [f'{c}_std' for c in std.columns]
    summary = pd.concat([mean, std, grouped.size().rename('n_seeds')], axis=1)
    summary = summary.reset_index().sort_values(['dataset', 'horizon', 'model'])
    summary.to_csv(os.path.join(tables_dir, 'metrics_mean.csv'), index=False)

    for metric in METRIC_COLS:
        for h in sorted(frame['horizon'].unique()):
            part = summary[summary['horizon'] == h]
            if part.empty:
                continue
            pivot = part.pivot(index='model', columns='dataset', values=metric)
            pivot = pivot.reindex(model_order(pivot.index))
            pivot.to_csv(os.path.join(tables_dir, f'pivot_{metric}_h{h:02d}.csv'))
    return frame, summary


def print_summary(summary):
    """A compact readout: mean over datasets, per model and horizon."""
    if summary.empty:
        return
    for h in sorted(summary['horizon'].unique()):
        part = summary[summary['horizon'] == h]
        grouped = part.groupby('model', observed=True)
        table = grouped[['MSE_ln', 'MAE_ln', 'QLIKE']].mean()
        table['n'] = grouped['dataset'].nunique()
        table = table.reindex(model_order(table.index)).dropna(how='all')
        print(f"\n  h = {h}   mean over the datasets scored so far")
        print(f"  {'model':<14} {'n':>3} {'MSE[ln]':>10} {'MAE[ln]':>10} "
              f"{'QLIKE':>10}")
        print('  ' + '-' * 50)
        for model, row in table.iterrows():
            print(f"  {model:<14} {int(row['n']):>3} {row['MSE_ln']:>10.6f} "
                  f"{row['MAE_ln']:>10.6f} {row['QLIKE']:>10.6f}")


# ---------------------------------------------------------------------------

def aggregate(results_dir=DEFAULT_RESULTS_DIR, strict=False, quiet=False):
    """Score every stored cell and write the tables, forecasts and loss matrices."""
    results_dir = os.path.abspath(results_dir)
    rows, blocks, floors, targets = collect(results_dir)
    check_targets(rows, strict)

    expected_models = set(MODELS)
    coverage, seedmean_rows = {}, []
    for (dataset, h), cells in sorted(blocks.items()):
        target = targets[(dataset, h)]
        dates = target.index
        present = {model for (model, _) in cells}
        missing = sorted(expected_models - present)
        coverage[f'{dataset}_h{h:02d}'] = {
            'n_obs': int(len(dates)),
            'first': str(dates[0].date()), 'last': str(dates[-1].date()),
            'models': sorted(present), 'missing': missing}
        write_block(results_dir, dataset, h, cells, target, not missing)
        # One repeat is not an ensemble, and its seedmean files would only
        # duplicate the per-seed ones.
        if len(block_seeds(cells)) > 1:
            seedmean_rows += write_seedmean(
                results_dir, dataset, dataset_spec(dataset).asset, h, cells,
                target, floors[dataset])

    frame, summary = write_tables(results_dir, rows, seedmean_rows)

    manifest = {
        'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_cells': int(len(frame)),
        'qlike_floor': {k: float(v) for k, v in sorted(floors.items())},
        'qlike_floor_note': 'inert under --log: exp(.) forecasts are positive, '
                            'so no forecast is ever floored (n_floored == 0)',
        'max_target_dev': float(frame['target_dev'].max()) if len(frame) else 0.0,
        'target_dev_note': 'largest gap between a cell\'s own actuals and '
                           'Y^(h) rebuilt from the CSV; ~1e-7 is the deep '
                           'models\' float32 round trip through the scaler',
        'blocks': coverage,
    }
    with open(os.path.join(results_dir, 'manifest.json'), 'w') as fh:
        json.dump(manifest, fh, indent=2)

    if not quiet:
        print(f'\nScored {len(frame)} cell(s) from {len(blocks)} '
              f'(dataset, horizon) block(s).')
        print(f'  tables    : {os.path.join(results_dir, "tables")}')
        print(f'  forecasts : {os.path.join(results_dir, "forecasts")}')
        print(f'  losses    : {os.path.join(results_dir, "losses")}  '
              f'(DM / MCS input)')
        floored = int(frame['n_floored'].sum())
        if floored:
            print(f'  [WARN] {floored} non-positive variance forecast(s) were '
                  f'floored for QLIKE; under --log there should be none.')
        print_summary(summary)
    return frame, summary


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='Score the benchmark forecasts and build the DM/MCS inputs.')
    ap.add_argument('--results_dir', default=DEFAULT_RESULTS_DIR)
    ap.add_argument('--strict', action='store_true',
                    help='treat a misaligned block as an error instead of '
                         'skipping it')
    args = ap.parse_args(argv)
    aggregate(args.results_dir, strict=args.strict)
    return 0


if __name__ == '__main__':
    sys.exit(main())
