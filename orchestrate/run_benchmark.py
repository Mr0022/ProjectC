#!/usr/bin/env python3
"""Run every model on every dataset at every horizon, and store the forecasts.

    10 deep models + HAR-RV   x   5 forex + 5 crypto   x   h = 1, 5, 22

Each deep-model cell is trained with the hyper-parameters Optuna selected for
that model (``tuning/ProjectC_tuning/<Model>_best.json``) under the protocol
those studies used -- ``--aggregate_mean --log``, ``seq_len 96``, univariate RV
-- so the target is HAR-RV's Y^(h) on the ln(RV) scale and the baseline is
scored on identical rows against identical actuals. HAR-RV itself is fitted by
``HAR-RV_RUN.PY --log``, unmodified.

    python orchestrate/run_benchmark.py                      # the whole grid
    python orchestrate/run_benchmark.py --dry_run            # print the plan
    python orchestrate/run_benchmark.py --validate           # build every config
    python orchestrate/run_benchmark.py --assets crypto
    python orchestrate/run_benchmark.py --models DLinear FITS --horizons 1
    python orchestrate/run_benchmark.py --quick              # 5 epochs, smoke test
    python orchestrate/run_benchmark.py --itr 3             # 3 repeats instead of 10

What a cell writes
------------------
``<results_dir>/runs/<dataset>/h<hh>/<Model>_seed<seed>.npz`` holding the
per-observation test forecasts -- ``pred_ln`` (what the model produced),
``pred_rv`` (the lognormal back-transform, which is what QLIKE is defined on),
``true_ln``, and the target dates. Metrics are NOT computed here: they are
means of per-observation losses, and Diebold-Mariano and the MCS need the terms
rather than the means, so both come from the same file in
``aggregate_results.py``. That is also why the forecasts, not the losses, are
what a run persists -- a loss can be changed afterwards without retraining.

Design notes
------------
* **One subprocess per cell.** A cell is trained by re-invoking this file in
  ``--worker`` mode, so an out-of-memory kill, a CUDA fault or a stray
  ``sys.exit`` costs one cell instead of the sweep. The worker trains
  in-process through run.py's own parser and Exp_Long_Term_Forecast, so a cell
  is exactly what the printed ``python -u run.py ...`` line reproduces.

* **Resumable.** A cell whose ``.npz`` exists is skipped, and a cell that
  failed is recorded in ``failures.csv`` and skipped too (``--retry_failed`` to
  re-run). An interrupted sweep is restarted by running the same command again.

* **Ordered dataset -> horizon -> model**, so an interrupted sweep leaves
  COMPLETE (dataset, horizon) blocks behind. A block is what the MCS needs: a
  test with a subset of the models in it is a different test, not a partial one.

* **The alignment is checked, not assumed.** Every cell asserts its forecast
  count against the dates ``benchmark_config.test_target_dates`` derives from
  the split calendar, so a model that silently drops or duplicates a window
  fails at its own cell rather than at the DM test three hours later.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from orchestrate.benchmark_config import (  # noqa: E402
    DEEP_MODELS, DEFAULT_ANCHOR_DIR, DEFAULT_N_SEEDS, DEFAULT_PATIENCE,
    DEFAULT_RESULTS_DIR, DEFAULT_SEED, DEFAULT_TRAIN_EPOCHS, HAR_MODEL,
    HORIZONS, anchor_argv, cell_id, cell_path, dataset_names, dataset_spec,
    discover_anchors, save_cell, seed_list, test_target_dates)

FAILURE_FIELDS = ['dataset', 'horizon', 'model', 'seed', 'when', 'why']

# HAR-RV is fitted once per dataset for all three horizons, so its failures are
# keyed on horizon 0 rather than on any one of them.
HAR_FAILURE_HORIZON = 0


# ---------------------------------------------------------------------------
# One deep-model cell
# ---------------------------------------------------------------------------

def train_cell(model, spec, h, seed, args):
    """Train one cell and return (frame, meta). Runs in the worker process.

    The three passes after training mirror what Exp_Long_Term_Forecast.test()
    does under ``--aggregate_mean --log``, using the experiment's own
    ``_forward_collect`` rather than a copy of it:

      * validation, for the selection metric of the checkpoint that was kept;
      * test, for the forecasts this benchmark is about;
      * train, for the Jensen correction. exp(.) of a log-scale forecast is the
        conditional MEDIAN; the mean needs exp(bias + sigma^2/2) with both
        terms measured on TRAINING residuals -- test residuals would leak the
        out-of-sample outcome into the forecast.

    The training pass is sequential and keeps the last partial batch, unlike
    the loader used for optimisation: the correction is part of the forecast
    this benchmark stores, so it must not depend on the shuffle order.
    """
    import torch
    from torch.utils.data import DataLoader
    from utils.metrics import lognormal_back_transform

    os.chdir(REPO_ROOT)                       # exp_basic scans models/
    import run as run_module
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

    name = cell_id(spec.name, h, model)
    checkpoints = os.path.join(args.checkpoint_dir, name)
    argv, anchor_meta = anchor_argv(
        model, spec, h, seed, anchor_dir=args.anchor_dir,
        checkpoints=checkpoints, train_epochs=args.train_epochs,
        patience=args.patience, use_gpu=args.use_gpu)

    parsed = run_module.build_parser().parse_args(argv)
    run_module.finalize_args(parsed)
    run_module.set_seed(seed)

    started = time.time()
    exp = Exp_Long_Term_Forecast(parsed)
    setting = f'{name}_{seed}'
    try:
        exp.train(setting)

        def sequential(dataset):
            return DataLoader(dataset, batch_size=parsed.batch_size,
                              shuffle=False, drop_last=False,
                              num_workers=parsed.num_workers)

        # Validation MSE of the checkpoint early stopping kept, over the WHOLE
        # split. Exp.vali averages per-batch means on a shuffled loader that
        # drops its last partial batch, so this is the same quantity computed
        # exactly rather than the number the stopping rule happened to see.
        vali_data, _ = exp._get_data(flag='val')
        val_pred, val_true = exp._forward_collect(sequential(vali_data))
        val_loss = float(np.mean((val_true - val_pred) ** 2))

        _, test_loader = exp._get_data(flag='test')
        pred_ln, true_ln = exp._forward_collect(test_loader)

        train_data, _ = exp._get_data(flag='train')
        tr_pred, tr_true = exp._forward_collect(sequential(train_data))
    finally:
        if not args.keep_checkpoints:
            shutil.rmtree(checkpoints, ignore_errors=True)

    resid = tr_true - tr_pred
    bias, resid_var = float(resid.mean()), float(resid.var(ddof=1))
    pred_rv = lognormal_back_transform(pred_ln, resid_var, bias)

    dates = test_target_dates(spec, h)
    if len(dates) != len(pred_ln):
        raise RuntimeError(
            f'{name}: the test loader emitted {len(pred_ln)} forecast(s) but '
            f'the {spec.asset} calendar puts {len(dates)} target window(s) in '
            f'the test split. The two are computed independently and must '
            f'agree, or the forecasts cannot be aligned across models.')

    meta = {
        'model': model, 'dataset': spec.name, 'asset': spec.asset,
        'horizon': h, 'seed': seed, 'n_obs': int(len(pred_ln)),
        'bias': bias, 'resid_var': resid_var, 'val_loss': val_loss,
        'seconds': round(time.time() - started, 1),
        'n_params': int(sum(p.numel() for p in exp.model.parameters())),
        'train_epochs': args.train_epochs, 'patience': args.patience,
        'anchor': os.path.relpath(anchor_meta['path'], REPO_ROOT),
        'anchor_val_loss': anchor_meta['best_val_loss'],
        'command': 'python -u run.py ' + ' '.join(argv),
        'torch': torch.__version__,
        'device': str(exp.device),
    }
    return pred_ln, pred_rv, true_ln, dates, meta


def run_worker(args):
    """``--worker``: train exactly one cell and write its .npz. One per process."""
    spec = dataset_spec(args.dataset)
    pred_ln, pred_rv, true_ln, dates, meta = train_cell(
        args.model, spec, args.horizon, args.seed, args)
    save_cell(args.out, pred_ln, pred_rv, true_ln, dates, meta)
    print(f"[worker] wrote {args.out}  n={meta['n_obs']}  "
          f"val_loss={meta['val_loss']:.6f}  {meta['seconds']:.0f}s", flush=True)


def worker_cmd(model, dataset, h, seed, out, args):
    """The command the parent spawns for one cell."""
    cmd = [sys.executable, '-u', os.path.abspath(__file__), '--worker',
           '--model', model, '--dataset', dataset, '--horizon', str(h),
           '--seed', str(seed), '--out', out,
           '--anchor_dir', args.anchor_dir,
           '--checkpoint_dir', args.checkpoint_dir,
           '--train_epochs', str(args.train_epochs),
           '--patience', str(args.patience)]
    if args.keep_checkpoints:
        cmd.append('--keep_checkpoints')
    if not args.use_gpu:
        cmd.append('--no_use_gpu')
    return cmd


# ---------------------------------------------------------------------------
# HAR-RV
# ---------------------------------------------------------------------------

def har_outdir(results_dir, dataset):
    return os.path.join(results_dir, 'har', dataset)


def convert_har(spec, h, results_dir):
    """Fold one horizon of a HAR-RV run into the same .npz format as a cell.

    HAR-RV_RUN.PY --log exports, per horizon, the fitted values for both splits
    with the back-transformed variance forecast alongside; the test rows of
    that file are the baseline's forecasts. Nothing is recomputed here -- the
    columns are read as they were written, so the baseline in this benchmark is
    the baseline that script reports.
    """
    path = os.path.join(har_outdir(results_dir, spec.name),
                        f'har_rv_log_h{h:02d}_fitted.csv')
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f'{spec.name} h={h}: HAR-RV wrote no {os.path.basename(path)}. '
            f'Check the HAR-RV_RUN.PY output above.')
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    test = frame[frame['split'] == 'test']

    dates = test_target_dates(spec, h)
    if not test.index.equals(dates):
        raise RuntimeError(
            f'{spec.name} h={h}: HAR-RV scored {len(test)} row(s) '
            f'({test.index.min().date()}..{test.index.max().date()}) but the '
            f'{spec.asset} calendar puts {len(dates)} window(s) in the test '
            f'split ({dates.min().date()}..{dates.max().date()}). The deep '
            f'models and HAR-RV must forecast the same dates, or the DM and '
            f'MCS tests compare different samples.')

    # sigma^2 back out of fitted_RV = exp(fitted + sigma^2/2); the export does
    # not carry it, and it belongs in the record next to the deep models'.
    resid_var = float(np.mean(2.0 * (np.log(test['fitted_RV'].values)
                                     - test['fitted'].values)))
    meta = {'model': HAR_MODEL, 'dataset': spec.name, 'asset': spec.asset,
            'horizon': h, 'seed': None, 'n_obs': int(len(test)),
            'bias': 0.0, 'resid_var': resid_var,
            'command': f'python -u HAR-RV_RUN.PY --data data/{spec.data_path} '
                       f'--asset {spec.asset} --log',
            'source': os.path.relpath(path, REPO_ROOT)}
    save_cell(cell_path(results_dir, spec.name, h, HAR_MODEL),
              test['fitted'].values, test['fitted_RV'].values,
              test['Y_h'].values, dates, meta)
    return meta


def run_har(spec, args):
    """Fit HAR-RV once for a dataset; it produces all three horizons in one pass."""
    outdir = har_outdir(args.results_dir, spec.name)
    cmd = [sys.executable, '-u', 'HAR-RV_RUN.PY',
           '--data', os.path.join('data', spec.data_path),
           '--asset', spec.asset, '--log', '--outdir', outdir]
    print(f"\n{'=' * 72}\n[HAR-RV] {spec.name} ({spec.asset})\n"
          f"{' '.join(cmd)}\n{'=' * 72}", flush=True)
    if args.dry_run:
        return True

    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                          timeout=args.timeout)
    if proc.returncode != 0:
        print(f'[WARN] HAR-RV {spec.name}: exited {proc.returncode}; last '
              f'stderr:\n{proc.stderr[-1500:]}', flush=True)
        return False
    for h in args.horizons:
        meta = convert_har(spec, h, args.results_dir)
        print(f"[OK] HAR-RV {spec.name} h={h}: {meta['n_obs']} forecast(s)",
              flush=True)
    return True


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------

def load_failures(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, newline='') as fh:
        for row in csv.DictReader(fh):
            done.add((row['dataset'], int(row['horizon']), row['model'],
                      row['seed']))
    return done


def record_failure(path, dataset, h, model, seed, why):
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'a', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=FAILURE_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({'dataset': dataset, 'horizon': h, 'model': model,
                         'seed': '' if seed is None else seed,
                         'when': time.strftime('%Y-%m-%d %H:%M:%S'),
                         'why': why})


def build_plan(args):
    """(har_datasets, dl_cells) still to run, in dataset -> horizon -> model order."""
    failed = set() if args.retry_failed else load_failures(args.failures_path)

    har_datasets = []
    if HAR_MODEL in args.models:
        for name in args.datasets:
            missing = any(not os.path.exists(
                cell_path(args.results_dir, name, h, HAR_MODEL))
                for h in args.horizons)
            # One fit covers every horizon, so a HAR-RV failure is recorded
            # against horizon 0 -- the whole dataset -- rather than against
            # whichever horizon happened to be missing when it was noticed.
            if missing and (name, HAR_FAILURE_HORIZON, HAR_MODEL, '') not in failed:
                har_datasets.append(name)

    dl_cells = []
    for name in args.datasets:
        for h in args.horizons:
            for model in args.deep_models:
                for seed in args.seeds:
                    path = cell_path(args.results_dir, name, h, model, seed)
                    if os.path.exists(path):
                        continue
                    if (name, h, model, str(seed)) in failed:
                        continue
                    dl_cells.append((name, h, model, seed, path))
    if args.limit:
        dl_cells = dl_cells[:args.limit]
    return har_datasets, dl_cells


# ---------------------------------------------------------------------------
# --validate: build every planned configuration before anything trains
# ---------------------------------------------------------------------------

def validate(args):
    """Build each (model, horizon) config and push one batch through it on CPU.

    A full sweep is hours of GPU time; a configuration the architecture rejects
    should not surface at hour six. Each config is built through run.py's own
    parser, so anything run.py would refuse is refused here too. This catches
    the shape and assertion failures that live in ``__init__`` and ``forward``;
    it cannot catch a failure that only appears during optimisation.

    Only (model, horizon) pairs are checked, not every cell: under
    ``--aggregate_mean`` the head is built with pred_len = 1 whatever h is, and
    the dataset changes the number of rows, never a tensor shape.
    """
    import torch
    os.chdir(REPO_ROOT)
    import run as run_module
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    from utils.timefeatures import time_features_from_frequency_str

    spec = dataset_spec(args.datasets[0])
    bad, checked = [], 0
    for model in args.deep_models:
        for h in args.horizons:
            checked += 1
            try:
                argv, _ = anchor_argv(model, spec, h, args.seeds[0],
                                      anchor_dir=args.anchor_dir,
                                      checkpoints='./checkpoints/_validate',
                                      train_epochs=1, patience=1,
                                      use_gpu=False)
                parsed = run_module.build_parser().parse_args(argv)
                run_module.finalize_args(parsed)
                exp = Exp_Long_Term_Forecast(parsed)
                n_mark = len(time_features_from_frequency_str(parsed.freq))
                b, dec_len = 4, parsed.label_len + parsed.pred_len
                x = torch.randn(b, parsed.seq_len, parsed.enc_in)
                x_mark = torch.randn(b, parsed.seq_len, n_mark)
                dec = torch.zeros(b, dec_len, parsed.dec_in)
                y_mark = torch.randn(b, dec_len, n_mark)
                exp.model.eval()
                with torch.no_grad():
                    exp.model(x, x_mark, dec, y_mark)
            except Exception as exc:            # noqa: BLE001 - report, don't stop
                bad.append((model, h, f'{type(exc).__name__}: {exc}'))
                print(f'[INVALID] {model} h={h}: {type(exc).__name__}: {exc}',
                      flush=True)

    print(f'\nValidated {checked} configuration(s): {checked - len(bad)} build '
          f'and run a batch, {len(bad)} do not.')

    # The row grid is the other thing a sweep can die on hours in, and it costs
    # nothing to check: every dataset must hold a usable test window at h=22.
    for name in args.datasets:
        s = dataset_spec(name)
        counts = {h: len(test_target_dates(s, h)) for h in args.horizons}
        print(f'  {name:<9} {s.asset:<7} test forecasts per horizon: '
              + ', '.join(f'h{h}={n}' for h, n in counts.items()))
    return bad


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def resolve_device(args, parser):
    """Fall back to CPU when there is no CUDA device, loudly.

    Exp_Basic._acquire_device sends the model to 'cuda:0' whenever use_gpu is
    set, without checking availability -- so on a CPU-only box every cell would
    die identically. Better to say so once.
    """
    if not args.use_gpu:
        return
    try:
        import torch
    except ImportError:
        parser.error('torch is not installed; the deep models cannot run. '
                     'Install requirements.txt, or pass --models HAR-RV to '
                     'run only the baseline.')
    if not torch.cuda.is_available():
        args.use_gpu = False
        print('[note] no CUDA device visible; training on CPU. The full grid '
              'is a GPU-scale job -- use --quick / --limit for a smoke test.')


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='Run 10 deep models + HAR-RV over 10 RV datasets at '
                    'h = 1, 5, 22 and store the per-observation forecasts.')
    ap.add_argument('--results_dir', default=DEFAULT_RESULTS_DIR,
                    help='root for runs/, har/, tables/, losses/')
    ap.add_argument('--anchor_dir', default=DEFAULT_ANCHOR_DIR,
                    help='directory holding the tuned <Model>_best.json files. '
                         'A per-dataset <anchor_dir>/<dataset>/<Model>_best.json '
                         'wins over the shared one when it exists')
    ap.add_argument('--models', nargs='+', default=None, metavar='MODEL',
                    help=f'subset of {", ".join(DEEP_MODELS)}, {HAR_MODEL} '
                         f'(default: all of them)')
    ap.add_argument('--datasets', nargs='+', default=None, metavar='NAME',
                    help='subset of the ten datasets (default: all)')
    ap.add_argument('--assets', nargs='+', default=None, choices=['forex', 'crypto'],
                    help='restrict to an asset class')
    ap.add_argument('--horizons', nargs='+', type=int, default=list(HORIZONS),
                    metavar='H', help='forecast horizons (default: 1 5 22)')
    ap.add_argument('--itr', type=int, default=DEFAULT_N_SEEDS, metavar='N',
                    help='repeats per cell; repeat i uses seed (--seed + i), '
                         'exactly run.py\'s rule. Each repeat is its own '
                         'subprocess and its own file, so one can fail or be '
                         'resumed without the other nine')
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED, metavar='SEED',
                    help='base seed for --itr')
    ap.add_argument('--seeds', nargs='+', type=int, default=None,
                    metavar='SEED',
                    help='explicit seed list, overriding --itr/--seed')
    ap.add_argument('--train_epochs', type=int, default=DEFAULT_TRAIN_EPOCHS)
    ap.add_argument('--patience', type=int, default=DEFAULT_PATIENCE)
    ap.add_argument('--checkpoint_dir', default='./checkpoints/benchmark',
                    help='where a cell writes its per-epoch checkpoints; they '
                         'are deleted after the cell unless --keep_checkpoints, '
                         'so point this at local disk when --results_dir lives '
                         'on a network mount such as Google Drive')
    ap.add_argument('--keep_checkpoints', action='store_true')
    ap.add_argument('--no_use_gpu', action='store_false', dest='use_gpu',
                    default=True, help='train on CPU')
    ap.add_argument('--timeout', type=int, default=6 * 3600,
                    help='per-cell subprocess timeout (s)')
    ap.add_argument('--retry_failed', action='store_true',
                    help='re-run cells recorded in failures.csv')
    ap.add_argument('--limit', type=int, default=None,
                    help='stop after this many deep-model cells (smoke tests)')
    ap.add_argument('--quick', action='store_true',
                    help='smoke test: 5 epochs, patience 3')
    ap.add_argument('--dry_run', action='store_true',
                    help='print the plan and the commands, train nothing')
    ap.add_argument('--validate', action='store_true',
                    help='build every planned configuration and push one batch '
                         'through it on CPU, then stop')
    ap.add_argument('--no_aggregate', action='store_true',
                    help='skip the tables/ and losses/ build at the end')

    # --worker: internal. One cell, in this process, then exit. It reuses
    # --seed for the one seed it trains, so there is nothing to keep in sync.
    ap.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--model', help=argparse.SUPPRESS)
    ap.add_argument('--dataset', help=argparse.SUPPRESS)
    ap.add_argument('--horizon', type=int, help=argparse.SUPPRESS)
    ap.add_argument('--out', help=argparse.SUPPRESS)

    args = ap.parse_args(argv)

    if args.quick:
        args.train_epochs, args.patience = 5, 3
    if args.worker:
        resolve_device(args, ap)
        run_worker(args)
        return 0

    args.results_dir = os.path.abspath(args.results_dir)
    args.anchor_dir = os.path.abspath(args.anchor_dir)
    args.failures_path = os.path.join(args.results_dir, 'failures.csv')

    names = args.datasets or dataset_names(args.assets)
    if args.datasets and args.assets:
        names = [n for n in names if dataset_spec(n).asset in set(args.assets)]
    for name in names:
        dataset_spec(name)                      # raises on an unknown name
    args.datasets = names

    unknown = [h for h in args.horizons if h < 1]
    if unknown:
        ap.error(f'horizons must be >= 1, got {unknown}')

    if args.itr < 1:
        ap.error(f'--itr must be >= 1, got {args.itr}')
    args.seeds = args.seeds or seed_list(args.seed, args.itr)

    requested = args.models or list(DEEP_MODELS) + [HAR_MODEL]
    unknown = [m for m in requested
               if m not in DEEP_MODELS and m != HAR_MODEL]
    if unknown:
        ap.error(f"unknown model(s) {unknown}; known: "
                 f"{', '.join(DEEP_MODELS)}, {HAR_MODEL}")
    args.models = requested
    args.deep_models = discover_anchors(
        args.anchor_dir, [m for m in requested if m != HAR_MODEL])
    missing = [m for m in requested
               if m != HAR_MODEL and m not in args.deep_models]
    if missing:
        print(f'[WARN] no <Model>_best.json in {args.anchor_dir} for '
              f'{", ".join(missing)}; those models are skipped. Tune them '
              f'first (tuning/optuna_tune.py) or drop the JSON in place.')

    if args.validate:
        validate(args)                          # always CPU, no device to pick
        return 0

    if args.deep_models:
        resolve_device(args, ap)

    har_datasets, dl_cells = build_plan(args)
    total = len(dl_cells)
    print(f'\nBenchmark plan  ({args.results_dir})')
    print(f'  datasets  : {len(args.datasets)}  [{", ".join(args.datasets)}]')
    print(f'  horizons  : {", ".join(str(h) for h in args.horizons)}')
    print(f'  models    : {len(args.deep_models)} deep'
          + (f' + {HAR_MODEL}' if HAR_MODEL in args.models else '')
          + f'  [{", ".join(args.deep_models)}]')
    seeds = args.seeds
    print(f'  seeds     : {len(seeds)} per cell  '
          f'[{seeds[0]}' + (f'..{seeds[-1]}]' if len(seeds) > 1 else ']'))
    print(f'  to train  : {total} deep-model cell(s), '
          f'{len(har_datasets)} HAR-RV fit(s), '
          f'{args.train_epochs} epochs max each')
    skipped = (len(args.datasets) * len(args.horizons) * len(args.deep_models)
               * len(args.seeds)) - total
    if skipped > 0:
        print(f'  not queued: {skipped} cell(s) already on disk, previously '
              f'failed, or past --limit')

    if args.dry_run:
        for name in har_datasets:
            run_har(dataset_spec(name), args)
        for name, h, model, seed, path in dl_cells:
            print(' '.join(worker_cmd(model, name, h, seed, path, args)))
        print(f'\n[dry-run] nothing was trained. {total} cell(s) planned.')
        return 0

    os.makedirs(args.results_dir, exist_ok=True)

    for name in har_datasets:
        if not run_har(dataset_spec(name), args):
            record_failure(args.failures_path, name, HAR_FAILURE_HORIZON,
                           HAR_MODEL, None, 'HAR-RV_RUN.PY failed')

    for index, (name, h, model, seed, path) in enumerate(dl_cells, start=1):
        cmd = worker_cmd(model, name, h, seed, path, args)
        print(f"\n{'=' * 72}\n[{index}/{total}] {name} h={h} {model} seed={seed}"
              f"\n{' '.join(cmd)}\n{'=' * 72}", flush=True)
        started = time.time()
        try:
            proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True,
                                  text=True, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            why = f'timed out after {args.timeout}s'
            print(f'[WARN] {name} h={h} {model}: {why}', flush=True)
            record_failure(args.failures_path, name, h, model, seed, why)
            continue
        seconds = time.time() - started

        if proc.returncode != 0 or not os.path.exists(path):
            why = f'exit {proc.returncode}'
            print(f'[WARN] {name} h={h} {model} seed={seed}: {why} after '
                  f'{seconds:.0f}s; last stderr:\n{proc.stderr[-1500:]}',
                  flush=True)
            record_failure(args.failures_path, name, h, model, seed, why)
            continue
        print(f'[OK] {name} h={h} {model} seed={seed} ({seconds:.0f}s)',
              flush=True)

    if not args.no_aggregate:
        from orchestrate.aggregate_results import aggregate
        aggregate(args.results_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
