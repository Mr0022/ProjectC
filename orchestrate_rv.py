#!/usr/bin/env python3
"""
Sweep every deep model over both scales and every horizon, then build the
Diebold-Mariano / Model Confidence Set inputs.

    python orchestrate_rv.py --dry-run          # print the commands, run nothing
    python orchestrate_rv.py                    # the full sweep + aggregation
    python orchestrate_rv.py --models DLinear PatchTST --horizons 1
    python orchestrate_rv.py --seeds 2021 2022 2023 2024 2025

The grid is  models x {raw, log} x {1, 5, 22} x seeds  -- 11 x 2 x 3 = 66 runs
at one seed. Each run trains on 2012-2022, early-stops on 2023-2024 and
forecasts 2025 onward, writing one row per test day to forecasts/<tag>.csv.
HAR-RV is swept too (two runs, one per scale, all horizons each), because
without it DM has no benchmark to test against.

Every run is a separate `python run.py` process, so one model blowing up costs
that cell and nothing else; the failure lands in runs_manifest.csv with its log
path and the sweep carries on.

Notes on the grid
-----------------
--aggregate_mean with --pred_len h is what makes a run comparable to HAR: the
model then forecasts the single number Y^(h), the h-day forward mean of RV, on
the same rows and dates HAR uses.

raw and log are not two views of one run, they are two different models: the
series is logged before training, so the network optimises a different
objective and gets a different fit. Both end up scored on the raw variance
scale (the --log forecast via the Jensen correction), which is what lets them
share an MCS.

Seeds: one by default. Neither DM nor MCS sees seed noise -- both treat the
forecasts as fixed -- so a single-seed win can be an artifact of
initialisation. Pass several seeds to enter them as separate MCS entries and
find out.
"""

import argparse
import csv
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

SCALES = ('raw', 'log')
HORIZONS = (1, 5, 22)
DEFAULT_SEED = 2021

# Flags a model needs beyond the shared set, keyed by model name.
#
# TimeMixer mixes ACROSS temporal scales and needs at least two of them; with
# the default down_sampling_layers = 0 run.py refuses to start. Everything else
# reads its extras through getattr defaults and runs on the shared flags alone.
MODEL_FLAGS = {
    'TimeMixer': ['--down_sampling_layers', '3',
                  '--down_sampling_window', '2',
                  '--down_sampling_method', 'avg'],
}


def discover_models(models_dir=None):
    """Every architecture in models/, which is also how exp_basic finds them."""
    models_dir = models_dir or os.path.join(HERE, 'models')
    return sorted(f[:-3] for f in os.listdir(models_dir)
                  if f.endswith('.py') and f != '__init__.py')


def run_tag(model, scale, horizon, seed):
    """Must match utils.forecast_export.run_tag -- the sweep and the exporter
    have to agree on the filename or --resume would re-run everything."""
    return f"{model}_{scale}_h{int(horizon):02d}_s{int(seed)}"


def build_command(model, scale, horizon, seed, args):
    """The `python run.py ...` argv for one cell of the grid."""
    tag = run_tag(model, scale, horizon, seed)
    # model_id must differ per cell: run.py's `setting` string keys the
    # checkpoint and results directories, and without it a raw run and a --log
    # run of the same model and horizon would overwrite each other.
    model_id = f"RV{scale}_h{horizon:02d}_s{seed}"

    cmd = [
        sys.executable, os.path.join(HERE, 'run.py'),
        '--task_name', 'long_term_forecast',
        '--is_training', '1',
        '--model_id', model_id,
        '--model', model,
        '--data', 'custom',
        '--root_path', args.root_path,
        '--data_path', args.data_path,
        '--features', 'S',
        '--target', 'RV',
        '--freq', 'd',
        # single-variate RV series: the model defaults of 7 would build the
        # wrong input/output width
        '--enc_in', '1', '--dec_in', '1', '--c_out', '1',
        '--seq_len', str(args.seq_len),
        '--label_len', str(args.label_len),
        '--pred_len', str(horizon),
        '--aggregate_mean',
        '--des', 'RV',
        '--itr', '1',
        '--train_epochs', str(args.train_epochs),
        '--batch_size', str(args.batch_size),
        '--patience', str(args.patience),
        '--learning_rate', str(args.learning_rate),
        '--d_model', str(args.d_model),
        '--d_ff', str(args.d_ff),
        '--num_workers', str(args.num_workers),
        '--fix_seed', str(seed),
        '--forecast_dir', args.forecast_dir,
        '--run_tag', tag,
    ]
    if scale == 'log':
        cmd.append('--log')
    if args.device == 'cpu':
        cmd.append('--no_use_gpu')
    cmd += MODEL_FLAGS.get(model, [])
    return tag, cmd


def har_commands(args):
    """HAR-RV, once per scale. One run covers all three horizons."""
    cmds = []
    for scale in args.scales:
        cmd = [sys.executable, os.path.join(HERE, 'HAR-RV_RUN.PY'),
               '--data', os.path.join(args.root_path, args.data_path),
               '--outdir', args.har_dir]
        if scale == 'log':
            cmd.append('--log')
        cmds.append((f'HAR-RV_{scale}', cmd))
    return cmds


def execute(tag, cmd, log_dir, timeout):
    """
    Run one command, tee-ing to logs/<tag>.log.

    Failures are returned, never raised: a sweep that dies on its third cell of
    sixty-six is worse than one that records the failure and keeps going.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{tag}.log')
    started = time.time()
    try:
        with open(log_path, 'w') as log:
            log.write(' '.join(cmd) + '\n\n')
            log.flush()
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                  cwd=HERE, timeout=timeout)
        rc, status = proc.returncode, ('ok' if proc.returncode == 0 else 'failed')
    except subprocess.TimeoutExpired:
        rc, status = -1, 'timeout'
    except Exception as exc:                                # noqa: BLE001
        with open(log_path, 'a') as log:
            log.write(f'\norchestrator error: {exc!r}\n')
        rc, status = -2, 'error'
    return status, rc, time.time() - started, log_path


def count_rows(path):
    """Forecast rows a run produced -- the quickest check that it really ran."""
    if not os.path.isfile(path):
        return 0
    with open(path) as f:
        return max(sum(1 for _ in f) - 1, 0)


def main(argv=None):
    all_models = discover_models()

    ap = argparse.ArgumentParser(
        description='Sweep all deep models x {raw,log} x horizons and build '
                    'the DM / MCS loss matrices.')
    ap.add_argument('--models', nargs='+', default=all_models,
                    help=f'default: all {len(all_models)} in models/')
    ap.add_argument('--scales', nargs='+', default=list(SCALES), choices=SCALES)
    ap.add_argument('--horizons', nargs='+', type=int, default=list(HORIZONS))
    ap.add_argument('--seeds', nargs='+', type=int, default=[DEFAULT_SEED])

    ap.add_argument('--root_path', default='./data/')
    ap.add_argument('--data_path', default='EURUSD-RV.csv')
    ap.add_argument('--forecast_dir', default='./forecasts')
    ap.add_argument('--har_dir', default='./har_results')
    ap.add_argument('--log_dir', default='./logs')
    ap.add_argument('--outdir', default='./analysis')

    # Capacity defaults are sized for this dataset -- one univariate series of
    # ~3.7k rows -- not for the multivariate benchmarks the repo defaults
    # (d_model 512, d_ff 2048) were written for.
    ap.add_argument('--seq_len', type=int, default=96,
                    help='look-back window. Does not shift the test dates: the '
                         'split border moves with it, so the first forecast is '
                         'the first test-year row whatever this is.')
    ap.add_argument('--label_len', type=int, default=48)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--d_model', type=int, default=64)
    ap.add_argument('--d_ff', type=int, default=128)
    ap.add_argument('--num_workers', type=int, default=2)

    ap.add_argument('--device', choices=('auto', 'cpu'), default='auto',
                    help="'cpu' forces --no_use_gpu on every run")
    ap.add_argument('--timeout', type=int, default=7200,
                    help='per-run wall-clock cap in seconds')
    ap.add_argument('--dry-run', action='store_true',
                    help='print the grid and the commands, execute nothing')
    ap.add_argument('--force', action='store_true',
                    help='re-run cells whose forecast CSV already exists '
                         '(default is to skip them, so an interrupted sweep '
                         'resumes where it stopped)')
    ap.add_argument('--skip-har', action='store_true',
                    help='deep models only. DM then has no benchmark; MCS still runs.')
    ap.add_argument('--skip-aggregate', action='store_true',
                    help='leave the per-run CSVs alone, build no matrices')
    args = ap.parse_args(argv)

    unknown = sorted(set(args.models) - set(all_models))
    if unknown:
        ap.error(f'unknown model(s): {unknown}. Available: {all_models}')

    grid = [(m, s, h, sd)
            for m in args.models
            for s in args.scales
            for h in args.horizons
            for sd in args.seeds]

    print('=' * 72)
    print(f'SWEEP: {len(args.models)} models x {len(args.scales)} scales x '
          f'{len(args.horizons)} horizons x {len(args.seeds)} seed(s) '
          f'= {len(grid)} runs')
    print(f'  models    : {" ".join(args.models)}')
    print(f'  scales    : {" ".join(args.scales)}')
    print(f'  horizons  : {" ".join(str(h) for h in args.horizons)}')
    print(f'  seeds     : {" ".join(str(s) for s in args.seeds)}')
    print(f'  forecasts : {args.forecast_dir}')
    print('=' * 72)

    if args.dry_run:
        for model, scale, horizon, seed in grid:
            tag, cmd = build_command(model, scale, horizon, seed, args)
            print(f'\n[{tag}]\n  ' + ' '.join(cmd))
        if not args.skip_har:
            for tag, cmd in har_commands(args):
                print(f'\n[{tag}]\n  ' + ' '.join(cmd))
        print(f'\n{len(grid)} run(s) would execute. Nothing was run.')
        return 0

    os.makedirs(args.forecast_dir, exist_ok=True)
    os.makedirs(args.har_dir, exist_ok=True)

    manifest, t0 = [], time.time()

    for i, (model, scale, horizon, seed) in enumerate(grid, 1):
        tag, cmd = build_command(model, scale, horizon, seed, args)
        out_csv = os.path.join(args.forecast_dir, f'{tag}.csv')

        if os.path.isfile(out_csv) and not args.force:
            print(f'[{i:>3}/{len(grid)}] {tag:<40} skipped (exists)')
            manifest.append(dict(tag=tag, model=model, scale=scale,
                                 horizon=horizon, seed=seed, status='skipped',
                                 returncode=0, seconds=0.0,
                                 n_forecasts=count_rows(out_csv), log=''))
            continue

        print(f'[{i:>3}/{len(grid)}] {tag:<40} running...', end='', flush=True)
        status, rc, secs, log_path = execute(tag, cmd, args.log_dir, args.timeout)
        n_rows = count_rows(out_csv)
        # A zero-exit run that wrote no forecasts has still failed for our
        # purposes -- there is nothing for DM or MCS to consume.
        if status == 'ok' and n_rows == 0:
            status = 'no-output'
        print(f' {status} ({secs:.0f}s, {n_rows} rows)')
        if status != 'ok':
            print(f'          see {log_path}')
        manifest.append(dict(tag=tag, model=model, scale=scale, horizon=horizon,
                             seed=seed, status=status, returncode=rc,
                             seconds=round(secs, 1), n_forecasts=n_rows,
                             log=log_path))

    if not args.skip_har:
        for tag, cmd in har_commands(args):
            print(f'[HAR] {tag:<46} running...', end='', flush=True)
            status, rc, secs, log_path = execute(tag, cmd, args.log_dir, args.timeout)
            print(f' {status} ({secs:.0f}s)')
            if status != 'ok':
                print(f'          see {log_path}')
            manifest.append(dict(tag=tag, model='HAR-RV',
                                 scale=tag.rsplit('_', 1)[-1], horizon='all',
                                 seed=0, status=status, returncode=rc,
                                 seconds=round(secs, 1), n_forecasts='',
                                 log=log_path))

    man_path = os.path.join(HERE, 'runs_manifest.csv')
    with open(man_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)

    ok = sum(1 for r in manifest if r['status'] in ('ok', 'skipped'))
    bad = [r for r in manifest if r['status'] not in ('ok', 'skipped')]
    print('\n' + '=' * 72)
    print(f'SWEEP DONE in {(time.time() - t0) / 60:.1f} min: '
          f'{ok} ok/skipped, {len(bad)} failed  ->  {man_path}')
    for r in bad:
        print(f'  FAILED  {r["tag"]:<40} {r["status"]} (rc={r["returncode"]}) '
              f'{r["log"]}')
    print('=' * 72)

    if args.skip_aggregate:
        return 0 if not bad else 1

    print('\nBuilding DM / MCS inputs...\n')
    import aggregate_forecasts
    agg_argv = ['--forecast-dir', args.forecast_dir,
                '--har-dir', args.har_dir,
                '--outdir', args.outdir,
                '--data', os.path.join(args.root_path, args.data_path)]
    if args.skip_har:
        agg_argv.append('--skip-har')
    rc = aggregate_forecasts.main(agg_argv)
    return rc or (1 if bad else 0)


if __name__ == '__main__':
    raise SystemExit(main())
