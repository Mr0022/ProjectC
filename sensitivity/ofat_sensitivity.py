#!/usr/bin/env python3
"""OFAT (one-factor-at-a-time) hyper-parameter sensitivity sweeps.

Every model is anchored at the configuration Optuna selected for it -- the
``best_params`` block of ``<Model>_best.json`` -- and ONE hyper-parameter at a
time is swept over the grid ``sensitivity/ofat_grids.py`` derives from the
tuning search space. Each swept point is trained through ``run.py --itr 5``,
so it carries a five-seed mean +/- std, and the per-seed test metrics are
parsed from run.py's stdout into a tidy long-format CSV that
``sensitivity/ofat_plots.py`` turns into the figures.

    python sensitivity/ofat_sensitivity.py                       # every model
    python sensitivity/ofat_sensitivity.py --models DLinear FITS
    python sensitivity/ofat_sensitivity.py --params learning_rate batch_size
    python sensitivity/ofat_sensitivity.py --quick                # 2 seeds x 5 epochs
    python sensitivity/ofat_sensitivity.py --dry_run              # print the plan

Design notes
------------
* **The anchor comes from the JSON, never from optuna.db.** Each
  ``<Model>_best.json`` carries both the winning parameters and the exact
  command line that produced them. The searched flags become the anchor of the
  sweep; everything else in that command -- dataset, protocol, epochs,
  patience, ``--aggregate_mean --log`` -- is reused VERBATIM as the base, so an
  OFAT point differs from the tuned run in exactly the one knob under study
  and nothing else. Dropping a new ``<Model>_best.json`` into the results
  directory is all it takes to sweep another model.

* **The anchor is trained once per model** and reused as the centre point of
  every panel, rather than retrained for each of the dozen knobs.

* **Local sensitivity.** The curves are only valid AROUND the optimum -- that
  is what OFAT is, and the figures say so.

* **Resumable.** Any (model, param, value) already in the CSV is skipped, so an
  interrupted sweep restarts where it left off; a point whose training failed
  is recorded in ``<out>.failures.csv`` and skipped too, unless
  ``--retry_failed`` is passed.

* **Read against seed noise.** Five seeds per point exist so a curve can be
  compared with the spread of the anchor: a knob that moves the metric by less
  than the seed noise has not been shown to matter. ``ofat_plots.py`` draws
  that band on every panel.

Run from the repository root -- ``exp_basic`` discovers models by scanning the
relative path ``models/``.
"""

import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sensitivity.ofat_grids import (MODELS, coupled_changes, grids, resolve,
                                    value_key)

# Columns of the results CSV. One row per (model, param, value, seed).
FIELDS = ['model', 'param', 'value', 'coupled', 'seed', 'val_loss',
          'mse', 'mae', 'qlike', 'mse_rv', 'mae_rv', 'scale',
          'itr', 'epochs', 'seconds']

# Metrics parsed out of the HAR-comparable block Exp_Long_Term_Forecast.test()
# prints under --aggregate_mean. Under --log the block reports MSE/MAE on the
# ln(RV) scale plus the back-transformed MSE_RV/MAE_RV; without it there is a
# single raw-RV pair and no _RV columns. MSE_RV must not match the plain MSE
# pattern, hence the anchored, colon-terminated forms.
_METRIC_RE = {
    'mse': re.compile(r'^\s*MSE\s*(?:\[ln\])?\s*:\s*([-\d.eE+]+)', re.M),
    'mae': re.compile(r'^\s*MAE\s*(?:\[ln\])?\s*:\s*([-\d.eE+]+)', re.M),
    'qlike': re.compile(r'^\s*QLIKE\s*(?:\[RV\])?\s*:\s*([-\d.eE+]+)', re.M),
    'mse_rv': re.compile(r'^\s*MSE_RV\s*:\s*([-\d.eE+]+)', re.M),
    'mae_rv': re.compile(r'^\s*MAE_RV\s*:\s*([-\d.eE+]+)', re.M),
}
_SCALE_RE = re.compile(r'HAR-COMPARABLE TEST METRICS\s*\[(\w+)\]')
_VALI_RE = re.compile(r'Vali Loss:\s*([-\d.eE+]+)')
# run.py announces each repeat before it trains, which is what splits a
# multi-seed stdout into per-seed sections.
_RUN_RE = re.compile(r'>{3,}start training \(run \d+/\d+, seed (\d+)\)')

# Flags the runner owns: whatever the tuning command said about them is
# replaced, so they are stripped from the base.
_RUNNER_FLAGS = {'model_id', 'des', 'itr', 'seed', 'checkpoints',
                 'train_epochs', 'patience'}


# ---------------------------------------------------------------------------
# Reading the tuned anchors
# ---------------------------------------------------------------------------
def _unwrap(value):
    """[2] -> 2. ModernTCN's per-stage backbone lists are one element long."""
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def parse_command(command):
    """Split a ``python -u run.py --a 1 --b`` string into {flag: [values]}.

    Bare switches map to an empty list, ``nargs='+'`` flags to several values.
    A token counts as a flag only if it starts with ``--`` and is not a
    negative number, so a value like ``-1`` is never mistaken for one.
    """
    tokens = command.split()
    if tokens[:1] == ['python']:
        tokens = tokens[1:]
    while tokens and (tokens[0].startswith('-') or tokens[0].endswith('.py')):
        if tokens[0].startswith('--'):
            break
        tokens = tokens[1:]

    flags, current = {}, None
    for token in tokens:
        is_flag = token.startswith('--') and not _is_number(token[2:])
        if is_flag:
            current = token[2:]
            flags[current] = []
        elif current is not None:
            flags[current].append(token)
    return flags


def _is_number(text):
    try:
        float(text)
    except ValueError:
        return False
    return True


def load_anchor(path):
    """Read one ``<Model>_best.json`` into (model, anchor, base).

    ``anchor`` holds the searched flags with their JSON types intact; ``base``
    holds every other flag of the winning command line, which the sweep reuses
    unchanged. Splitting them this way is what guarantees an OFAT point differs
    from the tuned run in one knob only.
    """
    with open(path) as fh:
        best = json.load(fh)
    anchor = {k: _unwrap(v) for k, v in best['best_params'].items()}
    base = parse_command(best['command'])
    for flag in set(anchor) | _RUNNER_FLAGS:
        base.pop(flag, None)
    return best['model'], anchor, base, best


def discover(results_dir, wanted=None):
    """All ``<Model>_best.json`` files in `results_dir`, in MODELS order."""
    found = {}
    for path in sorted(glob.glob(os.path.join(results_dir, '*_best.json'))):
        model = os.path.basename(path)[:-len('_best.json')]
        if model in MODELS:
            found[model] = path
    if wanted:
        missing = [m for m in wanted if m not in found]
        if missing:
            print(f'[WARN] no <Model>_best.json for {", ".join(missing)} in '
                  f'{results_dir}; skipping (drop the file in and re-run).')
        found = {m: p for m, p in found.items() if m in wanted}
    return [(m, found[m]) for m in MODELS if m in found]


# ---------------------------------------------------------------------------
# Building and running one point
# ---------------------------------------------------------------------------
def _emit(flag, value):
    argv = [f'--{flag}']
    if isinstance(value, (list, tuple)):
        argv += [str(v) for v in value]
    elif isinstance(value, float):
        argv.append(repr(value))
    elif isinstance(value, bool):
        return argv if value else []
    else:
        argv.append(str(value))
    return argv


def build_cmd(base, cfg, model_id, args, checkpoints):
    """The run.py command line for one OFAT point."""
    argv = [sys.executable, '-u', 'run.py']
    for flag, values in base.items():
        argv.append(f'--{flag}')
        argv.extend(values)
    for flag, value in cfg.items():
        argv.extend(_emit(flag, value))
    argv += ['--model_id', model_id, '--des', 'ofat',
             '--itr', str(args.itr), '--seed', str(args.seed),
             '--train_epochs', str(args.train_epochs),
             '--patience', str(args.patience),
             '--checkpoints', checkpoints]
    if not args.use_gpu:
        # Exp_Basic reads use_gpu directly, so a CPU-only box needs the flag
        # even though run.py's own device pick would have fallen back.
        argv.append('--no_use_gpu')
    return argv


def parse_metrics(stdout):
    """One dict per seed, in the order run.py trained them.

    A section is everything between two ``start training`` banners, so a repeat
    that crashed after training contributes no metrics and is simply absent --
    the caller checks the count against --itr rather than pairing by position.
    """
    marks = list(_RUN_RE.finditer(stdout))
    rows = []
    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(stdout)
        section = stdout[mark.end():end]
        scale = _SCALE_RE.search(section)
        if scale is None:
            continue                      # this repeat never reached its test
        row = {'seed': int(mark.group(1)), 'scale': scale.group(1)}
        for name, pattern in _METRIC_RE.items():
            hit = pattern.search(section)
            row[name] = float(hit.group(1)) if hit else ''
        vali = [float(v) for v in _VALI_RE.findall(section)]
        row['val_loss'] = min(vali) if vali else ''
        rows.append(row)
    return rows


def run_point(model, param, value, cfg, coupled, base, args, tag):
    """Train one OFAT point: per-seed rows, None if it failed, [] on a dry run."""
    model_id = f'ofat_{model}_{tag}'
    checkpoints = os.path.join(args.checkpoint_dir, model_id)
    cmd = build_cmd(base, cfg, model_id, args, checkpoints)

    label = 'anchor' if param is None else f'{param}={value_key(value)}'
    print(f"\n{'=' * 72}\n[OFAT] {model}  {label}"
          + (f'   (coupled: {coupled})' if coupled else '')
          + f"\n{' '.join(cmd)}\n{'=' * 72}", flush=True)
    if args.dry_run:
        return []

    started = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              cwd=REPO_ROOT, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        print(f'[WARN] {model} {label}: timed out after {args.timeout}s', flush=True)
        return None
    finally:
        if not args.keep_checkpoints:
            shutil.rmtree(checkpoints, ignore_errors=True)
    seconds = round(time.time() - started, 1)

    if proc.returncode != 0:
        print(f'[WARN] {model} {label}: run.py exited {proc.returncode} after '
              f'{seconds:.0f}s; last stderr:\n{proc.stderr[-1200:]}', flush=True)
        return None

    metrics = parse_metrics(proc.stdout)
    if len(metrics) < args.itr:
        print(f'[WARN] {model} {label}: parsed {len(metrics)}/{args.itr} seed '
              f'metrics; treating the point as failed.\n{proc.stdout[-1200:]}',
              flush=True)
        return None

    rows = []
    for m in metrics:
        rows.append({'model': model,
                     'param': param or 'anchor',
                     'value': '' if param is None else value_key(value),
                     'coupled': coupled,
                     'seed': m['seed'],
                     'val_loss': m['val_loss'],
                     'mse': m['mse'], 'mae': m['mae'], 'qlike': m['qlike'],
                     'mse_rv': m['mse_rv'], 'mae_rv': m['mae_rv'],
                     'scale': m['scale'],
                     'itr': args.itr, 'epochs': args.train_epochs,
                     'seconds': seconds})
    mean_mse = sum(float(r['mse']) for r in rows) / len(rows)
    print(f'[OK] {model} {label}: {len(rows)} seeds, mean MSE={mean_mse:.6f} '
          f'({seconds:.0f}s)', flush=True)
    return rows


# ---------------------------------------------------------------------------
# CSV bookkeeping
# ---------------------------------------------------------------------------
def validate(plans, itr_note=True):
    """Build every planned configuration and push one batch through it.

    A sweep is a day of GPU time; a configuration the architecture rejects
    should not be discovered at hour six. This constructs each point's model
    through run.py's own parser -- so anything run.py would refuse is refused
    here too -- and runs a single forward pass on CPU with a dummy batch shaped
    like the loader's. It catches the shape and assertion failures that live in
    __init__ and forward, which is where the clamped corners of a search space
    fail; it cannot catch a failure that only appears during optimisation.
    """
    import torch                                   # local: --validate only

    os.chdir(REPO_ROOT)                            # exp_basic scans models/
    import run as run_module
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    from utils.timefeatures import time_features_from_frequency_str

    bad, checked = [], 0
    for model, base, points, _ in plans:
        for param, value, cfg, coupled in points:
            checked += 1
            label = 'anchor' if param is None else f'{param}={value_key(value)}'
            argv = build_cmd(base, cfg, f'validate_{model}', _ValidateArgs(),
                             './checkpoints/_validate')[3:]
            try:
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
            except Exception as exc:               # noqa: BLE001 - report, don't stop
                bad.append((model, label, f'{type(exc).__name__}: {exc}'))
                print(f'[INVALID] {model} {label}: {type(exc).__name__}: {exc}',
                      flush=True)

    print(f'\nValidated {checked} configuration(s): {checked - len(bad)} build and '
          f'run a batch, {len(bad)} do not.')
    if bad:
        print('These would be recorded as failures by the sweep:')
        for model, label, why in bad:
            print(f'  {model:<14} {label:<34} {why}')
    elif itr_note:
        print('Nothing in the plan is rejected by the architecture; the sweep '
              'can be started.')
    return bad


class _ValidateArgs:
    """The runner flags build_cmd needs, with training turned down to nothing."""
    itr, seed, train_epochs, patience, use_gpu = 1, 2021, 1, 1, False


def load_done(path):
    """(model, param, value) triples already recorded."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, newline='') as fh:
        for row in csv.DictReader(fh):
            done.add((row['model'], row['param'], row['value']))
    return done


def append_rows(path, rows):
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerows(rows)


def record_failure(path, model, param, value, cfg):
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as fh:
        writer = csv.writer(fh)
        if new:
            writer.writerow(['model', 'param', 'value', 'config', 'when'])
        writer.writerow([model, param or 'anchor',
                         '' if param is None else value_key(value),
                         json.dumps(cfg, sort_keys=True),
                         time.strftime('%Y-%m-%d %H:%M:%S')])


def load_failures(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, newline='') as fh:
        for row in csv.DictReader(fh):
            done.add((row['model'], row['param'], row['value']))
    return done


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------
def plan_for(model, anchor, params):
    """[(param, value, cfg, coupled)] for one model, anchor point first.

    A grid value that CLAMPS back onto the anchor -- TimeMixer's
    down_sampling_layers 3 under window 4, say, where 4**3 does not divide 96 --
    is dropped rather than trained: it would retrain the anchor under a label
    claiming a value the model never received.
    """
    anchor_cfg = resolve(model, anchor)
    plan = [(None, None, anchor_cfg, '')]
    for param, values in grids(model, anchor).items():
        if params and param not in params:
            continue
        seen = {json.dumps(anchor_cfg, sort_keys=True)}
        for value in values:
            cfg = resolve(model, anchor, param, value)
            key = json.dumps(cfg, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            changes = coupled_changes(anchor_cfg, cfg, param)
            # A flag the point DROPS (TimeMixer's moving_avg once the
            # decomposition is the DFT one) reads as "off" rather than "None".
            coupled = ';'.join(f'{k}={"off" if v is None else v}'
                               for k, v in changes.items())
            plan.append((param, value, cfg, coupled))
    return plan


def main():
    ap = argparse.ArgumentParser(
        description='OFAT hyper-parameter sensitivity sweeps, seq_len=96 / h=1')
    ap.add_argument('--models', nargs='+', default=None, metavar='MODEL',
                    help='models to sweep (default: every <Model>_best.json in '
                         '--results_dir)')
    ap.add_argument('--params', nargs='+', default=None, metavar='FLAG',
                    help='restrict the sweep to these hyper-parameters')
    ap.add_argument('--results_dir', default='./tuning/ProjectC_tuning',
                    help='directory holding the <Model>_best.json anchors')
    ap.add_argument('--out', default='./sensitivity/ofat_results.csv')
    ap.add_argument('--itr', type=int, default=5,
                    help='seeds per point (run.py --itr). Five gives the '
                         'mean +/- std a sensitivity curve has to be read with; '
                         'three is the affordable minimum')
    ap.add_argument('--seed', type=int, default=2021,
                    help='base seed; repeat i uses --seed + i')
    ap.add_argument('--train_epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=7)
    ap.add_argument('--checkpoint_dir', default='./checkpoints/ofat',
                    help='where a point writes its per-epoch checkpoints. They '
                         'are deleted after the point unless --keep_checkpoints, '
                         'so point this at local disk when --out lives on a '
                         'network mount such as Google Drive')
    ap.add_argument('--keep_checkpoints', action='store_true')
    ap.add_argument('--no_use_gpu', action='store_false', dest='use_gpu',
                    default=True, help='train on CPU (a sweep on CPU is only '
                                       'realistic as a smoke test)')
    ap.add_argument('--timeout', type=int, default=6 * 3600,
                    help='per-point subprocess timeout (s)')
    ap.add_argument('--retry_failed', action='store_true',
                    help='re-run points recorded in <out>.failures.csv')
    ap.add_argument('--quick', action='store_true',
                    help='smoke test: 2 seeds, 5 epochs, patience 3')
    ap.add_argument('--dry_run', action='store_true',
                    help='print the plan and the commands, train nothing')
    ap.add_argument('--validate', action='store_true',
                    help='build every planned configuration and push one batch '
                         'through it on CPU, then stop. Catches the corners the '
                         'architecture rejects in seconds instead of at hour six '
                         'of a sweep')
    args = ap.parse_args()

    if args.quick:
        args.itr, args.train_epochs, args.patience = 2, 5, 3

    if args.models:
        unknown = [m for m in args.models if m not in MODELS]
        if unknown:
            ap.error(f"unknown model(s) {unknown}; known: {', '.join(MODELS)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    failures_path = args.out + '.failures.csv'
    done = load_done(args.out)
    skip = set() if args.retry_failed else load_failures(failures_path)

    anchors = discover(args.results_dir, args.models)
    if not anchors:
        ap.error(f'no <Model>_best.json files found in {args.results_dir}')

    # Build the whole plan first, so the cost is known before anything trains.
    plans, total = [], 0
    for model, path in anchors:
        _, anchor, base, best = load_anchor(path)
        points = [p for p in plan_for(model, anchor, args.params)
                  if (model, p[0] or 'anchor',
                      '' if p[0] is None else value_key(p[1])) not in done
                  and (model, p[0] or 'anchor',
                       '' if p[0] is None else value_key(p[1])) not in skip]
        plans.append((model, base, points, anchor))
        total += len(points)

    print(f'\nOFAT plan: {total} configuration(s) to train, {args.itr} seed(s) '
          f'each ({total * args.itr} trainings, {args.train_epochs} epochs max)')
    for model, _, points, anchor in plans:
        swept = sorted({p[0] for p in points if p[0]})
        print(f'  {model:<14} {len(points):>3} point(s)'
              + (f'   [{", ".join(swept)}]' if swept else '   [up to date]'))
    if done:
        print(f'  already recorded: {len(done)} point(s)')
    if skip:
        print(f'  previously failed, skipped: {len(skip)} point(s) '
              f'(--retry_failed to re-run)')

    if args.validate:
        validate(plans)
        return

    if args.dry_run:
        for model, base, points, _ in plans:
            for param, value, cfg, coupled in points:
                tag = 'anchor' if param is None else _tag(param, value)
                run_point(model, param, value, cfg, coupled, base, args, tag)
        print('\n[dry-run] nothing was trained.')
        return

    index = 0
    for model, base, points, _ in plans:
        for param, value, cfg, coupled in points:
            index += 1
            print(f'\n########## [{index}/{total}] {model} ##########')
            tag = 'anchor' if param is None else _tag(param, value)
            rows = run_point(model, param, value, cfg, coupled, base, args, tag)
            if rows:
                append_rows(args.out, rows)
            else:
                record_failure(failures_path, model, param, value, cfg)

    print(f'\nDone. Results in {args.out}')
    print('Next:  python sensitivity/ofat_plots.py')


def _tag(param, value):
    """Filesystem-safe label for a swept point (goes into run.py --model_id)."""
    return f'{param}_{value_key(value)}'.replace('.', 'p').replace('-', 'm')


if __name__ == '__main__':
    main()
