"""Optuna driver for the ten long-term forecasters.

    python tuning/optuna_tune.py --model DLinear --n_trials 50
    python tuning/optuna_tune.py --model all --n_trials 50 --train_epochs 30

Selection is on VALIDATION loss -- specifically the smallest validation loss
seen during training, which is exactly the epoch EarlyStopping checkpoints, so
the number being minimised is the one the saved model achieves. Each trial is
trained --n_seeds times (3 by default, i.e. run.py's --itr applied during the
search) and scored by the MEAN of those runs, because a single run on a
519-window validation split is too noisy to rank configurations on. The test
split is never read during a study; --retrain_best afterwards trains the
winning configuration with --itr 5 and prints the HAR-comparable test metrics.

Configurations are built by run.py's own parser, so every trial corresponds to
a `python -u run.py ...` command line, which is written into the results JSON
alongside the parameters.

Run from the repository root: exp_basic discovers models by scanning the
relative path models/.
"""

import argparse
import json
import os
import shutil
import sys
import time

# Import run.py from the repository root regardless of the caller's cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import optuna
import torch

import run as run_module
from tuning.search_spaces import FIXED_PROTOCOL, MODELS, suggest

# The ten models all live under long-term forecasting.
TASK = 'long_term_forecast'


def _flags(params):
    """Turn a {flag: value} mapping into an argv fragment.

    List values become space-separated arguments (run.py declares the
    ModernTCN backbone lists with nargs='+'), booleans become bare switches.
    """
    argv = []
    for key, value in params.items():
        if isinstance(value, bool):
            if value:
                argv.append(f'--{key}')
            continue
        argv.append(f'--{key}')
        if isinstance(value, (list, tuple)):
            argv.extend(str(v) for v in value)
        else:
            argv.append(str(value))
    return argv


def base_argv(args, model, model_id):
    """The part of the command line that every trial of a study shares."""
    argv = [
        '--task_name', TASK,
        '--is_training', '1',
        '--model', model,
        '--model_id', model_id,
        '--data', args.data,
        '--root_path', args.root_path,
        '--data_path', args.data_path,
        '--features', args.features,
        '--target', args.target,
        '--freq', args.freq,
        '--seq_len', str(FIXED_PROTOCOL['seq_len']),
        '--label_len', str(FIXED_PROTOCOL['label_len']),
        '--pred_len', str(FIXED_PROTOCOL['pred_len']),
        '--enc_in', str(args.enc_in),
        '--dec_in', str(args.enc_in),
        '--c_out', str(args.enc_in),
        '--train_epochs', str(args.train_epochs),
        '--patience', str(args.patience),
        '--num_workers', str(args.num_workers),
        '--itr', '1',
        '--des', 'optuna',
    ]
    if args.aggregate_mean:
        argv.append('--aggregate_mean')
    if args.log:
        argv.append('--log')
    if not args.use_gpu:
        argv.append('--no_use_gpu')
    return argv


def build_args(argv):
    """Parse an argv fragment with run.py's parser and apply its cross-flag rules."""
    parser = run_module.build_parser()
    parsed = parser.parse_args(argv)
    run_module.finalize_args(parsed)
    return parsed


class _ReportingEarlyStopping:
    """EarlyStopping that also feeds each epoch's validation loss to Optuna.

    exp.train() constructs its stopper internally and returns only the trained
    model, so there is no other seam to observe the epoch-by-epoch curve from.
    Swapping the name the module resolves gives the pruner a per-epoch signal
    without editing exp/, and the wrapped instance is the untouched original,
    so checkpointing and the stopping rule behave exactly as in a normal run.

    Pruning raises optuna.TrialPruned from inside the training loop; the
    objective catches it after cleaning up the trial's checkpoint directory.
    """

    real_cls = None      # set by _patch_early_stopping
    trial = None         # set per trial
    enabled = True
    losses = []          # reset per training run by train_once

    def __init__(self, *a, **kw):
        self._inner = type(self).real_cls(*a, **kw)
        self._epoch = 0

    def __call__(self, val_loss, model, path):
        self._epoch += 1
        type(self).losses.append(float(val_loss))
        self._inner(val_loss, model, path)
        trial = type(self).trial
        if trial is not None and type(self).enabled:
            trial.report(float(val_loss), self._epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    def __getattr__(self, name):
        # early_stop, counter, val_loss_min, ... all live on the wrapped object.
        return getattr(self._inner, name)


def _patch_early_stopping(exp_module):
    if _ReportingEarlyStopping.real_cls is None:
        _ReportingEarlyStopping.real_cls = exp_module.EarlyStopping
        exp_module.EarlyStopping = _ReportingEarlyStopping


def train_once(cli_args, seed, checkpoint_root):
    """Train one configuration and return its best validation loss.

    The value returned is the minimum over epochs, i.e. the loss of the
    checkpoint EarlyStopping keeps -- not the last epoch's, which is usually
    worse and would reward configurations that happen to stop cleanly.
    """
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    import exp.exp_long_term_forecasting as exp_module

    _patch_early_stopping(exp_module)
    _ReportingEarlyStopping.losses = []

    cli_args.checkpoints = checkpoint_root
    run_module.set_seed(seed)
    exp = Exp_Long_Term_Forecast(cli_args)
    setting = f'optuna_{cli_args.model}_{cli_args.model_id}_seed{seed}'
    try:
        exp.train(setting)
    finally:
        if cli_args.use_gpu and cli_args.gpu_type == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()

    losses = _ReportingEarlyStopping.losses
    if not losses:
        raise RuntimeError('training produced no validation losses')
    return float(np.min(losses))


def make_objective(args, model, checkpoint_root):
    def objective(trial):
        params = suggest(model, trial)
        argv = base_argv(args, model, f'tune_{model}_t{trial.number}') + _flags(params)
        trial.set_user_attr('params', {k: v for k, v in params.items()})
        trial.set_user_attr('command', 'python -u run.py ' + ' '.join(argv))

        try:
            cli_args = build_args(argv)
        except (SystemExit, ValueError) as exc:
            # A configuration run.py itself would refuse. Prune rather than
            # fail so one bad corner does not end the study.
            trial.set_user_attr('rejected', str(exc))
            raise optuna.TrialPruned()

        _ReportingEarlyStopping.trial = trial
        scores = []
        try:
            for i in range(args.n_seeds):
                # Repeats share the trial's pruning stream, so only the first
                # seed reports intermediate values; later ones would restart
                # the epoch counter and confuse the pruner. A trial pruned on
                # its first seed therefore never pays for the rest, which is
                # what keeps repeated training affordable.
                _ReportingEarlyStopping.enabled = (i == 0)
                scores.append(train_once(cli_args, args.seed + i, checkpoint_root))
        except optuna.TrialPruned:
            raise
        except (RuntimeError, ValueError, AssertionError, IndexError) as exc:
            # Shape/assert failures from a corner of the space: record and
            # prune, so the study keeps going and the cause stays visible.
            trial.set_user_attr('error', f'{type(exc).__name__}: {exc}')
            raise optuna.TrialPruned()
        finally:
            _ReportingEarlyStopping.trial = None
            _ReportingEarlyStopping.enabled = True
            shutil.rmtree(checkpoint_root, ignore_errors=True)

        # The trial's score is the MEAN over its seeds. The per-seed values and
        # their spread are kept as attributes so a suspiciously close ranking
        # can be checked against how noisy each configuration actually was.
        trial.set_user_attr('seed_scores', [round(s, 6) for s in scores])
        if len(scores) > 1:
            trial.set_user_attr('seed_std', round(float(np.std(scores, ddof=1)), 6))
        return float(np.mean(scores))

    return objective


def tune_model(args, model):
    checkpoint_root = os.path.join(args.checkpoint_dir or
                                   os.path.join(args.out_dir, '_checkpoints'), model)
    storage = f'sqlite:///{os.path.join(args.out_dir, "optuna.db")}'
    study = optuna.create_study(
        study_name=f'{args.study_prefix}_{model}',
        storage=storage,
        direction='minimize',
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed, multivariate=True),
        # No pruning before the model has had a chance to warm up: with
        # lradj=type1 the first epochs of a good configuration can look worse
        # than a bad one that started flat.
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=args.n_startup_trials, n_warmup_steps=args.n_warmup_steps),
    )

    # --n_trials is a TARGET for the study, not a batch size. Optuna's own
    # n_trials counts trials run by this call, so a resumed session would add
    # a second full batch on top of whatever the database already holds -- and
    # re-search every model that had already finished. Count what is there and
    # ask only for the remainder, which makes re-running the cell idempotent.
    #
    # Attempted, not completed, is the right thing to count: a pruned trial
    # spent compute and taught the sampler something, exactly as it would have
    # inside a single uninterrupted call of --n_trials.
    finished = (optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.PRUNED,
                optuna.trial.TrialState.FAIL)
    attempted = sum(t.state in finished for t in study.trials)
    # Trials the previous session was midway through when it was killed stay
    # RUNNING for ever. They produced no value, so they do not count towards
    # the target; they are reported so the number is not a mystery later.
    orphaned = sum(t.state == optuna.trial.TrialState.RUNNING for t in study.trials)
    remaining = max(0, args.n_trials - attempted)

    if attempted or orphaned:
        note = f'[{model}] study already holds {attempted} trials'
        if orphaned:
            note += f' (+{orphaned} left RUNNING by an interrupted session, ignored)'
        print(f'{note}; target {args.n_trials} -> running {remaining} more')

    started = time.time()
    if remaining:
        study.optimize(make_objective(args, model, checkpoint_root),
                       n_trials=remaining, timeout=args.timeout,
                       gc_after_trial=True)
    else:
        # Already at the target. Fall through anyway so the results file is
        # rewritten -- a session killed mid-search leaves the database correct
        # but the JSON stale or missing.
        print(f'[{model}] already at the {args.n_trials}-trial target; '
              f'refreshing the results file only')
    shutil.rmtree(checkpoint_root, ignore_errors=True)

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print(f'[{model}] no trial completed; nothing to record')
        return None

    best = study.best_trial
    result = {
        'model': model,
        'seq_len': FIXED_PROTOCOL['seq_len'],
        'pred_len': FIXED_PROTOCOL['pred_len'],
        'label_len': FIXED_PROTOCOL['label_len'],
        'best_val_loss': best.value,
        'n_seeds': args.n_seeds,
        'best_val_loss_per_seed': best.user_attrs.get('seed_scores'),
        'best_val_loss_std': best.user_attrs.get('seed_std'),
        'best_params': best.user_attrs.get('params', best.params),
        'command': best.user_attrs.get('command'),
        'n_trials': len(study.trials),
        'n_complete': len(completed),
        'n_pruned': sum(t.state == optuna.trial.TrialState.PRUNED
                        for t in study.trials),
        'seconds': round(time.time() - started, 1),
    }
    path = os.path.join(args.out_dir, f'{model}_best.json')
    with open(path, 'w') as fh:
        json.dump(result, fh, indent=2)

    print(f'\n[{model}] best val loss {best.value:.6f} over {len(completed)} '
          f'completed trials -> {path}')
    print(f'[{model}] {result["command"]}\n')
    return result


def retrain_best(args, result):
    """Re-run the winning configuration through run.py's own main().

    Goes through the real entry point so the reported test metrics come from
    exactly the command written into the results file -- no second code path
    that could drift from it.
    """
    argv = result['command'].split()[3:]  # drop the "python -u run.py" prefix
    # argparse lets a repeated option win, so appending is enough to override
    # the study's own --des/--itr without having to unpick the command line.
    argv += ['--des', 'best', '--itr', str(args.retrain_itr)]
    print(f'\n=== retraining best {result["model"]} ===')
    run_module.main(argv)


def main():
    parser = argparse.ArgumentParser(
        description='Optuna hyper-parameter search, seq_len=96 / pred_len=1')
    parser.add_argument('--model', default='all',
                        help="model name, or 'all' for every model in the registry")
    parser.add_argument('--n_trials', type=int, default=50,
                        help='TARGET number of trials per study, counting what the '
                             'database already holds -- re-running tops a study up to '
                             'this number instead of adding another full batch, and a '
                             'study already at the target is skipped. Raise it to '
                             'search further. The same budget for every model, so the '
                             'comparison is between architectures and not between '
                             'search efforts')
    parser.add_argument('--timeout', type=float, default=None,
                        help='per-model wall-clock budget in seconds')
    parser.add_argument('--n_seeds', '--itr', type=int, default=3,
                        help='repeats per trial: each configuration is trained this many '
                             'times (seeds --seed, +1, +2 ...) and scored by the MEAN of '
                             'its validation losses. This is run.py --itr applied during '
                             'the search. 3 is the default because 519 validation windows '
                             'make a single run too noisy to rank on; 1 searches three '
                             'times faster if the ranking is only indicative')
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--train_epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=7,
                        help='epochs without validation improvement before training stops. '
                             '7 is deliberately patient: on this small, noisy split a '
                             'configuration can sit flat for several epochs and then '
                             'improve, and a short fuse would rank it on the plateau '
                             'rather than on where it ends up')
    parser.add_argument('--n_startup_trials', type=int, default=10)
    parser.add_argument('--n_warmup_steps', type=int, default=5)
    parser.add_argument('--out_dir', default='./tuning/results')
    parser.add_argument('--checkpoint_dir', default=None,
                        help='where trials write their per-epoch checkpoints; defaults to '
                             '<out_dir>/_checkpoints. Point it at local disk when out_dir '
                             'lives on a network mount such as Google Drive -- the files '
                             'are rewritten every improving epoch and deleted per trial')
    parser.add_argument('--study_prefix', default='rv')
    parser.add_argument('--retrain_best', action='store_true',
                        help='after the search, train the winner and print test metrics')
    parser.add_argument('--retrain_itr', type=int, default=5,
                        help='seeds for the final run (run.py --itr). Five repeats give '
                             'the mean +/- std that a benchmark table should carry; a '
                             'single run sits closer to a best case than to a mean')

    # Dataset / protocol passthrough.
    parser.add_argument('--data', default='custom')
    parser.add_argument('--root_path', default='./data/')
    parser.add_argument('--data_path', default='EURUSD-RV.csv')
    parser.add_argument('--features', default='S')
    parser.add_argument('--target', default='RV')
    parser.add_argument('--freq', default='d')
    parser.add_argument('--enc_in', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--aggregate_mean', action='store_true', default=True,
                        help='on by default: puts the loss on the HAR-RV target scale')
    parser.add_argument('--no_aggregate_mean', action='store_false', dest='aggregate_mean')
    parser.add_argument('--log', action='store_true', default=True,
                        help='on by default: train on ln(RV)')
    parser.add_argument('--raw', action='store_false', dest='log',
                        help='train on raw RV instead of ln(RV)')
    parser.add_argument('--no_use_gpu', action='store_false', dest='use_gpu', default=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    models = MODELS if args.model == 'all' else (args.model,)
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        parser.error(f"unknown model(s) {unknown}; known: {', '.join(MODELS)}")

    results = []
    for model in models:
        result = tune_model(args, model)
        if result is not None:
            results.append(result)

    if results:
        summary = os.path.join(args.out_dir, 'summary.json')
        with open(summary, 'w') as fh:
            json.dump(results, fh, indent=2)
        print('\n' + '=' * 72)
        print(f'{"model":<14}{"best val loss":>16}{"trials":>10}{"seconds":>10}')
        for r in sorted(results, key=lambda r: r['best_val_loss']):
            print(f'{r["model"]:<14}{r["best_val_loss"]:>16.6f}'
                  f'{r["n_complete"]:>10}{r["seconds"]:>10.0f}')
        print('=' * 72)
        print(f'written to {summary}')

    if args.retrain_best:
        for result in results:
            retrain_best(args, result)


if __name__ == '__main__':
    main()
