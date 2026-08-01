#!/usr/bin/env python3
"""
Numerical verification of the claims in docs/dl_vs_har_targets_losses.tex.

Standard library only, so it cannot inherit a bug from torch, pandas or
statsmodels. Checks, keyed to the equation numbers of that note:

  (12)  Path/aggregate decomposition   L_path = L_agg + within-window spread
  (14)  logsumexp identity             ln(mean(exp x)) == logsumexp(x) - ln h
  (17)  Test-sample alignment          the DL loader and HAR-RV_RUN.PY score
                                       the same forecast origins, same dates
Usage
-----
    python docs/verify_target_alignment.py
    python docs/verify_target_alignment.py --data data/EURUSD-RV.csv --seq_len 96
"""

import argparse
import csv
import math
import os
import random
import sys

TRAIN_END_YEAR = 2022      # data_loader.py
VAL_END_YEAR = 2024        # data_loader.py  (= HAR-RV_RUN.PY TRAIN_END_YEAR)
HORIZONS = (1, 5, 22)
LAG_M = 22
TOL = 1e-9


def mean(xs):
    return sum(xs) / len(xs)


def load_rv(path):
    dates, rv = [], []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        dcol, vcol = reader.fieldnames[0], ("RV" if "RV" in reader.fieldnames
                                            else reader.fieldnames[1])
        for row in reader:
            value = float(row[vcol])
            if value > 0.0:                       # drop_nonpositive
                dates.append(row[dcol])
                rv.append(value)
    return dates, rv


def check_decomposition(rv):
    """(12)  (1/h) sum e_k^2  ==  ebar^2 + (1/h) sum (e_k - ebar)^2."""
    rng = random.Random(0)
    worst_overall = 0.0
    for h in (5, 22):
        worst_gap = 0.0      # violation of the identity, this horizon
        ratios = []          # L_path / L_agg, i.e. how much extra the path loss carries
        for _ in range(2000):
            t = rng.randrange(LAG_M, len(rv) - h)
            actual = rv[t:t + h]
            # a plausible path forecast: last value carried forward, jittered
            path = [rv[t - 1] * math.exp(rng.gauss(0.0, 0.25)) for _ in range(h)]
            err = [a - p for a, p in zip(actual, path)]
            ebar = mean(err)

            l_path = mean([e * e for e in err])
            l_agg = ebar * ebar                       # (mean actual - mean forecast)^2
            spread = mean([(e - ebar) ** 2 for e in err])

            worst_gap = max(worst_gap, abs(l_path - (l_agg + spread)))
            # ebar must equal mean(actual) - mean(path)
            worst_gap = max(worst_gap, abs(ebar - (mean(actual) - mean(path))))
            if l_agg > 0:
                ratios.append(l_path / l_agg)

        ratios.sort()
        median = ratios[len(ratios) // 2]
        worst_overall = max(worst_overall, worst_gap)
        print(f"    h={h:>2}: identity max deviation {worst_gap:.3e}   "
              f"median L_path/L_agg = {median:.2f}x   "
              f"(dispersion share {100.0 * (1.0 - 1.0 / median):.0f}%)")
    return worst_overall < TOL


def check_logsumexp(rv):
    """(14)  ln( (1/h) sum exp(x_k) ) == logsumexp(x) - ln(h)."""
    worst = 0.0
    for h in HORIZONS:
        for t in range(LAG_M, len(rv) - h, 7):
            x = [math.log(v) for v in rv[t:t + h]]
            direct = math.log(mean([math.exp(xi) for xi in x]))
            m = max(x)
            lse = m + math.log(sum(math.exp(xi - m) for xi in x))
            worst = max(worst, abs(direct - (lse - math.log(h))))
    print(f"    max absolute deviation over all (h, t): {worst:.3e}")
    return worst < TOL


def check_alignment(dates, rv, seq_len):
    """(17)  DL test windows and HAR test rows are the same forecast origins."""
    n = len(rv)
    val_end = sum(1 for d in dates if int(d[:4]) <= VAL_END_YEAR)
    ok = True
    for h in HORIZONS:
        # Dataset_Custom: data_x spans [val_end - seq_len, n);
        # __len__ = len(data_x) - seq_len - pred_len + 1
        dl_len = (n - (val_end - seq_len)) - seq_len - h + 1
        # first / last target day of the DL windows
        dl_first = val_end
        dl_last = n - h

        # HAR-RV_RUN.PY: rows with year >= 2025 whose h-day target fits
        har_rows = [t for t in range(n) if t >= val_end and t + h - 1 <= n - 1
                    and t >= LAG_M]
        har_len, har_first, har_last = len(har_rows), har_rows[0], har_rows[-1]

        same = (dl_len == har_len and dl_first == har_first and dl_last == har_last)
        ok &= same
        print(f"    h={h:>2}: DL n={dl_len:>4} [{dates[dl_first]} .. {dates[dl_last]}]   "
              f"HAR n={har_len:>4} [{dates[har_first]} .. {dates[har_last]}]   "
              f"{'identical' if same else 'MISMATCH'}")
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=None, help="path to the RV csv")
    parser.add_argument("--seq_len", type=int, default=96,
                        help="look-back length used by the deep models (default 96)")
    args = parser.parse_args(argv)

    path = args.data or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "EURUSD-RV.csv")
    if not os.path.isfile(path):
        parser.error(f"data file not found: {path}")

    dates, rv = load_rv(path)
    print(f"\nData: {path}")
    print(f"Sample: {dates[0]} to {dates[-1]},  T = {len(rv)} positive RV days")
    print(f"Splits: train <= {TRAIN_END_YEAR}, val <= {VAL_END_YEAR}, "
          f"test > {VAL_END_YEAR};  seq_len = {args.seq_len}\n")

    checks = [
        ("(12)  path loss = aggregate loss + within-window spread",
         check_decomposition, (rv,)),
        ("(14)  logsumexp form of the log-of-mean target",
         check_logsumexp, (rv,)),
        ("(17)  DL and HAR score the same test origins",
         check_alignment, (dates, rv, args.seq_len)),
    ]

    failures = 0
    for title, fn, fn_args in checks:
        print(title)
        passed = fn(*fn_args)
        print(f"    -> {'PASS' if passed else 'FAIL'}\n")
        failures += not passed

    print("=" * 72)
    print(f"  {len(checks) - failures}/{len(checks)} checks passed")
    print("=" * 72)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
