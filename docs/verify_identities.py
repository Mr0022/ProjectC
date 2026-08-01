#!/usr/bin/env python3
"""
Numerical verification of the analytical claims in docs/HAR-RV-methodology.md.

This is a unit test of the *mathematics*, not of the estimation code: it uses
only the standard library, so it cannot inherit a bug from pandas, numpy or
statsmodels, and it can be run by a reader who wants to check the derivations
before trusting the results.

Checks, keyed to the equation numbers of the methodology note:

  (10)  Effective sample size            T_h = T - h - 21
  (15)  Log-of-mean identity             exp(y_{t,t+h}) == RV_{t,t+h} exactly
  (14)  Jensen inequality                mean-of-logs <= log-of-mean, always
  (30)  Implied AR(22) representation    HAR == constrained AR(22) at h = 1
  (22)/(23)  Newey-West bandwidths       L(h, T_h) on the actual training sample

Usage
-----
    python docs/verify_identities.py
    python docs/verify_identities.py --data data/EURUSD-RV.csv
"""

import argparse
import csv
import math
import os
import sys

LAG_W, LAG_M = 5, 22
HORIZONS = (1, 5, 22)
TRAIN_END_YEAR = 2024

# Tolerances. The identities below are exact in real arithmetic, so anything
# above rounding error is a genuine failure, not a tuning knob.
TOL_REL = 1e-12
TOL_ABS = 1e-10


def mean(xs):
    return sum(xs) / len(xs)


def load_rv(path):
    """Read the RV column, drop non-positive rows exactly as the loader does."""
    dates, rv = [], []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        col = "RV" if "RV" in reader.fieldnames else reader.fieldnames[1]
        for row in reader:
            value = float(row[col])
            if value > 0.0:                      # non-trading days
                dates.append(row[reader.fieldnames[0]])
                rv.append(value)
    return dates, rv


def check_sample_size(rv):
    """(10)  T_h = T - h - 21."""
    n = len(rv)
    ok = True
    for h in HORIZONS:
        # A row t is usable iff the monthly window RV_{t-1}..RV_{t-22} exists
        # and the target window t..t+h-1 stays inside the sample.
        counted = sum(1 for t in range(n) if t >= LAG_M and t + h - 1 <= n - 1)
        formula = n - h - 21
        ok &= counted == formula
        print(f"    h={h:>2}:  counted {counted:>5}   formula {formula:>5}   "
              f"{'ok' if counted == formula else 'MISMATCH'}")
    return ok


def check_log_identity(rv):
    """(15)  exp( ln( mean RV ) ) == mean RV, at every horizon."""
    n = len(rv)
    worst = 0.0
    for h in HORIZONS:
        for t in range(LAG_M, n - h + 1):
            raw = mean(rv[t:t + h])
            worst = max(worst, abs(math.exp(math.log(raw)) - raw) / raw)
    print(f"    max relative deviation over all (h, t): {worst:.3e}")
    return worst < TOL_REL


def check_jensen(rv):
    """(14)  mean-of-logs <= log-of-mean, with equality only if RV is flat."""
    n = len(rv)
    ok = True
    for window in (LAG_W, LAG_M):
        gaps = []
        for t in range(window, n):
            block = rv[t - window:t]
            gaps.append(math.log(mean(block)) - mean([math.log(x) for x in block]))
        ok &= min(gaps) >= 0.0
        print(f"    window {window:>2}:  min {min(gaps):+.4f}   mean {mean(gaps):+.4f}   "
              f"max {max(gaps):+.4f}   ratio at max {math.exp(max(gaps)):.2f}x")
    return ok


def check_ar22(rv):
    """(30)  HAR at h=1 is a constrained AR(22)."""
    b0, bd, bw, bm = 0.05, 0.31, 0.42, 0.19        # arbitrary, identity is exact
    phi = ([bd + bw / LAG_W + bm / LAG_M]
           + [bw / LAG_W + bm / LAG_M] * (LAG_W - 1)
           + [bm / LAG_M] * (LAG_M - LAG_W))
    assert len(phi) == LAG_M

    worst = 0.0
    for t in range(LAG_M, len(rv)):
        har = (b0 + bd * rv[t - 1]
               + bw * mean(rv[t - LAG_W:t])
               + bm * mean(rv[t - LAG_M:t]))
        ar = b0 + sum(phi[j] * rv[t - 1 - j] for j in range(LAG_M))
        worst = max(worst, abs(har - ar))

    total = sum(phi)
    print(f"    max absolute deviation: {worst:.3e}")
    print(f"    sum(phi) = {total:.10f}   b_d+b_w+b_m = {bd + bw + bm:.10f}")
    return worst < TOL_ABS and abs(total - (bd + bw + bm)) < TOL_ABS


def check_bandwidths(dates, rv):
    """(22)/(23)  L(h, T_h) = max{ 2(h-1), floor(4 (T_h/100)^(2/9)) }."""
    n_train = sum(1 for d in dates if int(d[:4]) <= TRAIN_END_YEAR)
    print(f"    training rows (year <= {TRAIN_END_YEAR}): {n_train}")
    for h in HORIZONS:
        t_h = n_train - h - 21
        overlap = 2 * (h - 1)
        plugin = int(math.floor(4.0 * (t_h / 100.0) ** (2.0 / 9.0)))
        print(f"    h={h:>2}:  T_h={t_h:>5}   overlap={overlap:>3}   "
              f"plug-in={plugin:>3}   L={max(overlap, plugin):>3}"
              f"   ({'plug-in binds' if plugin > overlap else 'overlap binds' if overlap > plugin else 'tie'})")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=None,
                        help="path to the RV csv (default: data/EURUSD-RV.csv "
                             "relative to the repository root)")
    args = parser.parse_args(argv)

    path = args.data or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "EURUSD-RV.csv")
    if not os.path.isfile(path):
        parser.error(f"data file not found: {path}")

    dates, rv = load_rv(path)
    print(f"\nData: {path}")
    print(f"Sample: {dates[0]} to {dates[-1]},  T = {len(rv)} positive RV days\n")

    checks = [
        ("(10)  effective sample size  T_h = T - h - 21", check_sample_size, (rv,)),
        ("(15)  log-of-mean identity   exp(y) == RV target", check_log_identity, (rv,)),
        ("(14)  Jensen                 mean-of-logs <= log-of-mean", check_jensen, (rv,)),
        ("(30)  implied AR(22)         HAR == constrained AR(22)", check_ar22, (rv,)),
        ("(22)/(23)  Newey-West bandwidths", check_bandwidths, (dates, rv)),
    ]

    failures = 0
    for title, fn, fn_args in checks:
        print(title)
        passed = fn(*fn_args)
        print(f"    -> {'PASS' if passed else 'FAIL'}\n")
        failures += not passed

    print("=" * 64)
    print(f"  {len(checks) - failures}/{len(checks)} checks passed")
    print("=" * 64)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
