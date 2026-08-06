"""
==============================================================================
HAR-RV Model: Multi-Horizon Realized Volatility Forecasting
              RAW RV SCALE by default  |  ln(RV) SCALE with --log
Based on: Corsi, F. (2009). A simple approximate long-memory model of
          realized volatility. Journal of Financial Econometrics, 7(2), 174-196.

Horizons : h = 1  (daily),  h = 5  (weekly),  h = 22  (monthly)

Split logic is imported from data_provider/splits.py, the same module the deep
models split on, so the two families can never drift apart. The two asset
classes do not share a calendar -- pick one with --asset:

    --asset forex   train 2012-01..2022-12 | val 2023-01..2024-12 | test 2025-01..end
    --asset crypto  train 2018-06..2024-06 | val 2024-07..2025-06 | test 2025-07..2026-06

HAR-RV is OLS: there are no hyperparameters to tune, so a held-out validation
set would serve no purpose. The validation months are therefore folded back
into the estimation sample, on whichever calendar is in force:

    forex    Train : start of sample - 2024-12   [= DL train + val]
             Test  : 2025-01 - end of sample     [= DL test]
    crypto   Train : 2018-06 - 2025-06           [= DL train + val]
             Test  : 2025-07 - 2026-06           [= DL test]

The TEST window is identical to the deep models', so out-of-sample losses
remain directly comparable. Note the deep models never fit on the validation
months; they only use them for early stopping / model selection.

Windows are EMBARGOED at their edges. Y^(h) at row t averages rows t..t+h-1, so
assigning rows to a split on their own date alone would put h-1 rows (21 at
h=22) in the training sample whose targets average test-period RV, and would
peg the test row count to wherever the CSV happens to stop rather than to the
test boundary. horizon_month_mask in splits.py admits a row only when its whole
target window falls inside the window, which drops exactly those h-1 rows at
each edge. The deep models get this free -- Dataset_* enumerates windows as
len - seq_len - pred_len + 1 inside an already-sliced split, so a target can
never cross a border there -- and with the embargo both families report
(rows in window) - h + 1 forecasts on identical rows. --no-embargo restores the
usual unembargoed HAR convention for measuring what the leak was worth; it is
not a fair benchmark against the deep models.

--asset defaults to forex and is NOT inferred from the filename: running a
crypto CSV without --asset crypto silently trains on the wrong window, so the
calendar in force is printed at the top of every run and on every figure.

------------------------------------------------------------------------------
USAGE (command line / notebook)
------------------------------------------------------------------------------
    python HAR-RV_RUN.PY                         # uses default data/EURUSD-RV.csv
    python HAR-RV_RUN.PY --data EURUSD_RV.CSV    # pick a data file
    python HAR-RV_RUN.PY --data path/to/my_rv.csv --outdir results
    python HAR-RV_RUN.PY --log                   # same run on the ln(RV) scale
    python HAR-RV_RUN.PY --data btcusdt-RV.csv --asset crypto     # crypto calendar

In a Jupyter/Colab cell prefix with '!':
    !python HAR-RV_RUN.PY --data EURUSD_RV.CSV

The two scales write to different filenames (har_rv_* vs har_rv_log_*), so
running both back to back leaves both sets of results side by side:

    python HAR-RV_RUN.PY --outdir results        # raw-scale results
    python HAR-RV_RUN.PY --outdir results --log  # log-scale results

The --data path is resolved leniently: if the exact path is not found the
script also looks inside ./data and matches case-insensitively while treating
'-' and '_' as interchangeable, so 'EURUSD_RV.CSV' resolves to the shipped
'data/EURUSD-RV.csv'. Run `python HAR-RV_RUN.PY --help` for all options.

==============================================================================
TWO SCALES, ONE PIPELINE

DEFAULT (no flag) -- RAW RV. Corsi (2009) eq. 3 in its original form, i.e. a
LEVELS regression:

    Target      Y_t^(h) = (1/h) * Sum_{k=1}^{h}  RV_{t+k}
    Regressors  RV_d    =            RV_{t-1}
                RV_w    = (1/5)  * Sum_{k=1}^{5}  RV_{t-k}
                RV_m    = (1/22) * Sum_{k=1}^{22} RV_{t-k}

No logs are taken anywhere in this mode, so there is no Jensen gap to worry
about -- there is no non-linear transform.

--log -- ln(RV), on the LOG OF MEAN convention. Each window is averaged on the
raw variance scale and the log is taken of that average, i.e. the log sits
OUTSIDE every sum:

    Target      Y_t^(h) = ln(     (1/h) * Sum_{k=1}^{h}  RV_{t+k} )
    Regressors  x_d     = ln(              RV_{t-1}              )
                x_w     = ln(     (1/5)  * Sum_{k=1}^{5}  RV_{t-k} )
                x_m     = ln(     (1/22) * Sum_{k=1}^{22} RV_{t-k} )

This is the log-HAR of Corsi (2009) sec. 3.2 and Andersen, Bollerslev &
Diebold (2007). Everything downstream -- OLS, HAC covariance, diagnostics,
figures, exports -- runs on these columns without further special-casing,
because the loader hands the same four column names to the same code in both
modes.

LOG OF MEAN vs MEAN OF LOGS -- why the placement of the log matters. The two
are different regressors: ln((1/5)*Sum RV) = ln(arithmetic mean), whereas
(1/5)*Sum ln RV = ln(geometric mean), and geometric <= arithmetic always.
This script uses LOG OF MEAN throughout, which buys one important property:

    exp( Y_t^(h) )  ==  the raw-mode target, exactly, at EVERY horizon.

So a back-transformed --log forecast predicts the very same quantity the
levels regression predicts, and the two modes can be ranked against each
other directly at h=1, 5 and 22 alike. Under mean-of-logs the target would be
a GEOMETRIC forward mean -- systematically smaller, and much smoother, since
a lone spike moves a geometric mean only through an h-th root. Cross-mode
losses would then favour the log fit for arithmetic rather than forecasting
reasons, and would be honest only at h=1, where the two means coincide.

The daily component is identical under both conventions: a one-day window has
nothing to aggregate, so it is ln(RV_{t-1}) either way.

WHAT --log CHANGES IN THE OUTPUT
  1. MSE and MAE are reported in ln(RV) units -- the scale the model is
     actually fitted on. They are NOT comparable to raw-scale losses. MSE_RV
     and MAE_RV are also reported, on the back-transformed variance scale,
     and those ARE comparable to a raw-scale run at every horizon.
  2. QLIKE is only defined for variances, so it is evaluated after mapping
     the forecast back with the lognormal (Jensen) correction
         E[RV | F] = exp( E[ln RV | F] + sigma^2 / 2 )
     where sigma^2 is the TRAINING residual variance of the log regression.
     Plain exp() returns the conditional MEDIAN and is biased low as a mean
     estimate -- systematically, not just noisily. The uncorrected number is
     printed alongside as QLIKE_naive so the size of the correction is
     visible rather than taken on trust.
  3. Non-positive forecasts become impossible: exp(.) > 0 always. The QLIKE
     floor stays in the code but never binds, and 'neg pred' is 0 by
     construction. This is the main practical gain over the levels fit.
  4. Heteroskedasticity largely goes away -- compare the Breusch-Pagan
     statistics across the two modes. Serial correlation does NOT, so the
     Newey-West HAC covariance is still used, unchanged.
  5. Outputs are written as har_rv_log_* instead of har_rv_*, so a --log run
     never overwrites the raw-scale results it is meant to be compared with.

SUM vs MEAN is NOT cosmetic in the DEFAULT raw mode. Under --log the two
differ by ln(window), a constant the intercept absorbs; on the raw scale they
differ by a MULTIPLICATIVE factor:
  * For the REGRESSORS it is still harmless -- using sums instead of means
    just rescales beta_w by 1/5 and beta_m by 1/22. Fit, residuals, R^2 and
    all losses are identical.
  * For the TARGET it is NOT harmless -- switching to the h-day sum scales
    Y, every coefficient and every residual by h, so MSE scales by h^2 and
    MAE by h. R^2 is unchanged. Corsi's convention is the AVERAGE, used here,
    which also keeps the three horizons on a comparable scale.

CONSEQUENCES OF WORKING IN LEVELS (default mode) -- read before interpreting
output. Each of these is what --log exists to address:
  1. Residuals are severely heteroskedastic and right-skewed. RV has skew
     ~7.7 and excess kurtosis ~117 in this sample; volatility-of-volatility
     scales with the level of RV, so the error variance explodes in stress
     periods. Newey-West HAC covariance keeps the standard errors valid
     (it is heteroskedasticity- AND autocorrelation-consistent), but OLS is
     no longer efficient. A Breusch-Pagan test was added to diagnostics to
     make the size of the problem visible.
  2. MSE in levels is dominated by a handful of spike days. Two or three
     crisis observations can drive the ranking of competing models. MAE and
     QLIKE are far more robust here and should carry the weight.
  3. Fitted values are no longer positive by construction. A levels OLS can
     and does predict RV < 0 in calm periods, which is meaningless as a
     variance and undefined under QLIKE. Such forecasts are never silently
     dropped -- dropping the worst ones would bias the losses downward -- and
     are handled by scale rather than discarded:
       * MSE and MAE score max(pred, 0), the best admissible forecast, since
         a variance below zero is not a prediction the model could make.
       * QLIKE is infinite at zero, so it floors them at a small positive
         multiple of mean training RV instead.
     They are counted and reported per horizon either way, so how much the
     numbers lean on the guard stays visible.
  4. Losses are on the RV scale, so they are NOT comparable to the ln(RV)-scale
     numbers produced by --log. Any deep-learning baseline must be evaluated
     on whichever scale is being reported.

INPUT: the loader accepts either a raw 'RV' column or an 'ln_RV' column and
converts once as needed. Note that --log needs raw variances regardless, since
its windows are averaged before the log is applied.
Non-positive RV rows -- exchange holidays with no trading -- are dropped. That
drop is also what keeps --log defined at all, since ln(0) is not.

HAC bandwidth -- Newey-West (1987) Bartlett kernel, L chosen as

    L = max( 2*(h-1),  floor(4*(T/100)^(2/9)) )

  * 2*(h-1) is the overlap term (Patton & Sheppard, 2009; Bollerslev et al.,
    2016). Averaging RV over h forward days induces an MA(h-1) structure in
    the residuals of the h-step regression, and the bandwidth must at least
    cover it. It is a FLOOR, not the whole answer.
  * floor(4*(T/100)^(2/9)) is the Newey & West (1994) plug-in rule, which
    handles serial correlation that is in the data rather than induced by
    the overlap. With T ~ 3.3k this evaluates to 8.

    h=1  -> L=8   (overlap floor 0, so the data-driven rule binds)
    h=5  -> L=8   (overlap floor 8)
    h=22 -> L=42  (overlap floor 42)

Why h=1 is NOT L=0. There is no overlap at h=1, so 2*(h-1) = 0 -- but that
does not mean the residuals are serially uncorrelated, and here they are
not: Ljung-Box(20) on the h=1 training residuals rejects at p ~ 0.002.
Setting L=0 would assume away autocorrelation the data plainly shows.

Two distinct problems existed at h=1, and only one of them was cosmetic.

  (a) Wrong description. HAC with maxlags=0 does NOT give classical OLS
      standard errors -- the Bartlett kernel keeps only the j=0 term, so it
      reduces exactly to White/HC1. The old comment named the wrong object.
  (b) Wrong inference. HC1 is robust to heteroskedasticity but NOT to
      autocorrelation, and the h=1 residuals are autocorrelated. So the
      reported h=1 standard errors were genuinely understated for the
      persistent regressors -- a substantive error, not a labelling one:

          b_d :  SE 0.049 -> 0.030   (t  3.02 -> 5.04)
          b_w :  SE 0.080 -> 0.099   (t  4.78 -> 3.88)
          b_m :  SE 0.060 -> 0.087   (t  5.11 -> 3.50)

      (classical OLS understates further still: SE(b_d) = 0.020.)

All three coefficients stay significant at 1%, so no conclusion in this
sample flips -- but the old t-statistics on the weekly and monthly
components were inflated by 23% and 46% respectively.

Pass --nw-lag-rule overlap to fall back to the pure L=2*(h-1) convention as
a robustness check.

The bandwidth rule is scale-independent: the overlap floor is a property of
the h-day target window and the plug-in term a property of the sample size,
so --log gets the same L at each horizon. The standard errors quoted above
are from the raw-scale fit; --log produces its own, generally tighter.

Metrics
  default : MSE, MAE, QLIKE (Patton, 2011), all on the RAW RV scale.
  --log   : MSE, MAE in ln(RV) units; QLIKE on the back-transformed variance
            (Jensen-corrected); MSE_RV, MAE_RV likewise back-transformed for
            comparison against a raw-scale run.
==============================================================================
"""

# -- Standard Library ---------------------------------------------------------
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

# -- Third-party --------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates  as mdates
from   matplotlib.ticker import AutoMinorLocator

import statsmodels.api as sm
from   statsmodels.regression.linear_model import OLS
from   statsmodels.stats.stattools         import durbin_watson
from   statsmodels.stats.diagnostic        import acorr_ljungbox, het_breuschpagan
from   scipy                               import stats

# Split calendars, shared with the deep models (data_provider/data_loader.py).
# splits.py is deliberately pandas/numpy only, so importing it here does not
# drag torch or sklearn into a HAR-RV run.
from   data_provider.splits import (SPLITS, har_bounds, month_mask, fmt_month,
                                    horizon_month_mask, month_of, next_month,
                                    to_ordinal)


# ==============================================================================
# 0.  CONFIGURATION  (defaults; overridable on the command line -- see parse_args)
# ==============================================================================

DATA_FILE = "data/EURUSD-RV.csv"   # default data set (override with --data)
OUT_DIR   = "."                    # where figures / CSVs are written (--outdir)

# --log: transform RV -> ln(RV) at load time and run the ENTIRE pipeline on the
# log scale (regressors, target, fit, losses, figures). False = raw RV levels,
# the default and the behaviour of every previous version of this script.
LOG_MODE = False

# Asset class, i.e. which calendar in data_provider/splits.py to split on.
# har_bounds() folds that calendar's validation months into HAR's training
# sample (OLS has nothing to tune) and leaves the test window untouched, so the
# deep models and HAR-RV are scored on exactly the same out-of-sample rows.
# Set from --asset; the four bounds below are filled in by resolve_split().
ASSET = "forex"

# Inclusive (year, month) bounds. None = open on that side.
TRAIN_START = None
TRAIN_END   = None
TEST_START  = None
TEST_END    = None

# Embargo the h-1 rows at each window edge whose h-day target would close on the
# far side of the boundary (horizon_month_mask in splits.py). ON by default: at
# the train|test seam those rows are training labels built from test data, and
# the deep models' window enumeration already excludes their equivalent. Turn it
# off with --no-embargo to reproduce the unembargoed numbers and measure what
# the leak was worth on a given sample.
EMBARGO = True

# Forecast horizons. 'overlap_lag' = 2*(h-1) is the MA(h-1) floor induced by
# the overlapping h-day target; 'nw_lag' is the bandwidth actually used and is
# filled in at fit time by nw_bandwidth(), which can raise it above the floor.
HORIZONS = {
    1  : {"label": "Daily   (h=1)",  "overlap_lag":  0, "nw_lag":  0},
    5  : {"label": "Weekly  (h=5)",  "overlap_lag":  8, "nw_lag":  8},
    22 : {"label": "Monthly (h=22)", "overlap_lag": 42, "nw_lag": 42},
}

# Bandwidth rule: 'auto' = max(overlap floor, Newey-West 1994 plug-in),
# 'overlap' = the pure 2*(h-1) convention (robustness check).
NW_LAG_RULE = "auto"

LAG_W    = 5    # weekly component window
LAG_M    = 22   # monthly component window
DPI_SAVE = 300

# QLIKE guard: forecasts <= 0 are floored at this multiple of mean training RV
# before QLIKE is evaluated. Reported, never silently dropped. Under --log the
# level forecast is exp(.) > 0 by construction, so the floor never binds.
QLIKE_FLOOR_FRAC = 1e-4

# -- Publication palette ------------------------------------------------------
C_ACTUAL = "#1a1a2e"
C_H      = {1: "#2166ac", 5: "#fc8d59", 22: "#d73027"}
C_TRAIN  = "#2166ac"
C_TEST   = "#d73027"

plt.rcParams.update({
    "font.family":         "serif",
    "font.serif":          ["Times New Roman", "DejaVu Serif"],
    "font.size":           11,
    "axes.titlesize":      11,
    "axes.labelsize":      11,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           True,
    "grid.alpha":          0.30,
    "grid.linestyle":      ":",
    "grid.linewidth":      0.6,
    "lines.linewidth":     1.0,
    "figure.dpi":          150,
    "savefig.dpi":         DPI_SAVE,
    "savefig.bbox":        "tight",
    "savefig.facecolor":   "white",
    "legend.framealpha":   0.90,
    "legend.edgecolor":    "0.75",
    "legend.fontsize":     9,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
})

SEP  = "=" * 72
THIN = "-" * 72

# Human-readable split windows, e.g. "2018-06 - 2025-06". Built once in
# load_base_features() from the requested bounds intersected with what the file
# actually contains, so every table, figure title and export says the same
# thing and says it in months -- crypto's windows do not fall on year ends.
TRAIN_LABEL = ""
TEST_LABEL  = ""


# ==============================================================================
# 0a.  SCALE LABELS  (single source of truth for raw-vs-log wording)
# ==============================================================================
# The modelling columns keep the names RV / RV_d / RV_w / RV_m in BOTH modes --
# under --log they simply hold ln(RV) instead of RV. Only the human-readable
# labels and the output filenames switch, which keeps every regression, figure
# and export path identical between the two modes.

def scale_tag() -> str:
    """Machine-readable scale identifier; goes in the metrics CSV."""
    return "ln_RV" if LOG_MODE else "raw_RV"

def scale_name() -> str:
    """Short scale name for prose and axis labels."""
    return "ln(RV)" if LOG_MODE else "raw RV"

def scale_banner() -> str:
    """Loud scale marker for section headers."""
    return "LOG SCALE" if LOG_MODE else "RAW SCALE"

def out_prefix() -> str:
    """
    Filename stem for figures and CSVs. Log-mode runs write har_rv_log_* so a
    --log run never overwrites the raw-scale results it is meant to be compared
    against (and vice versa).
    """
    return "har_rv_log" if LOG_MODE else "har_rv"

def var_labels() -> dict:
    """
    Coefficient row labels for the estimation table.

    Under --log the regressor is ln(mean RV over the window), so the label
    wraps the window in ln(.). Writing it the other way round ("Weekly ln RV")
    would read as a weekly mean OF logs -- the convention this script
    deliberately does not use.
    """
    names = ({"RV_d": "ln(Daily RV)", "RV_w": "ln(Weekly RV)",
              "RV_m": "ln(Monthly RV)"} if LOG_MODE else
             {"RV_d": "Daily RV", "RV_w": "Weekly RV", "RV_m": "Monthly RV"})
    out = {"const": "Intercept (b0)"}
    for key, short in (("RV_d", "b_d"), ("RV_w", "b_w"), ("RV_m", "b_m")):
        out[key] = names[key].ljust(10) + f" ({short})"
    return out

def coef_panel_labels() -> dict:
    """Coefficient panel titles for the coefficient-path figure."""
    if LOG_MODE:
        return {"const": "b0  (Intercept)",    "RV_d": "b_d (ln Daily RV)",
                "RV_w": "b_w (ln Weekly RV)", "RV_m": "b_m (ln Monthly RV)"}
    return {"const": "b0  (Intercept)", "RV_d": "b_d (Daily)",
            "RV_w": "b_w (Weekly)",    "RV_m": "b_m (Monthly)"}


# ==============================================================================
# 0b.  ARGUMENT PARSING & DATA-PATH RESOLUTION
# ==============================================================================

def parse_args(argv=None):
    """
    Command-line interface. Every knob has a sensible default so the script
    also runs with no arguments at all (uses data/EURUSD-RV.csv).
    """
    p = argparse.ArgumentParser(
        prog="HAR-RV_RUN.PY",
        description="HAR-RV multi-horizon realized-volatility forecasting "
                    "(Corsi, 2009). Runs on the raw RV scale by default, or on "
                    "the ln(RV) scale with --log.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data", "-d", default=DATA_FILE, metavar="CSV",
        help="Path to the RV CSV file (a 'date' index plus an 'RV' or "
             "'ln_RV' column). Resolved leniently: if not found as given, "
             "./data is also searched and '-'/'_' plus case are ignored, so "
             "'EURUSD_RV.CSV' matches the shipped 'data/EURUSD-RV.csv'.")
    p.add_argument(
        "--outdir", "-o", default=OUT_DIR, metavar="DIR",
        help="Directory for output figures and CSVs (created if missing).")
    p.add_argument(
        "--log", action="store_true", default=LOG_MODE,
        help="Run the whole pipeline on the ln(RV) scale, using the LOG OF MEAN "
             "convention: each window (and the h-day target) is averaged on the "
             "raw variance scale and the log is taken of that average, so the "
             "log sits outside every sum. The OLS fit is then the log-HAR of "
             "Corsi (2009) sec. 3.2 / Andersen et al. (2007) and MSE/MAE are "
             "reported in ln(RV) units. QLIKE needs variances, so it is "
             "evaluated after exp() with the lognormal Jensen correction; "
             "because exp() of this target recovers the raw-mode target "
             "exactly, the back-transformed losses are comparable to a "
             "raw-scale run at every horizon. Outputs are written as "
             "har_rv_log_* so they do not overwrite a raw-scale run.")
    p.add_argument(
        "--asset", choices=sorted(SPLITS), default=ASSET,
        help="Which split calendar to use (data_provider/splits.py). 'forex' = "
             "train ..2022-12 / val 2023-01..2024-12 / test 2025-01.., 'crypto' "
             "= train 2018-06..2024-06 / val 2024-07..2025-06 / test 2025-07.."
             "2026-06. HAR-RV folds the validation months into training and "
             "keeps the test window, so it stays comparable to the deep models "
             "run with the matching --data. NOT inferred from the filename.")
    for flag, dest in (("--train-start", "train_start"),
                       ("--train-end",   "train_end"),
                       ("--test-start",  "test_start"),
                       ("--test-end",    "test_end")):
        p.add_argument(
            flag, dest=dest, default=None, metavar="YYYY-MM",
            help=f"Override the {dest.replace('_', ' ')} boundary of --asset "
                 f"(inclusive); None keeps the --asset boundary. Accepts "
                 f"YYYY-MM, or a bare YYYY meaning "
                 f"{'January' if dest.endswith('start') else 'December'} of "
                 f"that year. 'open' removes the boundary.")
    # Deprecated year-granularity spellings, kept so older invocations still
    # run. They cannot express crypto's mid-year boundaries; --train-end /
    # --test-start supersede them.
    p.add_argument("--train-end-year", type=int, default=None,
                   metavar="YEAR", help=argparse.SUPPRESS)
    p.add_argument("--test-start-year", type=int, default=None,
                   metavar="YEAR", help=argparse.SUPPRESS)
    p.add_argument(
        "--qlike-floor-frac", type=float, default=QLIKE_FLOOR_FRAC, metavar="FRAC",
        help="Non-positive forecasts are floored at FRAC * mean(training RV) "
             "before QLIKE is evaluated.")
    p.add_argument(
        "--no-embargo", dest="embargo", action="store_false", default=EMBARGO,
        help="Assign rows to a split on their own date only, the usual HAR "
             "convention. The h-day target spans t..t+h-1, so this puts h-1 "
             "training rows (21 at h=22) in the sample whose targets average "
             "test-period RV, and makes the test row count depend on where the "
             "CSV stops rather than on --test-end. Default is to embargo those "
             "rows, which is what the deep models' window enumeration does and "
             "what keeps the two families' row counts equal. Use this only to "
             "measure what the leak is worth, not to report a benchmark.")
    p.add_argument(
        "--nw-lag-rule", choices=["auto", "overlap"], default=NW_LAG_RULE,
        help="Newey-West bandwidth. 'auto' = max(2*(h-1), Newey-West 1994 "
             "plug-in), which keeps h=1 from collapsing to L=0 despite the "
             "residual autocorrelation Ljung-Box detects. 'overlap' = the "
             "pure 2*(h-1) convention, for robustness checks.")
    return p.parse_args(argv)


def parse_month(text: str, flag: str, *, is_start: bool):
    """
    Parse a boundary written as 'YYYY-MM', 'YYYY' or 'open'.

    A bare year is widened to the whole year in the direction the boundary
    faces -- January for a start bound, December for an end bound -- so
    '--train-end 2024' means "through the end of 2024", which is what the
    retired --train-end-year flag meant. 'open' returns None (no boundary).
    """
    text = str(text).strip().lower()
    if text in ("open", "none"):
        return None
    parts = text.split("-")
    try:
        if len(parts) == 1:
            year, month = int(parts[0]), (1 if is_start else 12)
        elif len(parts) == 2:
            year, month = int(parts[0]), int(parts[1])
        else:
            raise ValueError
        if not 1 <= month <= 12:
            raise ValueError
    except ValueError:
        raise SystemExit(
            f"{flag}: expected YYYY-MM, YYYY or 'open', got '{text}'.")
    return (year, month)


def resolve_split(args):
    """
    Turn --asset plus any boundary overrides into the four HAR-RV bounds.

    The asset's calendar comes from data_provider/splits.py and is passed
    through har_bounds(), which folds the deep models' validation months into
    HAR's training sample and leaves their test window untouched. Overrides are
    applied on top, so --asset picks a coherent default and the individual
    flags stay available for robustness checks.
    """
    train_start, train_end, test_start, test_end = har_bounds(SPLITS[args.asset])

    # Retired spellings, mapped onto the month-granular ones.
    if args.train_end_year is not None:
        train_end = (args.train_end_year, 12)
    if args.test_start_year is not None:
        test_start = (args.test_start_year, 1)

    if args.train_start is not None:
        train_start = parse_month(args.train_start, "--train-start", is_start=True)
    if args.train_end is not None:
        train_end = parse_month(args.train_end, "--train-end", is_start=False)
    if args.test_start is not None:
        test_start = parse_month(args.test_start, "--test-start", is_start=True)
    if args.test_end is not None:
        test_end = parse_month(args.test_end, "--test-end", is_start=False)

    if train_end is None:
        raise SystemExit("--train-end cannot be open: training needs an end.")
    if test_start is None:
        test_start = next_month(train_end)
    # An overlap would fit and score on the same rows -- the one mistake these
    # flags make easy, and one that quietly flatters every reported loss.
    if to_ordinal(test_start) <= to_ordinal(train_end):
        raise SystemExit(
            f"Test starts {fmt_month(test_start)} but training runs through "
            f"{fmt_month(train_end)}: the windows overlap, so the model would "
            f"be scored on rows it was fitted on.")
    return train_start, train_end, test_start, test_end


def resolve_data_path(path: str) -> str:
    """
    Resolve a user-supplied data path leniently.

    Search order:
      1. The path exactly as given.
      2. The same path with './data/' prepended (this repo keeps CSVs there).
      3. A fuzzy match inside the candidate directories, comparing basenames
         case-insensitively and treating '-' and '_' as equivalent -- so
         'EURUSD_RV.CSV', 'eurusd-rv.csv' and 'data/EURUSD-RV.csv' all point
         at the same shipped file.

    Returns the first path that exists; if nothing matches, returns the path
    unchanged so the downstream FileNotFoundError names what the user asked for.
    """
    if os.path.isfile(path):
        return path

    base       = os.path.basename(path)
    candidates = [path, os.path.join("data", base)]
    for cand in candidates:
        if os.path.isfile(cand):
            return cand

    def _norm(name: str) -> str:
        return name.lower().replace("-", "_")

    target   = _norm(base)
    seen     = set()
    for d in (os.path.dirname(path) or ".", "data", "."):
        if d in seen or not os.path.isdir(d):
            continue
        seen.add(d)
        for fname in os.listdir(d):
            if _norm(fname) == target:
                return os.path.join(d, fname)

    return path   # let load_base_features raise a clear FileNotFoundError


# ==============================================================================
# 1.  DATA LOADING & BASE REGRESSOR CONSTRUCTION
# ==============================================================================

def set_split_labels(index):
    """
    Build the printed split windows, e.g. 'Train 2018-06 - 2025-06'.

    The requested bounds are intersected with what the file actually holds, so
    an open bound reports the real sample edge rather than '?', and a closed
    bound that reaches past the data (a crypto CSV that stops early) reports
    where the data really ends. Labels are months throughout: crypto's windows
    do not fall on year ends, and printing them as years would round the
    boundary the run was configured with.
    """
    global TRAIN_LABEL, TEST_LABEL

    first, last = month_of(index[0]), month_of(index[-1])
    lo = first if TRAIN_START is None else max(TRAIN_START, first)
    hi = last if TEST_END is None else min(TEST_END, last)
    TRAIN_LABEL = f"{fmt_month(lo)} - {fmt_month(TRAIN_END)}"
    TEST_LABEL  = f"{fmt_month(TEST_START)} - {fmt_month(hi)}"


def load_base_features(filepath: str) -> pd.DataFrame:
    """
    Load the realized-variance series and build the three HAR regressors.

    RAW mode (default) -- Corsi (2009) eq. 3, a levels regression:

        RV_d = RV_{t-1}
        RV_w = (1/5)  * Sum_{k=1}^{5}  RV_{t-k}
        RV_m = (1/22) * Sum_{k=1}^{22} RV_{t-k}

    LOG mode (--log) -- the log-HAR, LOG OF MEAN. Each window is averaged on
    the RAW variance scale first and the log is taken of that average:

        x_d  = ln(        RV_{t-1}                 )
        x_w  = ln( (1/5)  * Sum_{k=1}^{5}  RV_{t-k} )
        x_m  = ln( (1/22) * Sum_{k=1}^{22} RV_{t-k} )

    Log-of-mean vs mean-of-logs. Both appear in the literature and they are
    NOT the same regressor: ln( (1/5)*Sum RV ) = ln( arithmetic mean ) while
    (1/5)*Sum ln RV = ln( geometric mean ), and by Jensen the latter is always
    the smaller. This function implements LOG OF MEAN, so every component is
    the log of an arithmetic average of variances -- the aggregation happens
    on the scale where averaging variances is meaningful, and only then is the
    log applied.

    The daily component is unaffected by the choice: a window of one has
    nothing to aggregate, so log-of-mean and mean-of-logs both reduce to
    ln(RV_{t-1}) exactly.

    Why this convention matters downstream: because the target (see
    build_horizon_target) is likewise ln( arithmetic forward mean ),
    exponentiating a --log forecast recovers the SAME quantity raw mode
    predicts, at every horizon. Under mean-of-logs it would recover a
    geometric mean instead, which is a different -- and systematically
    smaller, smoother -- object, and cross-mode losses would only be
    comparable at h=1.

    Sums vs means remain cosmetic on this scale: ln(sum) = ln(mean) +
    ln(window), and the intercept absorbs the constant.

    Column handling
    ---------------
    'RV'    -> raw variance; logged here iff --log.
    'ln_RV' -> already logged; exponentiated once to raw RV, which --log needs
               anyway in order to average the windows on the variance scale.
               The one-day component still reads the input column directly, so
               it never round-trips through exp() and log().
    Otherwise the first numeric column is taken and ASSUMED to be raw RV;
    a warning is printed, because silently transforming the wrong column
    is exactly the bug this loader is written to avoid.

    Non-positive RV rows are dropped: they are exchange holidays with no
    trading rather than genuine zero-variance days. A zero actual makes QLIKE
    undefined, and ln(RV) undefined as well -- so the drop is what keeps --log
    well defined at all.

    Both scales are returned. 'RV' carries the one-day modelling series
    (ln(RV) under --log, raw RV otherwise); 'RV_raw' always carries raw RV,
    and under --log it is the series the multi-day windows are averaged over
    before the log is applied. 'RV_raw' also feeds the descriptive summary and
    the QLIKE floor.

    All windows use shift(1), so the information set ends strictly at t-1 --
    no look-ahead. Regressors are IDENTICAL across horizons; only the target
    changes.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"Data file not found: '{filepath}'. Pass a valid path with "
            f"--data (e.g. --data data/EURUSD-RV.csv).")

    raw = pd.read_csv(filepath, index_col=0, parse_dates=True)
    raw.index.name = "date"

    ln_in = None   # pre-logged input column, when the file ships one
    if "RV" in raw.columns:
        s_raw = raw["RV"].sort_index().dropna().astype(float)
        print("  Column 'RV' found -> raw realized variance.")
    elif "ln_RV" in raw.columns:
        ln_in = raw["ln_RV"].sort_index().dropna().astype(float)
        s_raw = np.exp(ln_in)
        print("  Column 'ln_RV' found -> already on the log scale.")
    else:
        col   = raw.select_dtypes("number").columns[0]
        s_raw = raw[col].sort_index().dropna().astype(float)
        print(f"  WARNING: no 'RV'/'ln_RV' column. Using '{col}' AS RAW RV.")

    n_bad = int((s_raw <= 0).sum())
    if n_bad:
        print(f"  Dropped {n_bad} non-positive RV row(s) (non-trading days): "
              f"{[str(d.date()) for d in s_raw.index[s_raw <= 0]]}")
        s_raw = s_raw[s_raw > 0]
        if ln_in is not None:
            ln_in = ln_in.reindex(s_raw.index)

    if LOG_MODE:
        s = ln_in if ln_in is not None else np.log(s_raw)
        print("  --log: HAR components are LOGS OF MEANS -- each window is")
        print("         averaged on the raw variance scale, then logged.")
    else:
        s = s_raw
        if ln_in is not None:
            print("  Exponentiated once to raw RV (raw mode).")

    set_split_labels(s.index)

    df = pd.DataFrame({"RV": s, "RV_raw": s_raw})
    if LOG_MODE:
        # ln( arithmetic mean of RV over the window ). Averaging happens on
        # RV_raw -- the scale on which averaging variances means something --
        # and the log is applied to the aggregate, never term by term.
        prev = s_raw.shift(1)
        df["RV_d"] = s.shift(1)                              # window 1: ln RV_{t-1}
        df["RV_w"] = np.log(prev.rolling(LAG_W).mean())
        df["RV_m"] = np.log(prev.rolling(LAG_M).mean())
    else:
        df["RV_d"] = s.shift(1)
        df["RV_w"] = s.shift(1).rolling(LAG_W).mean()
        df["RV_m"] = s.shift(1).rolling(LAG_M).mean()
    return df   # NaN rows dropped per-horizon after target is attached


# ==============================================================================
# 2.  HORIZON-SPECIFIC TARGET CONSTRUCTION
# ==============================================================================

def build_horizon_target(df_base: pd.DataFrame, h: int) -> pd.DataFrame:
    """
    Construct the h-day forward target -- Corsi (2009) eq. 3:

        raw  :  Y_t^(h) =     (1/h) * Sum_{k=1}^{h} RV_{t+k}
        --log:  Y_t^(h) = ln( (1/h) * Sum_{k=1}^{h} RV_{t+k} )

    For h = 1 both collapse to the one-day value, RV_t or ln RV_t.

    Note where the log sits: OUTSIDE the sum. The forward window is averaged
    on the raw variance scale and the log is taken of that average, matching
    the log-of-mean convention used for the regressors. So the --log target is
    the log of the ARITHMETIC forward mean, and

        exp( Y_t^(h) ) == the raw-mode target, exactly, at every horizon.

    That identity is what makes the two modes comparable. Had the log been
    taken inside the sum -- (1/h) * Sum ln RV_{t+k} -- the target would be the
    log of a GEOMETRIC forward mean: a smaller and much smoother object, since
    a single spike lifts a geometric mean by only h-th roots. Back-transformed
    losses would then flatter the log model for reasons that are arithmetic
    rather than forecasting skill, and would be honestly comparable only at
    h=1 where the two means coincide.

    Window convention: rolling(h).mean().shift(-(h-1)) spans t ... t+h-1.
    Every regressor is shifted by one day, so the information set ends at
    t-1 while the target covers the next h days -- a genuine forecast with
    no look-ahead, indexed one day earlier than the formula above.

    That forward reach is why rows cannot be assigned to a split on their own
    date alone: split_by_calendar embargoes the h-1 rows at each window edge
    whose target would close outside the window.

    Using the h-day SUM instead would multiply Y, all coefficients and all
    residuals by h (MSE by h^2) on the raw scale; on the log scale it would
    shift Y by the constant ln(h), which the intercept absorbs. R^2 is
    invariant either way, but reported raw-scale losses are not -- hence the
    average, which is also Corsi's convention.
    """
    df = df_base.copy()
    if h == 1:
        # Window of one: nothing to aggregate, so the modelling series already
        # is the target on whichever scale is in force.
        df["Y_h"] = df["RV"]
    else:
        fwd = df["RV_raw"].rolling(h).mean().shift(-(h - 1))
        df["Y_h"] = np.log(fwd) if LOG_MODE else fwd
    # Only the leading warm-up rows are dropped here. The trailing rows, where
    # the forward window runs off the end of the sample and Y_h is NaN, stay in
    # place so the frame remains a gap-free row grid: split_by_calendar has to
    # look h-1 rows AHEAD of each candidate origin to apply the embargo, and it
    # cannot do that on a frame whose tail has already been cut away. It drops
    # them itself, along with the rest of the far-edge embargo.
    return df.dropna(subset=["RV_d", "RV_w", "RV_m"])


# ==============================================================================
# 3.  CALENDAR TRAIN / TEST SPLIT   (bounds shared with the deep models)
# ==============================================================================

def split_by_calendar(df: pd.DataFrame, h: int,
                      train_start=None, train_end=None,
                      test_start=None, test_end=None, embargo=None):
    """
    Chronological split on the same calendar the deep models use, with the
    validation months folded into training (see har_bounds in splits.py).

    Bounds are inclusive (year, month) pairs; None is open on that side. Rows
    outside both windows -- before `train_start` or after `test_end` -- are
    dropped from the estimation and evaluation samples, but they were still
    available to the rolling windows that built RV_d/RV_w/RV_m, so a bounded
    training sample keeps a full 22-day look-back at its first row instead of
    losing a month to NaNs.

    `df` must arrive as the gap-free row grid build_horizon_target returns --
    trailing NaN targets included -- because the embargo counts h-1 rows ahead
    of each candidate origin.

    THE EMBARGO. Y_h at row t averages rows t .. t+h-1, so a row selected on its
    own date can carry a target reaching h-1 rows past the window it was put in.
    `horizon_month_mask` admits a row only when its target closes inside the same
    window, which trims h-1 rows from each edge and fixes two things at once:

      train|test seam -- without it the last h-1 TRAINING rows carry targets
          averaging test-period RV, so the estimated coefficients are a function
          of the data they are then scored on. 21 rows at h=22.
      far edge of the test window -- without it the test row count depends on
          where the CSV happens to stop rather than on `test_end`. That is
          invisible while a file ends exactly at its boundary (every file here
          does today) and silently adds h-1 rows the moment one runs past it,
          which would score HAR on rows the deep models never see.

    Both leave the deep models untouched: Dataset_Custom enumerates windows as
    len - seq_len - pred_len + 1 within an already-sliced split, so neither case
    can arise there. The embargo is what puts HAR-RV under the same rule, and
    with it both families report (rows in window) - h + 1 forecasts.

    `embargo=False` (--no-embargo) restores the unembargoed selection, which is
    the standard HAR convention in the literature and useful for quantifying
    what the boundary leak is worth on a given sample. It is not a fair
    benchmark against the deep models: it hands HAR h-1 extra training rows the
    deep models are denied, and those rows have seen the test period.
    """
    train_start = TRAIN_START if train_start is None else train_start
    train_end   = TRAIN_END   if train_end   is None else train_end
    test_start  = TEST_START  if test_start  is None else test_start
    test_end    = TEST_END    if test_end    is None else test_end
    embargo     = EMBARGO     if embargo     is None else embargo

    def select(lo, hi):
        if embargo:
            return horizon_month_mask(df.index, h, lo, hi)
        return month_mask(df.index, lo, hi)

    # The embargo already implies a complete forward window, so this dropna is
    # a no-op under it; it is what removes the unusable tail when it is off.
    train = df[select(train_start, train_end)].dropna(subset=["Y_h"]).copy()
    test  = df[select(test_start,  test_end)].dropna(subset=["Y_h"]).copy()

    if len(train) == 0:
        raise ValueError(
            f"Train set is empty. Check that your data covers "
            f"{fmt_month(train_start) if train_start else 'the start of sample'}"
            f" .. {fmt_month(train_end)}, and that --asset matches the file.")
    if len(test) == 0:
        raise ValueError(
            f"Test set is empty. Check that your data covers "
            f"{fmt_month(test_start)} .. "
            f"{fmt_month(test_end) if test_end else 'the end of sample'}, "
            f"and that --asset matches the file.")
    return train, test


# ==============================================================================
# 4.  HAR-RV OLS ESTIMATION  (per horizon)
# ==============================================================================

def newey_west_1994_lag(nobs: int) -> int:
    """
    Newey & West (1994) automatic bandwidth: L = floor(4 * (T/100)^(2/9)).

    This is the data-driven part of the rule -- it targets serial correlation
    that is present in the series itself, independently of any overlap in the
    dependent variable. For T ~ 3.3k it returns 8.
    """
    return int(np.floor(4.0 * (nobs / 100.0) ** (2.0 / 9.0)))


def nw_bandwidth(h: int, nobs: int, rule: str = "auto") -> int:
    """
    Newey-West bandwidth for horizon h on a training sample of `nobs` rows.

    'auto'    : max(2*(h-1), newey_west_1994_lag(nobs)).
                2*(h-1) covers the MA(h-1) autocorrelation MECHANICALLY
                induced by the overlapping h-day target (Patton & Sheppard,
                2009). It is a floor: at h=1 it is 0, which would imply the
                residuals are serially uncorrelated. They are not -- Ljung-Box
                rejects on this sample -- so the plug-in rule takes over and
                the h=1 regression gets a genuine HAC covariance instead of a
                merely heteroskedasticity-robust one. For h=5 and h=22 the
                overlap floor (8 and 42) dominates, so those horizons keep the
                conventional bandwidth exactly.
    'overlap' : the pure 2*(h-1) convention, for robustness checks.
    """
    overlap = HORIZONS[h]["overlap_lag"]
    if rule == "overlap":
        return overlap
    return max(overlap, newey_west_1994_lag(nobs))


def fit_har_rv(train: pd.DataFrame, h: int):
    """
    Estimate HAR-RV by OLS for forecast horizon h -- in LEVELS by default, or
    on the log scale under --log:

        raw  :  Y_t^(h) = b0 + b_d*RV_d   + b_w*RV_w   + b_m*RV_m   + e_t
        --log:  Y_t^(h) = b0 + b_d*x_d    + b_w*x_w    + b_m*x_m    + e_t
                where each x is ln(mean RV over its window) and Y^(h) is
                ln(mean RV over the h-day forward window) -- log outside.

    Standard errors are Newey-West (1987) HAC with the bandwidth from
    nw_bandwidth(). The HAC estimator is doing more work in levels than under
    --log: raw-RV residuals are strongly heteroskedastic, and Newey-West
    corrects for that as well as for the autocorrelation. Logging shrinks the
    heteroskedasticity a long way (compare the Breusch-Pagan statistics
    between the two modes) but does not remove the serial correlation, so HAC
    is still the right covariance under --log.

    Caution when reading the code: statsmodels' cov_type="HAC" with maxlags=0
    is NOT classical OLS -- the Bartlett kernel retains only the j=0 term, so
    it reduces exactly to White/HC1. That is robust, but it is not a HAC
    correction, which is why h=1 no longer runs with L=0.

    The bandwidth actually used is written back into HORIZONS[h]["nw_lag"] so
    the tables and figures report what was estimated rather than a constant.
    """
    nw_lag = nw_bandwidth(h, len(train), NW_LAG_RULE)
    HORIZONS[h]["nw_lag"] = nw_lag
    Y = train["Y_h"]
    X = sm.add_constant(train[["RV_d", "RV_w", "RV_m"]])
    return OLS(Y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": nw_lag, "use_correction": True},
    )


# ==============================================================================
# 5.  LOSS FUNCTIONS
#     Raw mode : everything on the RAW RV scale.
#     --log    : MSE/MAE in ln(RV) units (the scale the model is fitted on),
#                QLIKE after exp() because QLIKE is only defined for variances.
# ==============================================================================

def mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean((actual - predicted) ** 2))

def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))

def qlike(actual: np.ndarray, predicted: np.ndarray, floor: float):
    """
    QLIKE loss (Patton, 2011) on raw RV. Smaller = better.

        QLIKE = mean( RV/RV_hat - ln(RV/RV_hat) - 1 )

    Inputs are already variances, so there is no exp() step -- unlike the
    log-scale version, which had to exponentiate first.

    A levels OLS can predict RV_hat <= 0, where QLIKE is undefined. Such
    forecasts are FLOORED at `floor` rather than discarded: dropping them
    would remove the worst forecasts and bias QLIKE downward. The count is
    returned so the reader can judge how much the number leans on the floor.
    """
    actual    = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    n_bad     = int((predicted <= 0).sum())
    pred_safe = np.where(predicted <= 0, floor, predicted)
    valid     = actual > 0
    ratio     = actual[valid] / pred_safe[valid]
    return float(np.mean(ratio - np.log(ratio) - 1)), n_bad

def back_transform(pred_log: np.ndarray, resid_var: float) -> np.ndarray:
    """
    Map a log-scale forecast back to a variance, with the lognormal (Jensen)
    correction:

        E[RV | F] = exp( E[ln RV | F] + sigma^2 / 2 )

    Taking exp() alone returns the conditional MEDIAN, which understates the
    conditional mean by exp(sigma^2/2) -- a systematic downward bias, not
    noise, and it does not average out over the sample. `resid_var` is the
    training residual variance of the log regression; the correction is exact
    when the log residuals are Gaussian and a good approximation otherwise
    (Andersen, Bollerslev, Diebold & Labys, 2003).

    With the log-of-mean target this lands on exactly the right object: the
    regressand is ln(arithmetic forward mean), so the corrected forecast
    estimates E[(1/h) * Sum RV_{t+k} | F] -- the same conditional mean the raw
    levels regression estimates directly. The correction fixes median -> mean;
    it does NOT convert one kind of average into another, which is why the
    aggregation convention has to be right in the first place.

    The result is strictly positive, which is why --log cannot produce the
    non-positive variance forecasts the levels regression does.
    """
    return np.exp(np.asarray(pred_log, dtype=float) + resid_var / 2.0)


def compute_metrics(actual, predicted, floor: float,
                    resid_var: float = 0.0) -> dict:
    """
    Losses for one split. `actual` / `predicted` are on the MODELLING scale.

    Raw mode: MSE, MAE and QLIKE are all raw-RV losses, exactly as before.

    --log mode:
      * MSE / MAE are in ln(RV) units -- the scale the model is fitted and
        should be judged on. Squared error in logs is a relative-error
        criterion and is not spike-dominated the way levels MSE is.
      * QLIKE is undefined on logs (it needs a variance ratio), so both sides
        are exponentiated first: actual exp(Y_h), forecast back_transform(.).
        Because the target is ln(arithmetic forward mean), exp(Y_h) is the
        raw-mode target exactly -- at h=1 the pair is literally (RV, RV_hat).
      * MSE_RV / MAE_RV repeat the errors on the back-transformed variance
        scale, so a --log run can be ranked against a raw run. With the
        log-of-mean convention this holds at EVERY horizon: both modes are
        predicting the same arithmetic forward mean and are scored against
        identical actuals, so the numbers sit side by side with no Jensen
        caveat. (Under mean-of-logs they would not; see
        build_horizon_target.)
      * QLIKE_naive is QLIKE without the Jensen correction, reported so the
        size of the correction is visible rather than taken on trust.
    """
    if not LOG_MODE:
        # A negative variance forecast is not admissible, so MSE and MAE are
        # measured against the clipped forecast max(pred, 0) -- the best
        # feasible prediction the model could have issued. QLIKE cannot use
        # zero (it is infinite there) and keeps the positive floor applied
        # inside qlike(). The deep models are scored the same way, so the two
        # tables stay comparable number for number.
        pred_clipped = np.maximum(np.asarray(predicted, dtype=float), 0.0)
        q, n_bad = qlike(actual, predicted, floor)
        return {"MSE"       : mse(actual, pred_clipped),
                "MAE"       : mae(actual, pred_clipped),
                "QLIKE"     : q,
                "n_neg_pred": n_bad,
                "pct_neg"   : 100.0 * n_bad / len(predicted)}

    actual_lvl    = np.exp(np.asarray(actual, dtype=float))
    pred_naive    = np.exp(np.asarray(predicted, dtype=float))
    pred_lvl      = back_transform(predicted, resid_var)
    q,       n_bad = qlike(actual_lvl, pred_lvl,   floor)
    q_naive, _     = qlike(actual_lvl, pred_naive, floor)
    return {"MSE"        : mse(actual, predicted),      # ln(RV) units
            "MAE"        : mae(actual, predicted),      # ln(RV) units
            "QLIKE"      : q,                           # back-transformed
            "QLIKE_naive": q_naive,
            "MSE_RV"     : mse(actual_lvl, pred_lvl),
            "MAE_RV"     : mae(actual_lvl, pred_lvl),
            "n_neg_pred" : n_bad,                       # 0 by construction
            "pct_neg"    : 100.0 * n_bad / len(predicted)}


# ==============================================================================
# 6.  DIAGNOSTICS
# ==============================================================================

def residual_diagnostics(result, lags: int = 20) -> dict:
    """
    Residual diagnostics on the training sample.

    Breusch-Pagan is included specifically because of the move to levels:
    raw-RV residual variance scales with the level of RV, so a large BP
    statistic is expected, not a sign of a coding error. It confirms that
    the HAC/robust covariance is doing necessary work and that plain OLS
    standard errors would be understated.

    Under --log the same statistics are computed on the log-scale residuals,
    where they read very differently: BP falls by orders of magnitude and
    Jarque-Bera improves sharply, because logging is what turns the
    multiplicative volatility-of-volatility into something close to additive
    noise. Comparing the two modes' diagnostics is the cleanest way to see
    what the transform buys.
    """
    residuals     = result.resid
    dw            = durbin_watson(residuals)
    lb            = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    jb_stat, jb_p = stats.jarque_bera(residuals)
    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, result.model.exog)
    return {
        "DW"      : dw,
        "LB_stat" : float(lb["lb_stat"].iloc[0]),
        "LB_p"    : float(lb["lb_pvalue"].iloc[0]),
        "JB_stat" : float(jb_stat),
        "JB_p"    : float(jb_p),
        "BP_stat" : float(bp_stat),
        "BP_p"    : float(bp_p),
        "Skew"    : float(stats.skew(residuals)),
        "Kurt"    : float(stats.kurtosis(residuals)),
    }


# ==============================================================================
# 7.  PRINT TABLES
# ==============================================================================

def print_section(title: str):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def print_split_info(train: pd.DataFrame, test: pd.DataFrame, h: int):
    print(f"\n  -- Split ({ASSET} calendar, follows Dataset_"
          f"{ASSET.capitalize()}) ------------")
    print(f"  {'Set':<8} {'Rows':>6}  {'Start':>12}  {'End':>12}  {'Window'}")
    print(f"  {THIN[:64]}")
    print(f"  {'Train':<8} {len(train):>6}  "
          f"{str(train.index[0].date()):>12}  {str(train.index[-1].date()):>12}  "
          f"{TRAIN_LABEL}")
    print(f"  {'Test':<8} {len(test):>6}  "
          f"{str(test.index[0].date()):>12}  {str(test.index[-1].date()):>12}  "
          f"{TEST_LABEL}")
    print(f"  {THIN[:64]}")
    if not EMBARGO:
        print(f"  ! --no-embargo: the last {h - 1} training row(s) carry targets "
              f"that average")
        print(f"    test-period RV, and the test row count follows where the file "
              f"ends rather")
        print(f"    than --test-end. Not comparable to the deep models.")
    elif h == 1:
        print(f"  Embargo: nothing to drop at h=1 -- the target is the row "
              f"itself.")
    else:
        print(f"  Embargo: {h - 1} row(s) trimmed from each window edge, the "
              f"origins whose")
        print(f"           {h}-day target would close past the boundary. Each "
              f"window keeps")
        print(f"           (its rows) - {h} + 1, which is exactly what "
              f"Dataset_{ASSET.capitalize()}")
        print(f"           enumerates for pred_len={h}.")
    print(f"  Note: OLS has no hyperparameters, so the deep models' validation")
    print(f"        months are folded into training. Test window is identical")
    print(f"        to the DL models' for a like-for-like comparison.\n")

def print_estimation_table(result, h: int):
    nw_lag   = HORIZONS[h]["nw_lag"]
    hlabel   = HORIZONS[h]["label"]
    params, bse   = result.params, result.bse
    tvals, pvals  = result.tvalues, result.pvalues
    ci       = result.conf_int(alpha=0.05)
    name_map = var_labels()

    overlap  = HORIZONS[h]["overlap_lag"]
    binds    = "overlap floor" if nw_lag == overlap else "NW-1994 plug-in"
    hdr_nw   = f"L={nw_lag} [{binds}; 2*(h-1)={overlap}]"
    print_section(f"HAR-RV [{scale_banner()}]  [{hlabel}]  "
                  f"Newey-West HAC  {hdr_nw}")
    print(f"{'Variable':<22} {'Coeff':>10} {'Std.Err':>10} {'t-stat':>10} "
          f"{'p-value':>10}  {'95% CI'}")
    print(THIN)
    for var in params.index:
        sig = ("***" if pvals[var] < 0.01 else "**" if pvals[var] < 0.05
               else "*" if pvals[var] < 0.10 else "")
        print(f"{name_map.get(var, var):<22} {params[var]:>10.4f} {bse[var]:>10.4f} "
              f"{tvals[var]:>10.3f} {pvals[var]:>10.4f}  "
              f"[{ci.loc[var,0]:7.4f}, {ci.loc[var,1]:7.4f}]{sig}")
    print(THIN)
    print("Significance: *** p<0.01  ** p<0.05  * p<0.10")
    if LOG_MODE:
        print("Coefficients are elasticities (ln RV on ln RV), so b_d+b_w+b_m")
    else:
        print("Coefficients are unit-free (RV regressed on RV), so b_d+b_w+b_m")
    print("near 1 indicates a near-unit-root / long-memory persistence.")
    b_sum = float(sum(result.params[v] for v in ("RV_d", "RV_w", "RV_m")))
    print(f"  b_d+b_w+b_m  : {b_sum:.4f}")
    print(f"\n  R2           : {result.rsquared:.4f}")
    print(f"  Adj. R2      : {result.rsquared_adj:.4f}")
    print(f"  F-statistic  : {result.fvalue:.3f}   [p = {result.f_pvalue:.4f}]")
    print(f"  Observations : {int(result.nobs):,}")
    print(f"  AIC          : {result.aic:.3f}   BIC : {result.bic:.3f}")

def print_metrics_by_horizon(all_metrics: dict):
    print_section(f"OUT-OF-SAMPLE FORECAST EVALUATION -- ALL HORIZONS  "
                  f"[{scale_banner()}]  (Test {TEST_LABEL})")
    print(f"  {'Metric':<12}", end="")
    for h in HORIZONS:
        print(f"  {HORIZONS[h]['label']:>20}", end="")
    print()
    print(THIN)

    # Row label carries the units, because under --log the rows of this one
    # table are deliberately on two different scales.
    rows = ([("MSE  [ln]", "MSE"), ("MAE  [ln]", "MAE"),
             ("QLIKE [RV]", "QLIKE"), ("MSE_RV", "MSE_RV"), ("MAE_RV", "MAE_RV")]
            if LOG_MODE else
            [("MSE", "MSE"), ("MAE", "MAE"), ("QLIKE", "QLIKE")])
    for label, key in rows:
        print(f"  {label:<12}", end="")
        for h in HORIZONS:
            print(f"  {all_metrics[h]['test'][key]:>20.6f}", end="")
        print()

    print(f"  {'neg pred':<12}", end="")
    for h in HORIZONS:
        m = all_metrics[h]["test"]
        cell = f"{m['n_neg_pred']} ({m['pct_neg']:.1f}%)"
        print(f"  {cell:>20}", end="")
    print()
    print(THIN)
    if LOG_MODE:
        print("  Note: MSE/MAE marked [ln] are in ln(RV) units -- the scale the")
        print("        model is fitted on -- and are NOT comparable to raw-scale")
        print("        losses. Squared error in logs is a relative criterion, so")
        print("        it is far less spike-dominated than levels MSE.")
        print("        QLIKE needs variances, so it is evaluated after exp() with")
        print("        the Jensen correction; MSE_RV/MAE_RV are the same forecasts")
        print("        on that back-transformed variance scale.")
        print("        MSE_RV/MAE_RV/QLIKE ARE directly comparable to a raw-scale")
        print("        run at EVERY horizon: the target is ln(arithmetic forward")
        print("        mean), so exp() recovers precisely the quantity raw mode")
        print("        predicts, and both modes are scored against identical")
        print("        actuals. (A mean-of-logs target would instead give a")
        print("        geometric forward mean and break that at h>1.)")
        print("        'neg pred' is 0 by construction here -- exp() cannot return")
        print("        a non-positive variance. That is the main practical gain")
        print("        over the levels fit.")
    else:
        print("  Note: All metrics on the RAW RV scale -- NOT comparable to the")
        print("        ln(RV)-scale losses produced by --log.")
        print("        MSE in levels is dominated by a few spike days; MAE and")
        print("        QLIKE (Patton, 2011) are the more robust rankings here.")
        print("        'neg pred' counts non-positive forecasts, floored before")
        print("        QLIKE. A large count means QLIKE leans on that floor.")

def print_diagnostics(diag: dict, h: int):
    print_section(f"RESIDUAL DIAGNOSTICS  [h={h}, {HORIZONS[h]['label']}, "
                  f"{scale_name()} residuals, "
                  f"Training Sample {TRAIN_LABEL}]")
    print(f"  {'Statistic':<30} {'Value':>12}")
    print(THIN)
    for k, v in diag.items():
        print(f"  {k:<30} {v:>12.4f}")
    print(THIN)
    if LOG_MODE:
        print("  BP = Breusch-Pagan, on ln(RV) residuals. Logging absorbs most of")
        print("  the level-dependent error variance, so BP here is far smaller")
        print("  than in the levels fit -- that gain is the point of --log. HAC")
        print("  is still used: it also covers the serial correlation, which the")
        print("  transform does NOT remove.")
    else:
        print("  BP = Breusch-Pagan. A large statistic is EXPECTED in levels:")
        print("  RV residual variance scales with the level of RV. HAC standard")
        print("  errors remain valid; plain OLS SEs would be understated.")
    if h > 1:
        print(f"  DW/LB: rejection is MECHANICAL at h>1 -- the overlapping "
              f"{h}-day\n  target induces MA({h-1}) residuals. The L="
              f"{HORIZONS[h]['nw_lag']} bandwidth covers it.")
    else:
        print("  DW/LB: h=1 has no overlap, so any rejection here is genuine")
        print(f"  serial correlation in the data -- the reason this horizon "
              f"uses\n  L={HORIZONS[h]['nw_lag']} rather than L=0.")


# ==============================================================================
# 8.  FIGURES
# ==============================================================================

def _ylab(h):
    # Under --log the log is OUTSIDE the bar: ln of the mean, not the mean of
    # logs. \overline{\ln RV} would state the wrong specification.
    if LOG_MODE:
        return (r"$\ln RV_t$" if h == 1
                else rf"$\ln \overline{{RV}}_{{t,t+{h}}}$")
    return (r"$RV_t$" if h == 1
            else rf"$\overline{{RV}}_{{t,t+{h}}}$")

def figure_multi_horizon_forecast(results_dict):
    horizons  = list(HORIZONS.keys())
    fig, axes = plt.subplots(len(horizons), 1,
                             figsize=(11, 3.8 * len(horizons)), sharex=False)
    fig.subplots_adjust(hspace=0.52)
    panel_letters = ["(A)", "(B)", "(C)"]

    for idx, h in enumerate(horizons):
        ax     = axes[idx]
        t_idx  = results_dict[h]["y_hat_test"].index
        actual = results_dict[h]["df_test"]["Y_h"]
        pred   = results_dict[h]["y_hat_test"]

        ax.plot(t_idx, actual.values, color=C_ACTUAL, lw=0.9, alpha=0.85,
                label="Actual $Y^{(h)}$")
        ax.plot(t_idx, pred.values, color=C_H[h], lw=1.0, ls="--", alpha=0.90,
                label=f"HAR-RV  h={h}")
        ax.fill_between(t_idx, actual.values, pred.values,
                        color="#aaaaaa", alpha=0.18, label="Error")
        # Zero line matters in levels, where forecasts can cross it into
        # meaningless negative variance. On the log scale zero is just RV=1,
        # an arbitrary point the series straddles anyway, so it is dropped.
        if not LOG_MODE:
            ax.axhline(0, color="black", lw=0.6, alpha=0.6)
        ax.set_ylabel(_ylab(h), fontsize=10)
        ax.set_title(f"{panel_letters[idx]}  {HORIZONS[h]['label']} -- "
                     f"Out-of-Sample Forecast ({TEST_LABEL})",
                     loc="left", pad=4)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.legend(loc="upper right", fontsize=8)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.suptitle(f"HAR-RV ({scale_name()}) Multi-Horizon Forecasts: "
                 f"h = 1, 5, 22  (Corsi, 2009)\n"
                 f"{ASSET} calendar -- Train: {TRAIN_LABEL}  |  "
                 f"Test: {TEST_LABEL}",
                 fontsize=12, fontweight="bold", y=1.01)
    stem = f"{out_prefix()}_fig1_multihoriz_forecast"
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT_DIR}/{stem}.{ext}", bbox_inches="tight")
    print(f"  -> Saved: {stem}.pdf / .png")
    plt.close(fig)


def figure_loss_comparison(all_metrics: dict):
    metrics_list  = ["MSE", "MAE", "QLIKE"]
    horizons      = list(HORIZONS.keys())
    labels        = [f"h={h}" for h in horizons]
    colors        = [C_H[h] for h in horizons]
    panel_letters = ["(A)", "(B)", "(C)"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.subplots_adjust(wspace=0.4)
    for idx, metric in enumerate(metrics_list):
        ax   = axes[idx]
        vals = [all_metrics[h]["test"][metric] for h in horizons]
        bars = ax.bar(labels, vals, color=colors, alpha=0.75,
                      edgecolor="white", width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.015,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)
        # QLIKE is always a variance-scale loss; MSE/MAE follow the fit scale.
        unit = ("ln(RV) scale" if LOG_MODE and metric != "QLIKE"
                else "RV scale")
        ax.set_title(f"{panel_letters[idx]}  {metric}", loc="left", pad=3)
        ax.set_ylabel(f"{metric}  ({unit})")
        ax.set_xlabel("Forecast Horizon")
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    fig.suptitle(f"HAR-RV ({scale_name()}): Out-of-Sample Loss by Horizon -- "
                 f"Test {TEST_LABEL}",
                 fontsize=12, fontweight="bold")
    stem = f"{out_prefix()}_fig2_loss_comparison"
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT_DIR}/{stem}.{ext}", bbox_inches="tight")
    print(f"  -> Saved: {stem}.pdf / .png")
    plt.close(fig)


def figure_coefficients_across_horizons(results_dict: dict):
    horizons   = list(HORIZONS.keys())
    coef_names = coef_panel_labels()
    x_pos         = np.arange(len(horizons))
    x_labels      = [f"h={h}" for h in horizons]
    panel_letters = ["(A)", "(B)", "(C)", "(D)"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)
    for i, key in enumerate(coef_names):
        ax     = axes[i // 2, i % 2]
        coeffs = [results_dict[h]["result"].params[key] for h in horizons]
        errs   = [1.96 * results_dict[h]["result"].bse[key] for h in horizons]
        ax.errorbar(x_pos, coeffs, yerr=errs, fmt="o-", color=C_TRAIN,
                    lw=1.2, ms=6, capsize=4, capthick=1, elinewidth=0.9,
                    label="Estimate +/- 1.96*SE")
        ax.axhline(0, color="gray", lw=0.7, ls="--")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_title(f"{panel_letters[i]}  {coef_names[key]}", loc="left", pad=3)
        ax.set_ylabel("Coefficient")
        ax.set_xlabel("Forecast Horizon")
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.legend(fontsize=8)

    fig.suptitle(f"HAR-RV ({scale_name()}): Coefficient Paths Across Horizons  "
                 f"(h = 1, 5, 22)",
                 fontsize=12, fontweight="bold", y=1.01)
    stem = f"{out_prefix()}_fig3_coeff_paths"
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT_DIR}/{stem}.{ext}", bbox_inches="tight")
    print(f"  -> Saved: {stem}.pdf / .png")
    plt.close(fig)


def figure_residual_diagnostics_multihoriz(results_dict: dict):
    horizons      = list(HORIZONS.keys())
    max_lag       = 44
    panel_letters = ["(A)", "(B)", "(C)"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.subplots_adjust(wspace=0.38)
    for idx, h in enumerate(horizons):
        ax     = axes[idx]
        resid  = results_dict[h]["result"].resid
        n      = len(resid)
        acf_sq = [pd.Series(resid.values ** 2).autocorr(lag=k)
                  for k in range(1, max_lag + 1)]
        ci_b   = 1.96 / np.sqrt(n)
        nw_l   = HORIZONS[h]["nw_lag"]

        ax.bar(range(1, max_lag + 1), acf_sq, color=C_H[h], alpha=0.70, width=0.7)
        ax.axhline( ci_b, color="black", ls="--", lw=0.8, label="95% CI")
        ax.axhline(-ci_b, color="black", ls="--", lw=0.8)
        ax.axhline(0, color="black", lw=0.5)
        if nw_l > 0:
            ax.axvline(nw_l, color=C_TEST, lw=1.0, ls=":", label=f"NW lag L={nw_l}")
        ax.set_title(f"{panel_letters[idx]}  h={h}  [{HORIZONS[h]['label']}]",
                     loc="left", pad=3)
        ax.set_xlabel("Lag (days)")
        ax.set_ylabel("ACF (squared residuals)")
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.legend(fontsize=8)

    fig.suptitle(f"HAR-RV ({scale_name()}): ACF of Squared Residuals -- Training "
                 f"Sample {TRAIN_LABEL}  "
                 f"(vertical line = NW bandwidth)",
                 fontsize=11, fontweight="bold")
    stem = f"{out_prefix()}_fig4_acf_residuals"
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT_DIR}/{stem}.{ext}", bbox_inches="tight")
    print(f"  -> Saved: {stem}.pdf / .png")
    plt.close(fig)


def figure_scatter_grid(results_dict: dict):
    horizons      = list(HORIZONS.keys())
    panel_letters = ["(A)", "(B)", "(C)"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.subplots_adjust(wspace=0.38)
    for idx, h in enumerate(horizons):
        ax   = axes[idx]
        act  = results_dict[h]["df_test"]["Y_h"].values
        pred = results_dict[h]["y_hat_test"].values
        lo   = min(act.min(), pred.min())
        hi   = max(act.max(), pred.max())

        ax.scatter(pred, act, s=10, color=C_H[h], alpha=0.40,
                   edgecolors="none", label="Test obs.")
        ax.plot([lo, hi], [lo, hi], color="black", lw=1.0, ls="--", label="45 deg line")
        slope, intercept, r, *_ = stats.linregress(pred, act)
        xs = np.array([lo, hi])
        ax.plot(xs, intercept + slope * xs, color=C_TEST, lw=1.0,
                label=f"OLS fit  R2={r**2:.3f}")
        # In levels the negative-forecast region is meaningful; shade it. Under
        # --log a negative axis value is just RV_hat < 1, which is most of the
        # sample and no kind of failure, so the shading is suppressed -- keeping
        # it would flag ordinary forecasts as invalid.
        if lo < 0 and not LOG_MODE:
            ax.axvspan(lo, 0, color="#d73027", alpha=0.08)
            ax.axvline(0, color="#d73027", lw=0.8, ls=":", label="RV_hat = 0")
        ax.set_xlabel(f"Predicted  {_ylab(h)}", fontsize=9)
        ax.set_ylabel(f"Actual     {_ylab(h)}", fontsize=9)
        ax.set_title(f"{panel_letters[idx]}  h={h}  [{HORIZONS[h]['label']}]",
                     loc="left", pad=3)
        ax.legend(fontsize=8)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    fig.suptitle(f"HAR-RV ({scale_name()}): Actual vs. Predicted -- Test "
                 f"{TEST_LABEL}  (all horizons)",
                 fontsize=12, fontweight="bold")
    stem = f"{out_prefix()}_fig5_scatter_grid"
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT_DIR}/{stem}.{ext}", bbox_inches="tight")
    print(f"  -> Saved: {stem}.pdf / .png")
    plt.close(fig)


# ==============================================================================
# 9.  EXPORT
# ==============================================================================

def export_all(results_dict: dict, all_metrics: dict):
    """
    Write fitted values, metrics and parameter tables.

    Filenames are prefixed by out_prefix(), so a --log run writes har_rv_log_*
    alongside (not on top of) the raw-scale har_rv_* files. Y_h / fitted /
    residual are on the fit scale; under --log two extra columns give the
    back-transformed variance forecast so the CSV is usable without re-deriving
    the Jensen correction downstream.
    """
    pfx = out_prefix()

    # -- 9a. Fitted values (one CSV per horizon) ------------------------------
    for h, res in results_dict.items():
        frames = []
        for split, dkey, pkey in (("train", "df_train", "y_hat_train"),
                                  ("test",  "df_test",  "y_hat_test")):
            part = res[dkey][["Y_h"]].copy()
            part["fitted"]   = res[pkey].values
            part["residual"] = part["Y_h"] - part["fitted"]
            if LOG_MODE:
                # Same objects the QLIKE / MSE_RV numbers are computed from.
                part["actual_RV"]  = np.exp(part["Y_h"])
                part["fitted_RV"]  = back_transform(part["fitted"].values,
                                                    res["resid_var"])
            part["split"] = split
            frames.append(part)
        full  = pd.concat(frames).sort_index()
        fname = f"{pfx}_h{h:02d}_fitted.csv"
        full.to_csv(f"{OUT_DIR}/{fname}")
        print(f"  -> Saved: {fname}")

    # -- 9b. Consolidated metrics table ---------------------------------------
    # 'scale' records which scale MSE/MAE are in, so raw and --log runs can be
    # concatenated later without losing track of the units.
    extra = ["QLIKE_naive", "MSE_RV", "MAE_RV"] if LOG_MODE else []
    rows  = []
    for h in results_dict:
        for split in ["train", "test"]:
            m   = all_metrics[h][split]
            row = {"horizon": h, "split": split, "scale": scale_tag(),
                   "MSE": m["MSE"], "MAE": m["MAE"], "QLIKE": m["QLIKE"]}
            row.update({k: m[k] for k in extra})
            row.update({"n_neg_pred": m["n_neg_pred"], "pct_neg": m["pct_neg"]})
            rows.append(row)
    pd.DataFrame(rows).to_csv(f"{OUT_DIR}/{pfx}_all_metrics.csv", index=False)
    print(f"  -> Saved: {pfx}_all_metrics.csv")

    # -- 9c. Parameter tables (one CSV per horizon) ---------------------------
    for h, res in results_dict.items():
        result = res["result"]
        ci     = result.conf_int()
        pd.DataFrame({"Coefficient": result.params, "Std_Error": result.bse,
                      "t_stat": result.tvalues, "p_value": result.pvalues,
                      "CI_lower": ci[0], "CI_upper": ci[1]}
                     ).to_csv(f"{OUT_DIR}/{pfx}_h{h:02d}_params.csv")
        print(f"  -> Saved: {pfx}_h{h:02d}_params.csv")


# ==============================================================================
# 10.  MAIN PIPELINE
# ==============================================================================

def main(argv=None):
    # -- 10.0  Resolve command-line configuration -----------------------------
    args = parse_args(argv)

    global DATA_FILE, OUT_DIR, ASSET
    global TRAIN_START, TRAIN_END, TEST_START, TEST_END
    global QLIKE_FLOOR_FRAC, NW_LAG_RULE, LOG_MODE, EMBARGO
    DATA_FILE        = resolve_data_path(args.data)
    OUT_DIR          = args.outdir
    ASSET            = args.asset
    QLIKE_FLOOR_FRAC = args.qlike_floor_frac
    NW_LAG_RULE      = args.nw_lag_rule
    LOG_MODE         = args.log
    EMBARGO          = args.embargo
    (TRAIN_START, TRAIN_END,
     TEST_START, TEST_END) = resolve_split(args)
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\n{SEP}")
    print(f"  HAR-RV MULTI-HORIZON MODEL  --  Corsi (2009)  --  {scale_banner()}")
    print(f"  Realized Variance  |  Horizons: h = 1, 5, 22  |  "
          f"{ASSET} calendar")
    if LOG_MODE:
        print("  --log: ln( arithmetic mean of RV ) on BOTH sides -- log outside")
        print("         the sums. MSE/MAE in ln(RV) units; QLIKE after exp().")
    else:
        print("  Aggregation: arithmetic averages of RV on BOTH sides -- no logs")
        print("  (pass --log to run the same pipeline on the ln(RV) scale)")
    print(SEP)

    # -- 10.1  Load base features ---------------------------------------------
    print(f"\n[1] Loading data from '{DATA_FILE}' ...")
    if os.path.abspath(DATA_FILE) != os.path.abspath(args.data):
        print(f"    (resolved from requested '{args.data}')")
    df_base = load_base_features(DATA_FILE)
    print(f"  Full series  : {df_base.index[0].date()} -> {df_base.index[-1].date()}")
    print(f"  Total rows   : {len(df_base):,}  (before per-horizon NaN drop)")
    print(f"  RV summary   : mean={df_base['RV_raw'].mean():.4f}  "
          f"sd={df_base['RV_raw'].std():.4f}  min={df_base['RV_raw'].min():.4f}  "
          f"max={df_base['RV_raw'].max():.4f}")
    print(f"  RV skew      : {df_base['RV_raw'].skew():.2f}   "
          f"excess kurt: {df_base['RV_raw'].kurt():.2f}")
    if LOG_MODE:
        # The reason to log: raw RV is violently right-skewed and fat-tailed,
        # ln(RV) is close to Gaussian. These two lines quantify the change on
        # the sample actually being used.
        print(f"  ln(RV) summ. : mean={df_base['RV'].mean():.4f}  "
              f"sd={df_base['RV'].std():.4f}  min={df_base['RV'].min():.4f}  "
              f"max={df_base['RV'].max():.4f}")
        print(f"  ln(RV) skew  : {df_base['RV'].skew():.2f}   "
              f"excess kurt: {df_base['RV'].kurt():.2f}   "
              f"<- modelling series")
    print(f"  Split ({ASSET}): Train {TRAIN_LABEL}  |  Test {TEST_LABEL}")

    # Floor is a raw-variance quantity, so it always comes from RV_raw.
    floor = QLIKE_FLOOR_FRAC * float(
        df_base.loc[month_mask(df_base.index, TRAIN_START, TRAIN_END),
                    "RV_raw"].mean())
    print(f"  QLIKE floor for non-positive forecasts: {floor:.3e}"
          + ("  (cannot bind under --log)" if LOG_MODE else ""))

    # -- 10.2  Per-horizon loop ------------------------------------------------
    results_dict, all_metrics = {}, {}

    for h in HORIZONS:
        print(f"\n{'-'*72}")
        print(f"  HORIZON  h = {h}  [{HORIZONS[h]['label']}]")
        print(f"{'-'*72}")

        df_h        = build_horizon_target(df_base, h)
        train, test = split_by_calendar(df_h, h)
        print_split_info(train, test, h)

        result = fit_har_rv(train, h)
        print_estimation_table(result, h)

        X_tr = sm.add_constant(train[["RV_d", "RV_w", "RV_m"]])
        X_te = sm.add_constant(test[["RV_d",  "RV_w", "RV_m"]])
        y_hat_train = result.predict(X_tr)
        y_hat_test  = result.predict(X_te)

        # Jensen correction term for --log. Estimated on TRAINING residuals
        # only -- using test residuals would leak the out-of-sample outcome
        # into the forecast. Zero in raw mode, where nothing is exponentiated.
        resid_var = float(result.mse_resid) if LOG_MODE else 0.0

        m_train = compute_metrics(train["Y_h"].values, y_hat_train.values,
                                  floor, resid_var)
        m_test  = compute_metrics(test["Y_h"].values,  y_hat_test.values,
                                  floor, resid_var)

        if LOG_MODE:
            print(f"\n  Back-transform: sigma^2 = {resid_var:.4f}  ->  "
                  f"exp(sigma^2/2) = {np.exp(resid_var / 2.0):.4f}x")
            print(f"    Test QLIKE  {m_test['QLIKE']:.6f} (corrected)  vs  "
                  f"{m_test['QLIKE_naive']:.6f} (naive exp)")
            print(f"    Naive exp() would forecast the conditional MEDIAN and "
                  f"sit ~{100.0 * (np.exp(resid_var / 2.0) - 1.0):.1f}% low "
                  f"as a mean.")
        elif m_test["n_neg_pred"]:
            print(f"\n  ! {m_test['n_neg_pred']} non-positive test forecast(s) "
                  f"({m_test['pct_neg']:.1f}%) -- floored for QLIKE. "
                  f"Expected in levels; --log cannot produce these.")

        print_diagnostics(residual_diagnostics(result), h)

        results_dict[h] = {"result": result, "df_train": train, "df_test": test,
                           "y_hat_train": y_hat_train, "y_hat_test": y_hat_test,
                           "resid_var": resid_var}
        all_metrics[h]  = {"train": m_train, "test": m_test}

    # -- 10.3  Cross-horizon comparison table ----------------------------------
    print_metrics_by_horizon(all_metrics)

    # -- 10.4  Figures ---------------------------------------------------------
    print(f"\n[2] Generating publication figures ...")
    figure_multi_horizon_forecast(results_dict)
    figure_loss_comparison(all_metrics)
    figure_coefficients_across_horizons(results_dict)
    figure_residual_diagnostics_multihoriz(results_dict)
    figure_scatter_grid(results_dict)

    # -- 10.5  Export ----------------------------------------------------------
    print(f"\n[3] Exporting results ...")
    export_all(results_dict, all_metrics)

    # -- 10.6  Summary ---------------------------------------------------------
    print(f"\n{SEP}")
    print(f"  FINAL SUMMARY -- OUT-OF-SAMPLE TEST SET  "
          f"({TEST_LABEL})  [{scale_banner()}]")
    print(THIN)
    mse_hdr = "MSE[ln]" if LOG_MODE else "MSE"
    mae_hdr = "MAE[ln]" if LOG_MODE else "MAE"
    print(f"  {'Horizon':<14} {'NW Lag':>8} {mse_hdr:>12} {mae_hdr:>12} "
          f"{'QLIKE':>12} {'neg pred':>10}")
    print(THIN)
    for h in HORIZONS:
        m = all_metrics[h]["test"]
        print(f"  {HORIZONS[h]['label']:<14} {HORIZONS[h]['nw_lag']:>8}  "
              f"{m['MSE']:>12.6f} {m['MAE']:>12.6f} {m['QLIKE']:>12.6f} "
              f"{m['n_neg_pred']:>10}")
    print(THIN)
    if LOG_MODE:
        print(f"  Back-transformed to the variance scale -- directly comparable to")
        print(f"  a raw-scale run at every horizon (same target, same actuals):")
        print(f"  {'Horizon':<14} {'':>8} {'MSE_RV':>12} {'MAE_RV':>12}")
        for h in HORIZONS:
            m = all_metrics[h]["test"]
            print(f"  {HORIZONS[h]['label']:<14} {'':>8}  "
                  f"{m['MSE_RV']:>12.6f} {m['MAE_RV']:>12.6f}")
        print(THIN)
    if LOG_MODE:
        print(f"""
  Aggregation (--log: average on the RAW scale, then take the log):
    Target     Y^(h) = ln( (1/h) * Sum_k RV_(t+k) )  over the forward window
    Regressors RV_d  = ln( RV_(t-1) )
               RV_w  = ln( mean RV over the previous 5 days )
               RV_m  = ln( mean RV over the previous 22 days )

  Reading the output:
    * MSE/MAE are in ln(RV) units and are NOT comparable to raw-scale losses.
      Re-evaluate any DL baseline on the same scale, or use MSE_RV/MAE_RV.
    * QLIKE is a variance loss, so it is computed after exp() with the
      lognormal Jensen correction exp(sigma^2/2); sigma^2 is the TRAINING
      residual variance. Plain exp() targets the median and runs low.
    * Non-positive forecasts are impossible here: exp(.) > 0 always. The
      QLIKE floor is inert, and 'neg pred' is 0 by construction.
    * Squared error in logs is a relative-error criterion, so single spike
      days no longer dominate the loss the way they do in levels.
    * Breusch-Pagan should be far smaller than in the levels fit -- that is
      the transform working. HAC is still used for the serial correlation,
      which logging does not remove.
    * The log sits OUTSIDE the sum on both sides, so exp(Y^(h)) is exactly
      the raw-mode target and MSE_RV/MAE_RV/QLIKE can be set beside a raw
      run at every horizon, not just h=1. Averaging logs instead would make
      the target a geometric forward mean -- smaller and smoother -- and
      would flatter this mode at h>1 for purely arithmetic reasons.

  Split:""")
    else:
        print(f"""
  Aggregation (raw scale, no logs anywhere):
    Target     Y^(h) = (1/h) * Sum_k RV_(t+k)   over the h-day forward window
    Regressors RV_d  = RV_(t-1)
               RV_w  = mean RV over the previous 5 days
               RV_m  = mean RV over the previous 22 days

  Reading the output:
    * Losses are on the RV scale and are NOT comparable to the ln(RV)-scale
      losses produced by --log. Re-evaluate any DL baseline on this scale.
    * MSE is spike-dominated in levels; prefer MAE and QLIKE for ranking.
    * Non-positive forecasts are possible and are reported per horizon.
    * Breusch-Pagan will reject strongly -- expected, not a bug. HAC SEs
      remain valid; OLS is simply no longer efficient.

  Split:""")
    print(f"""\
    Calendar  {ASSET}   (data_provider/splits.py)
    Train     {TRAIN_LABEL}
    Test      {TEST_LABEL}
    Dataset_{ASSET.capitalize()} (data_loader.py) splits train / val / test on
    exactly these boundaries -- both read them from splits.py. HAR-RV has no
    hyperparameters, so its validation months are folded into the estimation
    sample; the test window is shared, which is what keeps the out-of-sample
    losses comparable. Run the deep models with --data {ASSET} to match.
    Each window is embargoed by h-1 rows at its edges, so no training target
    averages test-period data and both families score (window rows) - h + 1
    forecasts. --no-embargo restores the unembargoed convention.

  References:
    Corsi, F. (2009). A simple approximate long-memory model of realized
      volatility. Journal of Financial Econometrics, 7(2), 174-196.
    Patton, A. J. (2011). Volatility forecast comparison using imperfect
      volatility proxies. Journal of Econometrics, 160(1), 246-256.
    Patton, A. J., & Sheppard, K. (2009). Evaluating volatility and
      correlation forecasts. Handbook of Financial Time Series. Springer.
    Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite,
      heteroskedasticity and autocorrelation consistent covariance matrix.
      Econometrica, 55(3), 703-708.
    Newey, W. K., & West, K. D. (1994). Automatic lag selection in covariance
      matrix estimation. Review of Economic Studies, 61(4), 631-653.
    Bollerslev, T., Patton, A. J., & Quaedvlieg, R. (2016). Exploiting
      the errors. Journal of Econometrics, 192(1), 1-18.
    Andersen, T. G., Bollerslev, T., Diebold, F. X., & Labys, P. (2003).
      Modeling and forecasting realized volatility. Econometrica, 71(2),
      579-625.  [log-scale RV modelling; lognormal back-transform]
    Andersen, T. G., Bollerslev, T., & Diebold, F. X. (2007). Roughing it up:
      including jump components in the measurement, modeling, and forecasting
      of return volatility. Review of Economics and Statistics, 89(4),
      701-720.  [log-HAR specification used by --log]
""")
    print(SEP)


if __name__ == "__main__":
    main()
