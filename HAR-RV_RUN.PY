"""
==============================================================================
HAR-RV Model: Multi-Horizon Realized Volatility Forecasting  --  RAW RV SCALE
Based on: Corsi, F. (2009). A simple approximate long-memory model of
          realized volatility. Journal of Financial Econometrics, 7(2), 174-196.

Horizons : h = 1  (daily),  h = 5  (weekly),  h = 22  (monthly)

Split logic follows Dataset_Custom (data_loader.py), which splits
    train 2012-2022  |  val 2023-2024  |  test 2025-end

HAR-RV is OLS: there are no hyperparameters to tune, so a held-out validation
set would serve no purpose. The validation years are therefore folded back
into the estimation sample:
    Train : start of sample - end of 2024  (year <= 2024)  [= DL train + val]
    Test  : 2025 onwards                   (year >= 2025)  [= DL test]
The TEST window is identical to the deep models', so out-of-sample losses
remain directly comparable. Note the deep models never fit on 2023-2024;
they only use it for early stopping / model selection.

------------------------------------------------------------------------------
USAGE (command line / notebook)
------------------------------------------------------------------------------
    python HAR-RV_RUN.PY                         # uses default data/EURUSD-RV.csv
    python HAR-RV_RUN.PY --data EURUSD_RV.CSV    # pick a data file
    python HAR-RV_RUN.PY --data path/to/my_rv.csv --outdir results

In a Jupyter/Colab cell prefix with '!':
    !python HAR-RV_RUN.PY --data EURUSD_RV.CSV

The --data path is resolved leniently: if the exact path is not found the
script also looks inside ./data and matches case-insensitively while treating
'-' and '_' as interchangeable, so 'EURUSD_RV.CSV' resolves to the shipped
'data/EURUSD-RV.csv'. Run `python HAR-RV_RUN.PY --help` for all options.

==============================================================================
CHANGE FROM THE LOG VERSION -- everything now lives on the RAW RV scale.

This is Corsi (2009) eq. 3 in its original form, i.e. a LEVELS regression:

    Target      Y_t^(h) = (1/h) * Sum_{k=1}^{h}  RV_{t+k}
    Regressors  RV_d    =            RV_{t-1}
                RV_w    = (1/5)  * Sum_{k=1}^{5}  RV_{t-k}
                RV_m    = (1/22) * Sum_{k=1}^{22} RV_{t-k}

No logs are taken anywhere. The whole log-of-sum vs mean-of-logs discussion
from the previous version is now moot -- there is no Jensen gap to worry
about because there is no non-linear transform.

SUM vs MEAN is NOT cosmetic any more. On the log scale ln(sum) = ln(mean) +
ln(window), a constant absorbed by the intercept. On the raw scale the two
differ by a MULTIPLICATIVE factor:
  * For the REGRESSORS it is still harmless -- using sums instead of means
    just rescales beta_w by 1/5 and beta_m by 1/22. Fit, residuals, R^2 and
    all losses are identical.
  * For the TARGET it is NOT harmless -- switching to the h-day sum scales
    Y, every coefficient and every residual by h, so MSE scales by h^2 and
    MAE by h. R^2 is unchanged. Corsi's convention is the AVERAGE, used here,
    which also keeps the three horizons on a comparable scale.

CONSEQUENCES OF WORKING IN LEVELS -- read before interpreting output:
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
     variance and undefined under QLIKE. Non-positive forecasts are counted,
     reported per horizon, and floored before QLIKE is evaluated -- they are
     never silently dropped, since dropping the worst forecasts would bias
     QLIKE downward.
  4. Losses are now on the RV scale, so they are NOT comparable to the
     ln(RV)-scale numbers produced by the previous version of this script.
     Any deep-learning baseline must be evaluated on the same scale.

INPUT: the loader accepts either a raw 'RV' column (used as-is) or an
'ln_RV' column (exponentiated back to raw RV once, at load time).
Non-positive RV rows -- exchange holidays with no trading -- are dropped.

HAC bandwidth (Patton & Sheppard, 2009; Bollerslev et al., 2016):
    L = 2*(h-1), applied via Newey-West (1987) Bartlett kernel.
    h=1  -> L=0  (standard OLS SEs; no overlap)
    h=5  -> L=8
    h=22 -> L=42

Metrics   : MSE, MAE, QLIKE (Patton, 2011) -- computed on the RAW RV scale
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


# ==============================================================================
# 0.  CONFIGURATION  (defaults; overridable on the command line -- see parse_args)
# ==============================================================================

DATA_FILE = "data/EURUSD-RV.csv"   # default data set (override with --data)
OUT_DIR   = "."                    # where figures / CSVs are written (--outdir)

# Mirrors data_provider/data_loader.py: VAL_END_YEAR = 2024, so the deep
# models' train (<=2022) and val (2023-2024) years both become HAR training
# data, and the test window (2025-end) is shared.
TRAIN_END_YEAR  = 2024
TEST_START_YEAR = 2025

# Forecast horizons and their Newey-West bandwidths: L = 2*(h-1)
HORIZONS = {
    1  : {"label": "Daily   (h=1)",  "nw_lag":  0},
    5  : {"label": "Weekly  (h=5)",  "nw_lag":  8},
    22 : {"label": "Monthly (h=22)", "nw_lag": 42},
}

LAG_W    = 5    # weekly component window
LAG_M    = 22   # monthly component window
DPI_SAVE = 300

# QLIKE guard: forecasts <= 0 are floored at this multiple of mean training RV
# before QLIKE is evaluated. Reported, never silently dropped.
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

# Populated at load time so printed headers track the actual sample.
SAMPLE_START_YEAR = None
SAMPLE_END_YEAR   = None


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
                    "(Corsi, 2009) on the raw RV scale.",
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
        "--train-end-year", type=int, default=TRAIN_END_YEAR, metavar="YEAR",
        help="Last calendar year (inclusive) of the training sample.")
    p.add_argument(
        "--test-start-year", type=int, default=TEST_START_YEAR, metavar="YEAR",
        help="First calendar year (inclusive) of the test sample.")
    p.add_argument(
        "--qlike-floor-frac", type=float, default=QLIKE_FLOOR_FRAC, metavar="FRAC",
        help="Non-positive forecasts are floored at FRAC * mean(training RV) "
             "before QLIKE is evaluated.")
    return p.parse_args(argv)


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

def load_base_features(filepath: str) -> pd.DataFrame:
    """
    Load the realized-variance series and build the three HAR regressors on
    the RAW RV scale (Corsi 2009, eq. 3):

        RV_d = RV_{t-1}
        RV_w = (1/5)  * Sum_{k=1}^{5}  RV_{t-k}
        RV_m = (1/22) * Sum_{k=1}^{22} RV_{t-k}

    Column handling
    ---------------
    'RV'    -> used directly (this file's format).
    'ln_RV' -> exponentiated once here, so everything downstream is raw.
    Otherwise the first numeric column is taken and ASSUMED to be raw RV;
    a warning is printed, because silently exponentiating the wrong column
    is exactly the bug this loader is written to avoid.

    Non-positive RV rows are dropped: they are exchange holidays with no
    trading rather than genuine zero-variance days, and a zero actual makes
    QLIKE undefined.

    All windows use shift(1), so the information set ends strictly at t-1 --
    no look-ahead. Regressors are IDENTICAL across horizons; only the target
    changes.
    """
    global SAMPLE_START_YEAR, SAMPLE_END_YEAR

    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"Data file not found: '{filepath}'. Pass a valid path with "
            f"--data (e.g. --data data/EURUSD-RV.csv).")

    raw = pd.read_csv(filepath, index_col=0, parse_dates=True)
    raw.index.name = "date"

    if "RV" in raw.columns:
        s = raw["RV"].sort_index().dropna().astype(float)
        print("  Column 'RV' found -> used directly (already raw variance).")
    elif "ln_RV" in raw.columns:
        s = np.exp(raw["ln_RV"].sort_index().dropna().astype(float))
        print("  Column 'ln_RV' found -> exponentiated once to raw RV.")
    else:
        col = raw.select_dtypes("number").columns[0]
        s = raw[col].sort_index().dropna().astype(float)
        print(f"  WARNING: no 'RV'/'ln_RV' column. Using '{col}' AS RAW RV.")

    n_bad = int((s <= 0).sum())
    if n_bad:
        print(f"  Dropped {n_bad} non-positive RV row(s) (non-trading days): "
              f"{[str(d.date()) for d in s.index[s <= 0]]}")
        s = s[s > 0]

    SAMPLE_START_YEAR = int(s.index[0].year)
    SAMPLE_END_YEAR   = int(s.index[-1].year)

    df = pd.DataFrame({"RV": s})
    df["RV_d"] = s.shift(1)
    df["RV_w"] = s.shift(1).rolling(LAG_W).mean()
    df["RV_m"] = s.shift(1).rolling(LAG_M).mean()
    return df   # NaN rows dropped per-horizon after target is attached


# ==============================================================================
# 2.  HORIZON-SPECIFIC TARGET CONSTRUCTION
# ==============================================================================

def build_horizon_target(df_base: pd.DataFrame, h: int) -> pd.DataFrame:
    """
    Construct the h-day forward AVERAGE realized variance as the dependent
    variable -- Corsi (2009) eq. 3, raw scale:

        Y_t^(h) = (1/h) * Sum_{k=1}^{h} RV_{t+k}

    For h = 1 this collapses to RV_t itself.

    Window convention: rolling(h).mean().shift(-(h-1)) spans t ... t+h-1.
    Every regressor is shifted by one day, so the information set ends at
    t-1 while the target covers the next h days -- a genuine forecast with
    no look-ahead, indexed one day earlier than the formula above.

    Using the h-day SUM instead would multiply Y, all coefficients and all
    residuals by h (MSE by h^2). R^2 is invariant, but reported losses are
    not -- hence the average, which is also Corsi's convention.
    """
    df = df_base.copy()
    if h == 1:
        df["Y_h"] = df["RV"]
    else:
        df["Y_h"] = df["RV"].rolling(h).mean().shift(-(h - 1))
    return df.dropna(subset=["Y_h", "RV_d", "RV_w", "RV_m"])


# ==============================================================================
# 3.  YEAR-BASED TRAIN / TEST SPLIT   (mirrors Dataset_Custom)
# ==============================================================================

def split_by_year(df: pd.DataFrame,
                  train_end_year: int  = TRAIN_END_YEAR,
                  test_start_year: int = TEST_START_YEAR):
    """Chronological year-based split identical to Dataset_Custom."""
    train = df[df.index.year <= train_end_year].copy()
    test  = df[df.index.year >= test_start_year].copy()

    if len(train) == 0:
        raise ValueError(
            f"Train set is empty. Check that your data contains years <= {train_end_year}.")
    if len(test) == 0:
        raise ValueError(
            f"Test set is empty. Check that your data contains years >= {test_start_year}.")
    return train, test


# ==============================================================================
# 4.  HAR-RV OLS ESTIMATION  (per horizon)
# ==============================================================================

def fit_har_rv(train: pd.DataFrame, h: int):
    """
    Estimate HAR-RV by OLS in LEVELS for forecast horizon h.

        Y_t^(h) = b0 + b_d*RV_d + b_w*RV_w + b_m*RV_m + e_t

    HAC bandwidth L = 2*(h-1), Newey-West (1987) Bartlett kernel. The HAC
    estimator is doing more work here than in the log version: raw-RV
    residuals are strongly heteroskedastic, and Newey-West corrects for
    that as well as for the overlap-induced autocorrelation.
    """
    nw_lag = HORIZONS[h]["nw_lag"]
    Y = train["Y_h"]
    X = sm.add_constant(train[["RV_d", "RV_w", "RV_m"]])
    return OLS(Y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": nw_lag, "use_correction": True},
    )


# ==============================================================================
# 5.  LOSS FUNCTIONS   (all on the RAW RV scale)
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

def compute_metrics(actual, predicted, floor: float) -> dict:
    q, n_bad = qlike(actual, predicted, floor)
    return {"MSE"       : mse(actual, predicted),
            "MAE"       : mae(actual, predicted),
            "QLIKE"     : q,
            "n_neg_pred": n_bad,
            "pct_neg"   : 100.0 * n_bad / len(predicted)}


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
    print(f"\n  -- Split (follows Dataset_Custom) --------------------------")
    print(f"  {'Set':<8} {'Rows':>6}  {'Start':>12}  {'End':>12}  {'Years'}")
    print(f"  {THIN[:60]}")
    print(f"  {'Train':<8} {len(train):>6}  "
          f"{str(train.index[0].date()):>12}  {str(train.index[-1].date()):>12}  "
          f"{SAMPLE_START_YEAR} - {TRAIN_END_YEAR}")
    print(f"  {'Test':<8} {len(test):>6}  "
          f"{str(test.index[0].date()):>12}  {str(test.index[-1].date()):>12}  "
          f"{TEST_START_YEAR} - {SAMPLE_END_YEAR}")
    print(f"  {THIN[:60]}")
    print(f"  Note: OLS has no hyperparameters, so the deep models' validation")
    print(f"        years are folded into training. Test window is identical")
    print(f"        to the DL models' for a like-for-like comparison.\n")

def print_estimation_table(result, h: int):
    nw_lag   = HORIZONS[h]["nw_lag"]
    hlabel   = HORIZONS[h]["label"]
    params, bse   = result.params, result.bse
    tvals, pvals  = result.tvalues, result.pvalues
    ci       = result.conf_int(alpha=0.05)
    name_map = {"const": "Intercept (b0)",
                "RV_d" : "Daily RV   (b_d)",
                "RV_w" : "Weekly RV  (b_w)",
                "RV_m" : "Monthly RV (b_m)"}

    hdr_nw = f"L={nw_lag} -- Corsi (2009) / Patton & Sheppard (2009)"
    print_section(f"HAR-RV [RAW SCALE]  [{hlabel}]  Newey-West HAC  {hdr_nw}")
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
    print("Coefficients are unit-free (RV regressed on RV), so b_d+b_w+b_m")
    print("near 1 indicates a near-unit-root / long-memory persistence.")
    print(f"\n  R2           : {result.rsquared:.4f}")
    print(f"  Adj. R2      : {result.rsquared_adj:.4f}")
    print(f"  F-statistic  : {result.fvalue:.3f}   [p = {result.f_pvalue:.4f}]")
    print(f"  Observations : {int(result.nobs):,}")
    print(f"  AIC          : {result.aic:.3f}   BIC : {result.bic:.3f}")

def print_metrics_by_horizon(all_metrics: dict):
    print_section(f"OUT-OF-SAMPLE FORECAST EVALUATION -- ALL HORIZONS  "
                  f"(Test {TEST_START_YEAR}-{SAMPLE_END_YEAR})")
    print(f"  {'Metric':<10}", end="")
    for h in HORIZONS:
        print(f"  {HORIZONS[h]['label']:>20}", end="")
    print()
    print(THIN)
    for metric in ["MSE", "MAE", "QLIKE"]:
        print(f"  {metric:<10}", end="")
        for h in HORIZONS:
            print(f"  {all_metrics[h]['test'][metric]:>20.6f}", end="")
        print()
    print(f"  {'neg pred':<10}", end="")
    for h in HORIZONS:
        m = all_metrics[h]["test"]
        cell = f"{m['n_neg_pred']} ({m['pct_neg']:.1f}%)"
        print(f"  {cell:>20}", end="")
    print()
    print(THIN)
    print("  Note: All metrics on the RAW RV scale -- NOT comparable to the")
    print("        ln(RV)-scale losses from the log version of this script.")
    print("        MSE in levels is dominated by a few spike days; MAE and")
    print("        QLIKE (Patton, 2011) are the more robust rankings here.")
    print("        'neg pred' counts non-positive forecasts, floored before")
    print("        QLIKE. A large count means QLIKE leans on that floor.")

def print_diagnostics(diag: dict, h: int):
    print_section(f"RESIDUAL DIAGNOSTICS  [h={h}, {HORIZONS[h]['label']}, "
                  f"Training Sample {SAMPLE_START_YEAR}-{TRAIN_END_YEAR}]")
    print(f"  {'Statistic':<30} {'Value':>12}")
    print(THIN)
    for k, v in diag.items():
        print(f"  {k:<30} {v:>12.4f}")
    print(THIN)
    print("  BP = Breusch-Pagan. A large statistic is EXPECTED in levels:")
    print("  RV residual variance scales with the level of RV. HAC standard")
    print("  errors remain valid; plain OLS SEs would be understated.")


# ==============================================================================
# 8.  FIGURES
# ==============================================================================

def _ylab(h):
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
        # Zero line matters in levels: forecasts can cross it.
        ax.axhline(0, color="black", lw=0.6, alpha=0.6)
        ax.set_ylabel(_ylab(h), fontsize=10)
        ax.set_title(f"{panel_letters[idx]}  {HORIZONS[h]['label']} -- "
                     f"Out-of-Sample Forecast ({TEST_START_YEAR}-{SAMPLE_END_YEAR})",
                     loc="left", pad=4)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.legend(loc="upper right", fontsize=8)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.suptitle(f"HAR-RV (raw RV) Multi-Horizon Forecasts: h = 1, 5, 22  (Corsi, 2009)\n"
                 f"Train: {SAMPLE_START_YEAR}-{TRAIN_END_YEAR}  |  "
                 f"Test: {TEST_START_YEAR}-{SAMPLE_END_YEAR}",
                 fontsize=12, fontweight="bold", y=1.01)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT_DIR}/har_rv_fig1_multihoriz_forecast.{ext}", bbox_inches="tight")
    print("  -> Saved: har_rv_fig1_multihoriz_forecast.pdf / .png")
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
        ax.set_title(f"{panel_letters[idx]}  {metric}", loc="left", pad=3)
        ax.set_ylabel(f"{metric}  (raw RV scale)")
        ax.set_xlabel("Forecast Horizon")
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    fig.suptitle(f"HAR-RV (raw RV): Out-of-Sample Loss by Horizon -- "
                 f"Test {TEST_START_YEAR}-{SAMPLE_END_YEAR}",
                 fontsize=12, fontweight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT_DIR}/har_rv_fig2_loss_comparison.{ext}", bbox_inches="tight")
    print("  -> Saved: har_rv_fig2_loss_comparison.pdf / .png")
    plt.close(fig)


def figure_coefficients_across_horizons(results_dict: dict):
    horizons   = list(HORIZONS.keys())
    coef_names = {"const": "b0  (Intercept)", "RV_d": "b_d (Daily)",
                  "RV_w" : "b_w (Weekly)",    "RV_m": "b_m (Monthly)"}
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

    fig.suptitle("HAR-RV (raw RV): Coefficient Paths Across Horizons  (h = 1, 5, 22)",
                 fontsize=12, fontweight="bold", y=1.01)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT_DIR}/har_rv_fig3_coeff_paths.{ext}", bbox_inches="tight")
    print("  -> Saved: har_rv_fig3_coeff_paths.pdf / .png")
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

    fig.suptitle(f"HAR-RV (raw RV): ACF of Squared Residuals -- Training Sample "
                 f"{SAMPLE_START_YEAR}-{TRAIN_END_YEAR}  (vertical line = NW bandwidth)",
                 fontsize=11, fontweight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT_DIR}/har_rv_fig4_acf_residuals.{ext}", bbox_inches="tight")
    print("  -> Saved: har_rv_fig4_acf_residuals.pdf / .png")
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
        # In levels the negative-forecast region is meaningful; shade it.
        if lo < 0:
            ax.axvspan(lo, 0, color="#d73027", alpha=0.08)
            ax.axvline(0, color="#d73027", lw=0.8, ls=":", label="RV_hat = 0")
        ax.set_xlabel(f"Predicted  {_ylab(h)}", fontsize=9)
        ax.set_ylabel(f"Actual     {_ylab(h)}", fontsize=9)
        ax.set_title(f"{panel_letters[idx]}  h={h}  [{HORIZONS[h]['label']}]",
                     loc="left", pad=3)
        ax.legend(fontsize=8)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    fig.suptitle(f"HAR-RV (raw RV): Actual vs. Predicted -- Test "
                 f"{TEST_START_YEAR}-{SAMPLE_END_YEAR}  (all horizons)",
                 fontsize=12, fontweight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT_DIR}/har_rv_fig5_scatter_grid.{ext}", bbox_inches="tight")
    print("  -> Saved: har_rv_fig5_scatter_grid.pdf / .png")
    plt.close(fig)


# ==============================================================================
# 9.  EXPORT
# ==============================================================================

def export_all(results_dict: dict, all_metrics: dict):
    # -- 9a. Fitted values (one CSV per horizon) ------------------------------
    for h, res in results_dict.items():
        out = res["df_train"][["Y_h"]].copy()
        out["fitted"]   = res["y_hat_train"].values
        out["residual"] = out["Y_h"] - out["fitted"]
        out["split"]    = "train"
        test_out = res["df_test"][["Y_h"]].copy()
        test_out["fitted"]   = res["y_hat_test"].values
        test_out["residual"] = test_out["Y_h"] - test_out["fitted"]
        test_out["split"]    = "test"
        full  = pd.concat([out, test_out]).sort_index()
        fname = f"{OUT_DIR}/har_rv_h{h:02d}_fitted.csv"
        full.to_csv(fname)
        print(f"  -> Saved: har_rv_h{h:02d}_fitted.csv")

    # -- 9b. Consolidated metrics table ---------------------------------------
    rows = []
    for h in results_dict:
        for split in ["train", "test"]:
            m = all_metrics[h][split]
            rows.append({"horizon": h, "split": split, "scale": "raw_RV",
                         "MSE": m["MSE"], "MAE": m["MAE"], "QLIKE": m["QLIKE"],
                         "n_neg_pred": m["n_neg_pred"], "pct_neg": m["pct_neg"]})
    pd.DataFrame(rows).to_csv(f"{OUT_DIR}/har_rv_all_metrics.csv", index=False)
    print("  -> Saved: har_rv_all_metrics.csv")

    # -- 9c. Parameter tables (one CSV per horizon) ---------------------------
    for h, res in results_dict.items():
        result = res["result"]
        ci     = result.conf_int()
        pd.DataFrame({"Coefficient": result.params, "Std_Error": result.bse,
                      "t_stat": result.tvalues, "p_value": result.pvalues,
                      "CI_lower": ci[0], "CI_upper": ci[1]}
                     ).to_csv(f"{OUT_DIR}/har_rv_h{h:02d}_params.csv")
        print(f"  -> Saved: har_rv_h{h:02d}_params.csv")


# ==============================================================================
# 10.  MAIN PIPELINE
# ==============================================================================

def main(argv=None):
    # -- 10.0  Resolve command-line configuration -----------------------------
    args = parse_args(argv)

    global DATA_FILE, OUT_DIR, TRAIN_END_YEAR, TEST_START_YEAR, QLIKE_FLOOR_FRAC
    DATA_FILE        = resolve_data_path(args.data)
    OUT_DIR          = args.outdir
    TRAIN_END_YEAR   = args.train_end_year
    TEST_START_YEAR  = args.test_start_year
    QLIKE_FLOOR_FRAC = args.qlike_floor_frac
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\n{SEP}")
    print("  HAR-RV MULTI-HORIZON MODEL  --  Corsi (2009)  --  RAW RV SCALE")
    print("  EUR/USD Realized Variance  |  Horizons: h = 1, 5, 22")
    print("  Aggregation: arithmetic averages of RV on BOTH sides -- no logs")
    print(SEP)

    # -- 10.1  Load base features ---------------------------------------------
    print(f"\n[1] Loading data from '{DATA_FILE}' ...")
    if os.path.abspath(DATA_FILE) != os.path.abspath(args.data):
        print(f"    (resolved from requested '{args.data}')")
    df_base = load_base_features(DATA_FILE)
    print(f"  Full series  : {df_base.index[0].date()} -> {df_base.index[-1].date()}")
    print(f"  Total rows   : {len(df_base):,}  (before per-horizon NaN drop)")
    print(f"  RV summary   : mean={df_base['RV'].mean():.4f}  "
          f"sd={df_base['RV'].std():.4f}  min={df_base['RV'].min():.4f}  "
          f"max={df_base['RV'].max():.4f}")
    print(f"  RV skew      : {df_base['RV'].skew():.2f}   "
          f"excess kurt: {df_base['RV'].kurt():.2f}")
    print(f"  Split: Train {SAMPLE_START_YEAR}-{TRAIN_END_YEAR}  |  "
          f"Test {TEST_START_YEAR}-{SAMPLE_END_YEAR}")

    floor = QLIKE_FLOOR_FRAC * float(
        df_base.loc[df_base.index.year <= TRAIN_END_YEAR, "RV"].mean())
    print(f"  QLIKE floor for non-positive forecasts: {floor:.3e}")

    # -- 10.2  Per-horizon loop ------------------------------------------------
    results_dict, all_metrics = {}, {}

    for h in HORIZONS:
        print(f"\n{'-'*72}")
        print(f"  HORIZON  h = {h}  [{HORIZONS[h]['label']}]")
        print(f"{'-'*72}")

        df_h        = build_horizon_target(df_base, h)
        train, test = split_by_year(df_h, TRAIN_END_YEAR, TEST_START_YEAR)
        print_split_info(train, test, h)

        result = fit_har_rv(train, h)
        print_estimation_table(result, h)

        X_tr = sm.add_constant(train[["RV_d", "RV_w", "RV_m"]])
        X_te = sm.add_constant(test[["RV_d",  "RV_w", "RV_m"]])
        y_hat_train = result.predict(X_tr)
        y_hat_test  = result.predict(X_te)

        m_train = compute_metrics(train["Y_h"].values, y_hat_train.values, floor)
        m_test  = compute_metrics(test["Y_h"].values,  y_hat_test.values,  floor)

        if m_test["n_neg_pred"]:
            print(f"\n  ! {m_test['n_neg_pred']} non-positive test forecast(s) "
                  f"({m_test['pct_neg']:.1f}%) -- floored for QLIKE. "
                  f"Expected in levels; a log-scale HAR cannot produce these.")

        print_diagnostics(residual_diagnostics(result), h)

        results_dict[h] = {"result": result, "df_train": train, "df_test": test,
                           "y_hat_train": y_hat_train, "y_hat_test": y_hat_test}
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
          f"({TEST_START_YEAR}-{SAMPLE_END_YEAR})  [RAW RV SCALE]")
    print(THIN)
    print(f"  {'Horizon':<14} {'NW Lag':>8} {'MSE':>12} {'MAE':>12} "
          f"{'QLIKE':>12} {'neg pred':>10}")
    print(THIN)
    for h in HORIZONS:
        m = all_metrics[h]["test"]
        print(f"  {HORIZONS[h]['label']:<14} {HORIZONS[h]['nw_lag']:>8}  "
              f"{m['MSE']:>12.6f} {m['MAE']:>12.6f} {m['QLIKE']:>12.6f} "
              f"{m['n_neg_pred']:>10}")
    print(THIN)
    print(f"""
  Aggregation (raw scale, no logs anywhere):
    Target     Y^(h) = (1/h) * Sum_k RV_(t+k)   over the h-day forward window
    Regressors RV_d  = RV_(t-1)
               RV_w  = mean RV over the previous 5 days
               RV_m  = mean RV over the previous 22 days

  Reading the output:
    * Losses are on the RV scale and are NOT comparable to the ln(RV)-scale
      losses from the log version. Re-evaluate any DL baseline on this scale.
    * MSE is spike-dominated in levels; prefer MAE and QLIKE for ranking.
    * Non-positive forecasts are possible and are reported per horizon.
    * Breusch-Pagan will reject strongly -- expected, not a bug. HAC SEs
      remain valid; OLS is simply no longer efficient.

  Split:
    Train  {SAMPLE_START_YEAR} - {TRAIN_END_YEAR}  (year <= {TRAIN_END_YEAR})
    Test   {TEST_START_YEAR} - {SAMPLE_END_YEAR}   (year >= {TEST_START_YEAR})
    Dataset_Custom (data_loader.py) splits train / val / test at the same
    year boundaries; HAR-RV has no hyperparameters, so its validation years
    are folded into the estimation sample. The test window is shared.

  References:
    Corsi, F. (2009). A simple approximate long-memory model of realized
      volatility. Journal of Financial Econometrics, 7(2), 174-196.
    Patton, A. J. (2011). Volatility forecast comparison using imperfect
      volatility proxies. Journal of Econometrics, 160(1), 246-256.
    Patton, A. J., & Sheppard, K. (2009). Evaluating volatility and
      correlation forecasts. Handbook of Financial Time Series. Springer.
    Bollerslev, T., Patton, A. J., & Quaedvlieg, R. (2016). Exploiting
      the errors. Journal of Econometrics, 192(1), 1-18.
""")
    print(SEP)


if __name__ == "__main__":
    main()
