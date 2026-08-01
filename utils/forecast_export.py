"""
Per-observation forecast export -- the input layer for Diebold-Mariano and
Model Confidence Set testing.

Aggregate losses (a single MSE, a single QLIKE) cannot support either test.
Both need the loss SERIES: DM tests the mean of the loss differential
d_t = L(model A)_t - L(model B)_t with a HAC variance, and MCS bootstraps over
the same per-observation losses. So every run has to leave behind one row per
forecast, keyed by date, on a scale shared with every other run.

This module defines that row format once and both producers write it:

  * the deep models, via Exp_Long_Term_Forecast.test()
  * HAR-RV_RUN.PY, via aggregate_forecasts.py reading its *_fitted.csv

Date convention
---------------
`date` is the first day of the target window -- the day the forecast is FOR,
with the information set ending the day before. Dataset_Custom.forecast_dates()
and HAR-RV_RUN.PY's build_horizon_target() both use it, so the two producers
join on `date` with no offset.

Scale convention
----------------
Two pairs of columns are always written:

  y_true_model / y_pred_model   the scale the model was FITTED on -- ln(RV)
                               for a --log run, raw RV otherwise. Losses here
                               are comparable only WITHIN a scale.
  y_true_rv    / y_pred_rv      raw variance, always. A --log forecast is
                               mapped back with the lognormal (Jensen)
                               correction. This is the scale on which a raw
                               run, a --log run and HAR-RV are all comparable,
                               and it is where QLIKE is defined.

Keeping both means the choice of comparison scale is made at analysis time
rather than being baked in at training time.
"""

import os

import numpy as np
import pandas as pd

from utils.metrics import lognormal_back_transform

# One row per forecast. Fixed order so every producer writes the same file.
FORECAST_COLUMNS = [
    "date", "model", "scale", "horizon", "seed", "split",
    "y_true_rv", "y_pred_rv",
    "y_true_model", "y_pred_model",
    "bias", "resid_var",
]

# Losses the analysis layer can build from those columns.
#   se_rv / qlike : raw-variance scale, comparable across raw and --log runs.
#                   Both are ROBUST in the sense of Patton (2011) -- they rank
#                   forecasts consistently even though RV is a noisy proxy for
#                   the latent variance.
#   se_model      : the fitted scale, within-scale comparisons only.
# ae_rv / ae_model are written for reporting but are deliberately NOT offered
# as DM/MCS losses: absolute error is not robust to proxy noise and can rank a
# worse forecast above a better one.
RV_LOSSES = ("se_rv", "qlike")
MODEL_SCALE_LOSSES = ("se_model",)


def run_tag(model, scale, horizon, seed):
    """Canonical stem for one run's artifacts. Sortable and collision-free."""
    return f"{model}_{scale}_h{int(horizon):02d}_s{int(seed)}"


def build_forecast_frame(dates, pred_model, true_model, *, model, scale,
                         horizon, seed, split="test", bias=0.0, resid_var=0.0):
    """
    Assemble the tidy per-forecast table for one run.

    `pred_model` / `true_model` are on the modelling scale, flat and aligned
    1:1 with `dates`.

    For scale == "log" the raw-variance columns come from the lognormal
    back-transform exp(mu + bias + sigma^2/2). Both correction terms MUST be
    estimated on TRAINING residuals -- taking them from the test residuals
    would feed the realised outcome back into the forecast and flatter the
    model. `bias` is carried because a network, unlike an OLS fit with an
    intercept, has no guarantee of zero mean residual.
    """
    pred_model = np.asarray(pred_model, dtype=float).reshape(-1)
    true_model = np.asarray(true_model, dtype=float).reshape(-1)
    dates = pd.to_datetime(pd.Series(np.asarray(dates)).reset_index(drop=True))

    if not (len(dates) == len(pred_model) == len(true_model)):
        raise ValueError(
            f"length mismatch: {len(dates)} dates, {len(pred_model)} preds, "
            f"{len(true_model)} trues. The forecast dates must line up 1:1 "
            f"with the forecasts or the DM/MCS join will silently misalign.")

    if scale == "log":
        y_true_rv = np.exp(true_model)
        y_pred_rv = lognormal_back_transform(pred_model, resid_var, bias)
    elif scale == "raw":
        y_true_rv = true_model
        y_pred_rv = pred_model
        bias, resid_var = 0.0, 0.0
    else:
        raise ValueError(f"scale must be 'raw' or 'log', got {scale!r}")

    return pd.DataFrame({
        "date": dates,
        "model": model,
        "scale": scale,
        "horizon": int(horizon),
        "seed": int(seed),
        "split": split,
        "y_true_rv": y_true_rv,
        "y_pred_rv": y_pred_rv,
        "y_true_model": true_model,
        "y_pred_model": pred_model,
        "bias": float(bias),
        "resid_var": float(resid_var),
    })[FORECAST_COLUMNS]


def write_forecast_frame(df, out_dir, tag):
    """Write one run's forecasts to <out_dir>/<tag>.csv and return the path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{tag}.csv")
    df.to_csv(path, index=False)
    return path


def add_losses(df, qlike_floor):
    """
    Attach per-observation losses to a forecast table.

    `qlike_floor` must be ONE number shared by every model in a comparison.
    QLIKE is undefined at a non-positive variance forecast, which an
    unconstrained raw-scale head can produce; flooring keeps those rows in the
    sample (dropping them would delete a model's worst forecasts and hand it an
    advantage in the very test meant to catch that). A --log forecast is
    positive by construction, so the floor never binds there -- which is
    exactly why the floor has to be common: a per-model floor would make the
    penalty a model chose for itself.
    """
    out = df.copy()
    true_rv = out["y_true_rv"].to_numpy(dtype=float)
    pred_rv = out["y_pred_rv"].to_numpy(dtype=float)

    out["n_floored"] = (pred_rv <= 0).astype(int)
    safe_rv = np.where(pred_rv <= 0, qlike_floor, pred_rv)

    out["se_rv"] = (true_rv - pred_rv) ** 2
    out["ae_rv"] = np.abs(true_rv - pred_rv)

    # QLIKE = RV/RV_hat - ln(RV/RV_hat) - 1, minimised at RV_hat = RV.
    ratio = np.where(true_rv > 0, true_rv / safe_rv, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["qlike"] = ratio - np.log(ratio) - 1.0

    err_model = out["y_true_model"].to_numpy(dtype=float) - \
        out["y_pred_model"].to_numpy(dtype=float)
    out["se_model"] = err_model ** 2
    out["ae_model"] = np.abs(err_model)
    return out
