import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


# ==============================================================================
# Realized-variance losses -- kept identical to HAR-RV_RUN.PY so a deep model
# and the HAR-RV baseline can be compared number for number.
# ==============================================================================

def QLIKE(pred, true, floor):
    """
    QLIKE loss (Patton, 2011) on the RAW variance scale. Smaller = better.

        QLIKE = mean( RV/RV_hat - ln(RV/RV_hat) - 1 )

    Both arguments must already be variances; a log-scale forecast has to be
    passed through lognormal_back_transform() first.

    A model with an unconstrained head can predict RV_hat <= 0, where QLIKE is
    undefined. Such forecasts are FLOORED at `floor` rather than discarded --
    dropping them would remove the worst forecasts and bias QLIKE downward --
    and the count is returned so the reader can judge how much the number
    leans on the floor.

    Returns (qlike, n_nonpositive_forecasts).
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    n_bad = int((pred <= 0).sum())
    safe = np.where(pred <= 0, floor, pred)
    valid = true > 0
    ratio = true[valid] / safe[valid]
    return float(np.mean(ratio - np.log(ratio) - 1)), n_bad


def lognormal_back_transform(pred_log, resid_var=0.0, bias=0.0):
    """
    Map a log-scale forecast back to a variance with the lognormal (Jensen)
    correction:

        E[RV | F] = exp( E[ln RV | F] + bias + sigma^2 / 2 )

    exp() alone returns the conditional MEDIAN, which understates the
    conditional mean by exp(sigma^2/2) -- a systematic bias that does not
    average out. `resid_var` and `bias` are the variance and the mean of the
    TRAINING residuals of the log-scale model.

    The `bias` term is what generalises this beyond OLS. HAR-RV_RUN.PY omits
    it because an OLS fit with an intercept has mean residual exactly zero, so
    it contributes nothing there; a neural network has no such guarantee, and
    dropping a non-zero mean residual would leave a systematic level error in
    the back-transformed forecast. With bias = 0 this reduces precisely to the
    HAR-RV correction, so the two models are still treated identically.

    Passing both terms as 0.0 gives the uncorrected (naive) back-transform.

    The result is strictly positive, so a log-scale model cannot produce the
    non-positive variance forecasts a raw-scale one can.
    """
    return np.exp(np.asarray(pred_log, dtype=float) + bias + resid_var / 2.0)
