"""Validation helpers: parameter recovery, interval coverage, CRPS, baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import to_wide
from .features import years_since
from .forecast import Forecast
from .model import PARAMS, FitResult


def crps_from_samples(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample-based CRPS, E|X - y| - 0.5 E|X - X'|, along axis 0."""
    term1 = np.abs(samples - y[None]).mean(axis=0)
    x = np.sort(samples, axis=0)
    n = x.shape[0]
    # E|X - X'| via the sorted-sample identity
    weights = (2 * np.arange(1, n + 1) - n - 1)[:, None]
    term2 = (weights * x).sum(axis=0) * 2 / (n * n)
    return term1 - 0.5 * term2


def true_parameters_on_model_scale(result: FitResult, truth: pd.DataFrame, full_bookings: pd.DataFrame) -> pd.DataFrame:
    """Express the simulator's true parameters in the fitted model's units.

    The simulator standardizes the spread and centers time over the full
    history; the model does it over the training window. This converts
    alpha and beta accordingly and turns tau into the marginal sd s.
    """
    tr = truth.set_index("customer_id").loc[result.data.customer_ids]
    raw_full = to_wide(full_bookings, "spread").to_numpy(dtype=float)
    m_f, sd_f = raw_full.mean(), raw_full.std()
    tf = result.data.transform
    beta = tr["beta"].to_numpy() * tf.spread_sd / sd_f
    full_dates = pd.DatetimeIndex(sorted(full_bookings["week_start"].unique()))
    full_origin = full_dates[0] + (full_dates[-1] - full_dates[0]) / 2
    shift_years = years_since([tf.t_origin], full_origin)[0]
    alpha = (
        tr["alpha"].to_numpy()
        + tr["b"].to_numpy() * shift_years
        + tr["beta"].to_numpy() * (tf.spread_mean - m_f) / sd_f
    )
    s = tr["tau"].to_numpy() / np.sqrt(1 - tr["rho"].to_numpy() ** 2)
    return pd.DataFrame(
        {
            "customer_id": result.data.customer_ids,
            "archetype": tr["archetype"].to_numpy(),
            "alpha": alpha,
            "b": tr["b"].to_numpy(),
            "beta": beta,
            "s": s,
        }
    )


def parameter_recovery(result: FitResult, truth_model_scale: pd.DataFrame, interval: float = 0.8) -> pd.DataFrame:
    """Correlation and interval coverage of true vs estimated customer parameters."""
    lo, hi = (1 - interval) / 2, 1 - (1 - interval) / 2
    theta = result.draws("theta")
    est = {
        "alpha": theta[..., PARAMS.index("alpha")],
        "b": theta[..., PARAMS.index("b")],
        "beta": theta[..., PARAMS.index("beta")],
        "s": result.draws("s"),
    }
    rows = []
    for name, x in est.items():
        true = truth_model_scale[name].to_numpy()
        mean = x.mean(axis=0)
        ql, qh = np.quantile(x, [lo, hi], axis=0)
        rows.append(
            {
                "parameter": name,
                "correlation": np.corrcoef(true, mean)[0, 1],
                "mae": np.abs(true - mean).mean(),
                f"coverage_{int(interval * 100)}": ((true >= ql) & (true <= qh)).mean(),
            }
        )
    return pd.DataFrame(rows)


def forecast_evaluation(fc: Forecast, actuals: pd.DataFrame, history: pd.DataFrame, interval: float = 0.8) -> dict:
    """Coverage and accuracy of the forecast against what actually happened.

    Compares the Bayesian forecast with a naive baseline: each customer's
    realized fulfillment over the last ``H`` training weeks.
    """
    lo, hi = (1 - interval) / 2, 1 - (1 - interval) / 2
    obs = to_wide(actuals, "bookings").reindex(index=fc.customer_ids, columns=fc.dates).to_numpy(dtype=float)
    ql, qh = np.quantile(fc.samples, [lo, hi], axis=0)
    weekly_cov = ((obs >= ql) & (obs <= qh)).mean()

    f_true = obs.sum(axis=1) / fc.commitment.sum(axis=1)
    f_samp = fc.fulfillment_samples()
    fl, fh = np.quantile(f_samp, [lo, hi], axis=0)
    f_med = np.median(f_samp, axis=0)

    H = len(fc.dates)
    hist_y = to_wide(history, "bookings").reindex(index=fc.customer_ids)
    hist_m = to_wide(history, "commitment").reindex(index=fc.customer_ids)
    baseline = hist_y.iloc[:, -H:].sum(axis=1).to_numpy() / hist_m.iloc[:, -H:].sum(axis=1).to_numpy()

    per_customer = pd.DataFrame(
        {
            "customer_id": fc.customer_ids,
            "fulfillment_actual": f_true,
            "fulfillment_median": f_med,
            "fulfillment_lower": fl,
            "fulfillment_upper": fh,
            "baseline": baseline,
            "crps": crps_from_samples(f_samp, f_true),
            "covered": (f_true >= fl) & (f_true <= fh),
        }
    )
    summary = {
        f"weekly_coverage_{int(interval * 100)}": float(weekly_cov),
        f"fulfillment_coverage_{int(interval * 100)}": float(per_customer["covered"].mean()),
        "fulfillment_mae_model": float(np.abs(f_med - f_true).mean()),
        "fulfillment_mae_baseline": float(np.abs(baseline - f_true).mean()),
        "fulfillment_crps_model": float(per_customer["crps"].mean()),
        # a point forecast's CRPS equals its absolute error
        "fulfillment_crps_baseline": float(np.abs(baseline - f_true).mean()),
    }
    return {"summary": summary, "per_customer": per_customer}
