"""Drift detection: is a customer behaving differently from what the model expects?

Three complementary signals:

1. **Surprise.** Compare the bookings that actually arrived with the forecast.
   Two rules are available:

   * ``"cumulative"`` (default): the PIT of *cumulative* bookings since the
     forecast origin. It accounts for the week-to-week correlation of the
     latent rate, so false alarms stay close to the nominal level.
   * ``"cusum"``: a CUSUM on weekly PITs. It reacts faster to a break, but it
     assumes independent weeks, so it raises more false alarms.

   For monitoring, build the forecast with ``spread_scenario="observed"``:
   after the fact the market is known, and the question is whether the
   bookings were surprising *given* that market.
2. **Current deviation.** The latent deviation u at the end of training,
   relative to its usual size s. A customer sitting far below its trend is a
   warning even before new data arrive.
3. **Parameter shift between two fits.** Has the estimated trend, price
   sensitivity or volatility moved since the last refit?

Principle: monitor what the model believes, not only how wrong it was.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .forecast import Forecast
from .model import PARAMS, FitResult


def predictive_pit(fc: Forecast, observed: np.ndarray) -> np.ndarray:
    """Randomization-free (mid) PIT of observed counts under the forecast.

    PIT = P(Y < y) + 0.5 * P(Y = y). Under a correct model it is roughly
    uniform; values near 0 mean "far fewer bookings than expected".
    Shape (C, H).
    """
    s = fc.samples  # (S, C, H)
    obs = observed[None]
    pit = (s < obs).mean(axis=0) + 0.5 * (s == obs).mean(axis=0)
    eps = 0.5 / s.shape[0]
    return np.clip(pit, eps, 1 - eps)


def cusum_alarms(pit: np.ndarray, k: float = 0.5, h: float = 4.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-sided CUSUM on z = Phi^-1(PIT).

    Returns ``(alarm_week_down, alarm_week_up, z)`` where alarm weeks are the
    index of the first alarm per customer (-1 if none).
    """
    z = norm.ppf(pit)
    C, H = z.shape
    down = np.zeros(C)
    up = np.zeros(C)
    alarm_down = np.full(C, -1)
    alarm_up = np.full(C, -1)
    for t in range(H):
        down = np.maximum(0.0, down - z[:, t] - k)
        up = np.maximum(0.0, up + z[:, t] - k)
        alarm_down = np.where((alarm_down < 0) & (down > h), t, alarm_down)
        alarm_up = np.where((alarm_up < 0) & (up > h), t, alarm_up)
    return alarm_down, alarm_up, z


def _mid_pit(samples: np.ndarray, observed: np.ndarray) -> np.ndarray:
    pit = (samples < observed[None]).mean(axis=0) + 0.5 * (samples == observed[None]).mean(axis=0)
    eps = 0.5 / samples.shape[0]
    return np.clip(pit, eps, 1 - eps)


def cumulative_pit(fc: Forecast, observed: np.ndarray) -> np.ndarray:
    """PIT of cumulative bookings from the forecast origin up to each week: (C, H)."""
    return _mid_pit(np.cumsum(fc.samples, axis=2), np.cumsum(observed, axis=1))


def _first(mask: np.ndarray) -> np.ndarray:
    """Index of the first True per row, -1 if none."""
    return np.where(mask.any(axis=1), mask.argmax(axis=1), -1)


def surprise_report(
    fc: Forecast,
    actuals: pd.DataFrame,
    method: str = "cumulative",
    alpha: float = 0.01,
    k: float = 0.5,
    h: float = 5.0,
) -> pd.DataFrame:
    """Per-customer drift report for the weeks in ``actuals`` (a bookings table).

    Parameters
    ----------
    method
        ``"cumulative"`` (alarm when the cumulative PIT leaves [alpha, 1 - alpha])
        or ``"cusum"`` (alarm when the CUSUM statistic exceeds ``h``).
    """
    obs = (
        actuals.pivot(index="customer_id", columns="week_start", values="bookings")
        .reindex(index=fc.customer_ids, columns=fc.dates)
        .to_numpy(dtype=float)
    )
    if np.isnan(obs).any():
        raise ValueError("actuals must cover every forecast customer and week")
    weekly = predictive_pit(fc, obs)
    cum = cumulative_pit(fc, obs)
    if method == "cumulative":
        a_down, a_up = _first(cum < alpha), _first(cum > 1 - alpha)
    elif method == "cusum":
        a_down, a_up, _ = cusum_alarms(weekly, k=k, h=h)
    else:
        raise ValueError("method must be 'cumulative' or 'cusum'")

    # If both directions fire, report the earlier one.
    down_first = (a_down >= 0) & ((a_up < 0) | (a_down <= a_up))
    up_first = (a_up >= 0) & ~down_first
    dates = np.asarray(fc.dates)
    week = np.where(down_first, a_down, np.where(up_first, a_up, -1))
    return pd.DataFrame(
        {
            "customer_id": fc.customer_ids,
            "customer_type": fc.customer_types,
            "cumulative_pit_last": cum[:, -1],
            "min_cumulative_pit": cum.min(axis=1),
            "mean_weekly_z": norm.ppf(weekly).mean(axis=1),
            "alarm": np.where(down_first, "down", np.where(up_first, "up", "none")),
            "alarm_week": [pd.Timestamp(dates[w]) if w >= 0 else pd.NaT for w in week],
        }
    )


def deviation_report(result: FitResult, z_threshold: float = 2.0) -> pd.DataFrame:
    """Where each customer's latent rate sits relative to its trend at the end of training."""
    u = result.draws("u_last")
    s = result.draws("s")
    ratio = u / s
    return pd.DataFrame(
        {
            "customer_id": result.data.customer_ids,
            "u_last_mean": u.mean(axis=0),
            "u_over_s_mean": ratio.mean(axis=0),
            "p_far_below": (ratio < -z_threshold).mean(axis=0),
            "p_far_above": (ratio > z_threshold).mean(axis=0),
        }
    )


def compare_fits(old: FitResult, new: FitResult, z_threshold: float = 2.0) -> pd.DataFrame:
    """Shift in trend, price sensitivity and volatility between two fits.

    ``z = (mean_new - mean_old) / sqrt(sd_old**2 + sd_new**2)`` per parameter.
    The level (alpha) is not compared, because it is defined relative to each
    fit's own time window.
    """
    common = np.intersect1d(old.data.customer_ids, new.data.customer_ids)
    rows = {"customer_id": common}
    flags = np.zeros(len(common), dtype=bool)
    for name in ("b", "beta", "log_s"):
        i = PARAMS.index(name)
        vals = []
        for fr in (old, new):
            th = fr.draws("theta")[..., i]
            pos = pd.Index(fr.data.customer_ids).get_indexer(common)
            vals.append((th[:, pos].mean(axis=0), th[:, pos].std(axis=0)))
        (m0, s0), (m1, s1) = vals
        z = (m1 - m0) / np.sqrt(s0**2 + s1**2)
        rows[f"{name}_old"] = m0
        rows[f"{name}_new"] = m1
        rows[f"{name}_z"] = z
        flags |= np.abs(z) > z_threshold
    out = pd.DataFrame(rows)
    out["shift_flag"] = flags
    return out
