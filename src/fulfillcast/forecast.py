"""Posterior predictive forecasts of future bookings and fulfillment.

Forecast = backward, then forward:

1. backward: MCMC gave posterior draws of all parameters (``FitResult``);
2. forward: for every draw, simulate the latent deviation u and the weekly
   bookings for the coming weeks.

The spread of the result contains both parameter uncertainty (step 1) and
booking noise (step 2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import PARAMS, FitResult


@dataclass
class Forecast:
    """Simulated future bookings for every customer and week."""

    customer_ids: np.ndarray
    customer_types: np.ndarray
    dates: pd.DatetimeIndex
    samples: np.ndarray  # (S, C, H) simulated bookings
    commitment: np.ndarray  # (C, H)
    lam: np.ndarray  # (S, C, H) simulated latent rate

    def weekly_summary(self, interval: float = 0.8) -> pd.DataFrame:
        """Median and interval of weekly bookings, long format."""
        lo, hi = (1 - interval) / 2, 1 - (1 - interval) / 2
        q = np.quantile(self.samples, [lo, 0.5, hi], axis=0)  # (3, C, H)
        C, H = self.commitment.shape
        return pd.DataFrame(
            {
                "customer_id": np.repeat(self.customer_ids, H),
                "week_start": np.tile(self.dates, C),
                "commitment": self.commitment.ravel(),
                "bookings_mean": self.samples.mean(axis=0).ravel(),
                "bookings_lower": q[0].ravel(),
                "bookings_median": q[1].ravel(),
                "bookings_upper": q[2].ravel(),
            }
        )

    def fulfillment_samples(self) -> np.ndarray:
        """Fulfillment over the whole horizon per draw: (S, C)."""
        return self.samples.sum(axis=2) / self.commitment.sum(axis=1)[None, :]

    def fulfillment_summary(self, interval: float = 0.8, low_threshold: float = 0.6) -> pd.DataFrame:
        """Per-customer forecast of fulfillment over the horizon."""
        f = self.fulfillment_samples()
        lo, hi = (1 - interval) / 2, 1 - (1 - interval) / 2
        q = np.quantile(f, [lo, 0.5, hi], axis=0)
        return pd.DataFrame(
            {
                "customer_id": self.customer_ids,
                "customer_type": self.customer_types,
                "commitment_total": self.commitment.sum(axis=1),
                "expected_bookings": self.samples.sum(axis=2).mean(axis=0),
                "fulfillment_median": q[1],
                "fulfillment_lower": q[0],
                "fulfillment_upper": q[2],
                f"p_below_{low_threshold:g}": (f < low_threshold).mean(axis=0),
            }
        )


def _thin(n_total: int, n_samples: int | None, rng: np.random.Generator) -> np.ndarray:
    if n_samples is None or n_samples >= n_total:
        return np.arange(n_total)
    return np.sort(rng.choice(n_total, size=n_samples, replace=False))


def simulate_spread_paths(history: np.ndarray, horizon: int, n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate future spread paths from per-customer AR(1) fits: (n_paths, C, horizon).

    Each customer's standardized spread follows x[t] = c + a * x[t-1] + e[t],
    fitted by least squares on the training window. Shocks e are correlated
    across customers (customers on the same lane share the market), using the
    residual covariance shrunk toward its diagonal.
    """
    x0, x1 = history[:, :-1], history[:, 1:]
    xm0, xm1 = x0.mean(axis=1, keepdims=True), x1.mean(axis=1, keepdims=True)
    a = ((x0 - xm0) * (x1 - xm1)).sum(axis=1) / ((x0 - xm0) ** 2).sum(axis=1)
    a = np.clip(a, 0.0, 0.99)
    c = xm1[:, 0] - a * xm0[:, 0]
    resid = x1 - (c[:, None] + a[:, None] * x0)
    cov = np.cov(resid)
    cov = 0.9 * cov + 0.1 * np.diag(np.diag(cov))
    vals, vecs = np.linalg.eigh(cov)
    root = vecs * np.sqrt(np.clip(vals, 0.0, None))
    C = history.shape[0]
    paths = np.empty((n_paths, C, horizon))
    x = np.repeat(history[None, :, -1], n_paths, axis=0)
    for h in range(horizon):
        x = c + a * x + rng.standard_normal((n_paths, C)) @ root.T
        paths[:, :, h] = x
    return paths


def forecast(
    result: FitResult,
    future: pd.DataFrame,
    spread_scenario: str = "ar1",
    n_samples: int | None = 1000,
    seed: int = 0,
) -> Forecast:
    """Simulate bookings for the weeks in ``future``.

    Parameters
    ----------
    result
        Output of :func:`fulfillcast.model.fit`.
    future
        Rows for the forecast weeks with at least ``customer_id``, ``week_start``
        and ``commitment``; ``spread`` is needed when ``spread_scenario="observed"``.
    spread_scenario
        How the unknown future spread enters the forecast:

        * ``"ar1"`` (default): simulate spread paths from an AR(1) fitted on the
          training window, so spread uncertainty is part of the forecast;
        * ``"persistence"``: keep each customer's last training spread;
        * ``"observed"``: use ``future["spread"]``. Use this for *monitoring*
          (the market is known after the fact) or for explicit scenarios.
          Using realized future spreads to judge forecast accuracy is
          train-serve skew: you would not have them at forecast time.
    n_samples
        Number of posterior draws to use (``None`` = all).
    """
    data = result.data
    rng = np.random.default_rng(seed)
    ids = data.customer_ids
    fut = future[future["customer_id"].isin(ids)]
    dates = pd.DatetimeIndex(sorted(fut["week_start"].unique()))
    if len(dates) == 0:
        raise ValueError("future contains no weeks for the fitted customers")
    if dates[0] <= data.dates[-1]:
        raise ValueError("future weeks must come after the training window")

    commit = fut.pivot(index="customer_id", columns="week_start", values="commitment").reindex(index=ids, columns=dates)
    if commit.isna().any().any():
        raise ValueError("future must have a commitment for every customer and week")
    commit = commit.to_numpy(dtype=float)
    C, H = commit.shape

    if spread_scenario not in ("ar1", "persistence", "observed"):
        raise ValueError("spread_scenario must be 'ar1', 'persistence' or 'observed'")

    t_future = data.transform.t_years(dates)
    F_future = data.transform.fourier(dates)

    theta = result.draws("theta")  # (S, C, P)
    idx = _thin(theta.shape[0], n_samples, rng)
    theta = theta[idx]
    alpha = theta[..., PARAMS.index("alpha")]
    b = theta[..., PARAMS.index("b")]
    beta = theta[..., PARAMS.index("beta")]
    tau = result.draws("tau")[idx]  # (S, C)
    rho = result.draws("rho")[idx][:, data.type_idx]  # (S, C)
    u = result.draws("u_last")[idx]  # (S, C)
    phi = result.draws("phi")[idx]  # (S,)
    if F_future.shape[1] > 0:
        delta = result.draws("delta")[idx]  # (S, type, K)
        season = np.einsum("hk,stk->sth", F_future, delta)[:, data.type_idx, :]  # (S, C, H)
    else:
        season = 0.0

    S = len(idx)
    if spread_scenario == "ar1":
        spread = simulate_spread_paths(data.spread, H, S, rng)  # (S, C, H)
    elif spread_scenario == "persistence":
        spread = np.repeat(data.spread[None, :, -1:], H, axis=2)  # (1, C, H)
    else:
        raw = fut.pivot(index="customer_id", columns="week_start", values="spread").reindex(index=ids, columns=dates)
        if raw.isna().any().any():
            raise ValueError("future must have a spread for every customer and week when spread_scenario='observed'")
        spread = data.transform.spread(raw.to_numpy(dtype=float))[None]  # (1, C, H)

    u_path = np.empty((S, C, H))
    for h in range(H):
        u = rho * u + tau * rng.standard_normal((S, C))
        u_path[:, :, h] = u

    log_lam = alpha[..., None] + b[..., None] * t_future[None, None, :] + beta[..., None] * spread + season + u_path
    lam = np.exp(log_lam)
    mu = commit[None] * lam
    p = phi[:, None, None] / (phi[:, None, None] + mu)
    y = rng.negative_binomial(phi[:, None, None], p)

    types = np.asarray(data.type_names)[data.type_idx]
    return Forecast(customer_ids=ids, customer_types=types, dates=dates, samples=y, commitment=commit, lam=lam)
