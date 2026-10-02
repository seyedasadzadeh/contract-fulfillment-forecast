"""Synthetic contract-booking data with known ground truth.

The generator follows the same structure as the model, so we can test whether
the model recovers the truth (a "recovery test"):

    bookings[c,t] ~ NegBinomial(commitment[c] * lambda[c,t], phi)
    log lambda[c,t] = alpha_c + b_c * t_years + beta_c * spread[c,t]
                      + season_type(t) + u[c,t] + break_c(t)
    u[c,t] = rho_type * u[c,t-1] + tau_c * eps

Five behaviour archetypes are planted:

* stable         business as usual
* growing        positive trend b_c
* declining      negative trend b_c
* opportunistic  large price sensitivity beta_c (and more volatility)
* break          stable, then a sudden level drop in year 2 (for drift detection)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import SimulationConfig
from .features import fourier_features, years_since


@dataclass
class SimulatedData:
    """Output of :func:`simulate_bookings`."""

    bookings: pd.DataFrame
    """Long table: one row per customer and week."""
    truth: pd.DataFrame
    """One row per customer with the true parameters and archetype."""
    season: pd.DataFrame
    """True seasonal log-effect per week and customer type."""


def _ar1(rng: np.random.Generator, n: int, rho: float, sd: float) -> np.ndarray:
    x = np.empty(n)
    x[0] = rng.normal(0, sd / np.sqrt(1 - rho**2))
    for t in range(1, n):
        x[t] = rho * x[t - 1] + rng.normal(0, sd)
    return x


def _draw_customer_params(rng: np.random.Generator, ctype: str, archetype: str) -> dict:
    """True behaviour parameters for one customer."""
    fwd = ctype == "FWD"
    p = {
        "alpha": rng.normal(np.log(0.75) if fwd else np.log(0.85), 0.20 if fwd else 0.15),
        "b": rng.normal(0.0, 0.05),
        "tau": float(np.exp(rng.normal(np.log(0.10), 0.25))),
        "beta": rng.normal(0.25, 0.08) if fwd else rng.normal(0.10, 0.05),
    }
    if archetype == "growing":
        p["b"] = rng.normal(0.35, 0.08)
    elif archetype == "declining":
        p["b"] = rng.normal(-0.45, 0.08)
    elif archetype == "opportunistic":
        p["beta"] = rng.normal(0.75, 0.10)
        p["tau"] = float(np.exp(rng.normal(np.log(0.16), 0.20)))
    return p


def _spot_and_contract(rng: np.random.Generator, cfg: SimulationConfig):
    """Lane spot-rate series (n_lanes, T) and lane base rates."""
    T = cfg.n_weeks
    t = np.arange(T)
    # Common market cycle: a slow swing plus persistent shocks.
    market = 0.22 * np.sin(2 * np.pi * t / 60 + 0.6) + _ar1(rng, T, 0.95, 0.04)
    base = rng.uniform(1500, 3500, size=cfg.n_lanes)  # USD per FEU
    spot = np.vstack([base[lane] * np.exp(market + _ar1(rng, T, 0.9, 0.04)) for lane in range(cfg.n_lanes)])
    return spot, base


def simulate_bookings(cfg: SimulationConfig | None = None) -> SimulatedData:
    """Generate the synthetic dataset described in the module docstring."""
    cfg = cfg or SimulationConfig()
    rng = np.random.default_rng(cfg.seed)
    T = cfg.n_weeks

    dates = pd.date_range(cfg.start_date, periods=T, freq="W-SUN")
    if dates[0] != pd.Timestamp(cfg.start_date):
        raise ValueError("start_date must be a Sunday")
    t_years = years_since(dates, dates[0])
    t_years = t_years - t_years.mean()

    # --- customers -------------------------------------------------------
    types = ["BCO"] * cfg.n_bco + ["FWD"] * cfg.n_fwd
    archetypes = cfg.bco_mix.as_list() + cfg.fwd_mix.as_list()
    n_cust = len(types)
    ids = [f"C{i + 1:03d}" for i in range(n_cust)]
    lanes = rng.integers(0, cfg.n_lanes, size=n_cust)

    params = [_draw_customer_params(rng, ct, ar) for ct, ar in zip(types, archetypes, strict=True)]
    truth = pd.DataFrame(params)
    truth.insert(0, "customer_id", ids)
    truth.insert(1, "customer_type", types)
    truth.insert(2, "archetype", archetypes)
    truth.insert(3, "lane", [f"L{lane + 1}" for lane in lanes])

    # Commitments: lognormal for BCOs; forwarders scaled to hit the target share.
    commit = rng.lognormal(np.log(cfg.bco_median_commitment), 0.6, size=n_cust)
    fwd_mask = truth["customer_type"].to_numpy() == "FWD"
    commit[fwd_mask] = rng.lognormal(0.0, 0.4, size=fwd_mask.sum())
    exp_rate = np.exp(truth["alpha"].to_numpy())
    bco_vol = (commit * exp_rate)[~fwd_mask].sum()
    fwd_unit = (commit * exp_rate)[fwd_mask].sum()
    scale = cfg.forwarder_share / (1 - cfg.forwarder_share) * bco_vol / fwd_unit
    commit[fwd_mask] *= scale
    commit = np.maximum(1, np.round(commit)).astype(int)
    truth["commitment"] = commit

    # Breaks.
    lo, hi = cfg.break_window
    is_break = truth["archetype"].to_numpy() == "break"
    break_week = np.where(is_break, rng.integers(lo, hi, size=n_cust), -1)
    truth["break_week"] = break_week
    truth["break_size"] = np.where(is_break, cfg.break_size, 0.0)

    # --- market ----------------------------------------------------------
    spot, base = _spot_and_contract(rng, cfg)
    contract = base[lanes] * np.exp(rng.normal(0.0, 0.08, size=n_cust))
    truth["contract_rate"] = contract.round(0)
    spot_c = spot[lanes]  # (C, T)
    spread = (spot_c - contract[:, None]) / contract[:, None]
    # The true model uses the spread standardized over the full history.
    spread_std = (spread - spread.mean()) / spread.std()

    # --- seasonality -----------------------------------------------------
    F = fourier_features(dates, 2)
    delta = {"BCO": np.array([-0.10, -0.06, 0.04, 0.02]), "FWD": np.array([-0.14, -0.08, 0.05, 0.03])}
    season = {k: F @ v for k, v in delta.items()}
    season_c = np.vstack([season[ct] for ct in types])

    # --- latent AR(1) deviations ------------------------------------------
    tau = truth["tau"].to_numpy()
    u = np.vstack([_ar1(rng, T, cfg.rho[ct], tau[i]) for i, ct in enumerate(types)])

    # --- latent rate and bookings ----------------------------------------
    log_lam = (
        truth["alpha"].to_numpy()[:, None]
        + truth["b"].to_numpy()[:, None] * t_years[None, :]
        + truth["beta"].to_numpy()[:, None] * spread_std
        + season_c
        + u
    )
    week_idx = np.arange(T)[None, :]
    log_lam += np.where(
        (break_week[:, None] >= 0) & (week_idx >= break_week[:, None]), truth["break_size"].to_numpy()[:, None], 0.0
    )
    lam = np.exp(log_lam)
    mu = commit[:, None] * lam
    p = cfg.phi / (cfg.phi + mu)
    y = rng.negative_binomial(cfg.phi, p)

    bookings = pd.DataFrame(
        {
            "week_start": np.tile(dates.date, n_cust),
            "customer_id": np.repeat(ids, T),
            "customer_type": np.repeat(types, T),
            "lane": np.repeat(truth["lane"].to_numpy(), T),
            "commitment": np.repeat(commit, T),
            "bookings": y.ravel(),
            "spot_rate": spot_c.ravel().round(1),
            "contract_rate": np.repeat(truth["contract_rate"].to_numpy(), T),
        }
    )
    bookings["week_start"] = pd.to_datetime(bookings["week_start"])
    bookings["spread"] = (bookings["spot_rate"] - bookings["contract_rate"]) / bookings["contract_rate"]
    bookings["realized_fulfillment"] = bookings["bookings"] / bookings["commitment"]

    season_df = pd.DataFrame({"week_start": dates, "BCO": season["BCO"], "FWD": season["FWD"]})
    truth["rho"] = [cfg.rho[ct] for ct in types]
    truth["phi"] = cfg.phi
    return SimulatedData(bookings=bookings, truth=truth, season=season_df)
