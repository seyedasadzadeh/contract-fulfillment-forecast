"""Convergence checks used as a quality gate before a fit is used downstream."""

from __future__ import annotations

import pandas as pd

from .model import FitResult

GATE_VARS = ("mu", "sigma", "rho", "phi", "delta", "theta")


def convergence_table(result: FitResult, var_names=GATE_VARS) -> pd.DataFrame:
    """R-hat and effective sample size per scalar parameter (needs the ArviZ object from ``fit``)."""
    import arviz as az

    names = [v for v in var_names if v in result.idata.posterior]
    summ = az.summary(result.idata, var_names=names)
    return summ[["r_hat", "ess_bulk", "ess_tail"]].astype(float)


def quality_gate(result: FitResult, rhat_max: float = 1.01, ess_min: float = 400, max_divergences: int = 0) -> dict:
    """Pass/fail summary: should this fit replace the one in production?

    Returns a dict with ``passed`` and the reasons, so a pipeline can keep the
    previous model when a new fit has not converged.
    """
    table = convergence_table(result)
    bad_rhat = table[table["r_hat"] > rhat_max]
    low_ess = table[table["ess_bulk"] < ess_min]
    n_div = result.n_divergences()
    reasons = []
    if n_div > max_divergences:
        reasons.append(f"{n_div} divergences")
    if len(bad_rhat):
        reasons.append(f"{len(bad_rhat)} parameters with R-hat > {rhat_max}")
    if len(low_ess):
        reasons.append(f"{len(low_ess)} parameters with bulk ESS < {ess_min}")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "max_r_hat": float(table["r_hat"].max()),
        "min_ess_bulk": float(table["ess_bulk"].min()),
        "worst": table.sort_values("ess_bulk").head(5).round(3).to_dict(orient="index"),
    }
