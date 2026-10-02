"""Plots for a quick visual check (needs matplotlib: pip install -e ".[dev]")."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .forecast import Forecast


def plot_customers(
    fc: Forecast,
    history: pd.DataFrame,
    actuals: pd.DataFrame | None,
    customer_ids: list[str],
    titles: list[str] | None = None,
    path: str | None = None,
):
    """Realized fulfillment history, forecast fan (median, 50% and 80% bands) and actuals."""
    import matplotlib.pyplot as plt

    n = len(customer_ids)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.6 * n), sharex=True)
    axes = np.atleast_1d(axes)
    pos = {c: i for i, c in enumerate(fc.customer_ids)}
    for ax, cid, title in zip(axes, customer_ids, titles or customer_ids, strict=True):
        h = history[history["customer_id"] == cid]
        ax.plot(h["week_start"], h["realized_fulfillment"], color="0.35", lw=1, label="history")
        i = pos[cid]
        f = fc.samples[:, i, :] / fc.commitment[i][None, :]
        q = np.quantile(f, [0.1, 0.25, 0.5, 0.75, 0.9], axis=0)
        ax.fill_between(fc.dates, q[0], q[4], color="#4C78A8", alpha=0.18, lw=0, label="80% interval")
        ax.fill_between(fc.dates, q[1], q[3], color="#4C78A8", alpha=0.32, lw=0, label="50% interval")
        ax.plot(fc.dates, q[2], color="#4C78A8", lw=1.6, label="forecast median")
        if actuals is not None:
            a = actuals[actuals["customer_id"] == cid]
            ax.plot(a["week_start"], a["realized_fulfillment"], color="#E45756", lw=1, label="actual")
        ax.axvline(fc.dates[0], color="0.6", ls=":", lw=1)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel("fulfillment")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(ncol=5, fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig
