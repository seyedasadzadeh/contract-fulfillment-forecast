"""Shared feature helpers (used by the simulator and the model)."""

from __future__ import annotations

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.25


def year_fraction(dates) -> np.ndarray:
    """Position within the calendar year in [0, 1)."""
    d = pd.DatetimeIndex(dates)
    return ((d.dayofyear - 1) / DAYS_PER_YEAR).to_numpy()


def fourier_features(dates, n_harmonics: int) -> np.ndarray:
    """Annual Fourier features, shape (n_dates, 2 * n_harmonics): sin, cos per harmonic."""
    frac = year_fraction(dates)
    cols = []
    for k in range(1, n_harmonics + 1):
        angle = 2 * np.pi * k * frac
        cols += [np.sin(angle), np.cos(angle)]
    if not cols:
        return np.zeros((len(frac), 0))
    return np.column_stack(cols)


def years_since(dates, origin) -> np.ndarray:
    """Time in years from ``origin`` (can be negative)."""
    d = pd.DatetimeIndex(dates)
    return ((d - pd.Timestamp(origin)).days / DAYS_PER_YEAR).to_numpy()
