"""Turn a validated bookings table into model-ready arrays.

All standardization statistics are computed on the training window only and
stored, so future weeks are transformed exactly the same way (no leakage).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import fourier_features, years_since
from .schemas import validate_bookings

TYPE_NAMES = ("BCO", "FWD")


@dataclass
class Transform:
    """Statistics fitted on the training window, reused for forecasting."""

    spread_mean: float
    spread_sd: float
    t_origin: pd.Timestamp
    n_fourier: int

    def spread(self, raw: np.ndarray) -> np.ndarray:
        return (raw - self.spread_mean) / self.spread_sd

    def t_years(self, dates) -> np.ndarray:
        return years_since(dates, self.t_origin)

    def fourier(self, dates) -> np.ndarray:
        return fourier_features(dates, self.n_fourier)


@dataclass
class ModelData:
    """Arrays for the model, customers in rows and weeks in columns."""

    customer_ids: np.ndarray
    type_idx: np.ndarray  # 0 = BCO, 1 = FWD
    dates: pd.DatetimeIndex
    y: np.ndarray  # bookings (C, T)
    commitment: np.ndarray  # (C, T)
    spread: np.ndarray  # standardized (C, T)
    t_years: np.ndarray  # (T,)
    fourier: np.ndarray  # (T, 2K)
    transform: Transform
    type_names: tuple[str, ...] = field(default=TYPE_NAMES)

    @property
    def n_customers(self) -> int:
        return len(self.customer_ids)

    @property
    def n_weeks(self) -> int:
        return len(self.dates)


def to_wide(bookings: pd.DataFrame, column: str) -> pd.DataFrame:
    """Pivot one column to a customer x week table."""
    return bookings.pivot(index="customer_id", columns="week_start", values=column).sort_index()


def split_by_date(bookings: pd.DataFrame, train_end) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into weeks <= train_end (train) and weeks after (test)."""
    train_end = pd.Timestamp(train_end)
    train = bookings[bookings["week_start"] <= train_end].copy()
    test = bookings[bookings["week_start"] > train_end].copy()
    return train, test


def prepare_model_data(bookings: pd.DataFrame, n_fourier: int = 2) -> ModelData:
    """Validate the bookings table and build :class:`ModelData` for fitting."""
    df = validate_bookings(bookings)
    y = to_wide(df, "bookings")
    m = to_wide(df, "commitment")
    s = to_wide(df, "spread")
    types = df.groupby("customer_id")["customer_type"].first().reindex(y.index)
    dates = pd.DatetimeIndex(y.columns)

    raw_spread = s.to_numpy(dtype=float)
    transform = Transform(
        spread_mean=float(raw_spread.mean()),
        spread_sd=float(raw_spread.std()),
        t_origin=dates[0] + (dates[-1] - dates[0]) / 2,
        n_fourier=n_fourier,
    )
    return ModelData(
        customer_ids=y.index.to_numpy(),
        type_idx=types.map({n: i for i, n in enumerate(TYPE_NAMES)}).to_numpy(dtype=int),
        dates=dates,
        y=y.to_numpy(dtype=int),
        commitment=m.to_numpy(dtype=float),
        spread=transform.spread(raw_spread),
        t_years=transform.t_years(dates),
        fourier=transform.fourier(dates),
        transform=transform,
    )
