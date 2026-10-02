"""Data contracts for input tables (Pandera).

Validate at the boundary: data is checked once when it enters the package,
so the modelling code inside can trust it.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series


class BookingsSchema(pa.DataFrameModel):
    """Weekly bookings: one row per customer and week (week starts on Sunday)."""

    week_start: Series[pd.Timestamp]
    customer_id: Series[str]
    customer_type: Series[str] = pa.Field(isin=["BCO", "FWD"])
    commitment: Series[int] = pa.Field(gt=0)
    bookings: Series[int] = pa.Field(ge=0)
    spread: Series[float] = pa.Field(ge=-1.0, le=5.0, description="(spot - contract) / contract")

    class Config:
        strict = False  # extra columns (lane, rates, ...) are allowed
        coerce = True

    @pa.check("week_start", name="weeks_start_on_sunday")
    def _sunday(cls, s: Series[pd.Timestamp]) -> Series[bool]:
        return s.dt.dayofweek == 6

    @pa.dataframe_check(name="one_row_per_customer_week")
    def _unique(cls, df: pd.DataFrame) -> bool:
        return not df.duplicated(["customer_id", "week_start"]).any()

    @pa.dataframe_check(name="one_type_per_customer")
    def _one_type(cls, df: pd.DataFrame) -> bool:
        return bool((df.groupby("customer_id")["customer_type"].nunique() == 1).all())

    @pa.dataframe_check(name="complete_weekly_panel")
    def _complete(cls, df: pd.DataFrame) -> bool:
        n_weeks = df["week_start"].nunique()
        return bool((df.groupby("customer_id").size() == n_weeks).all())


def validate_bookings(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a bookings table and return the coerced copy.

    Raises ``pandera.errors.SchemaErrors`` listing every failure case.
    """
    return BookingsSchema.validate(df, lazy=True)
