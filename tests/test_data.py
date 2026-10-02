import numpy as np
import pandas as pd
import pandera.errors as pa_errors
import pytest
from pydantic import ValidationError

from fulfillcast import FitConfig, SimulationConfig, simulate_bookings, validate_bookings


def test_simulated_panel_shape_and_dates(sim):
    b = sim.bookings
    assert b["customer_id"].nunique() == 100
    assert b["week_start"].nunique() == 104
    assert (b["week_start"].dt.dayofweek == 6).all()  # all Sundays
    assert len(b) == 100 * 104


def test_customer_mix_and_forwarder_share(sim):
    t = sim.truth
    assert (t["customer_type"] == "BCO").sum() == 90
    assert (t["customer_type"] == "FWD").sum() == 10
    share = sim.bookings.groupby("customer_type")["bookings"].sum() / sim.bookings["bookings"].sum()
    assert share["FWD"] == pytest.approx(0.40, abs=0.05)
    assert set(t["archetype"]) == {"stable", "growing", "declining", "opportunistic", "break"}


def test_simulation_is_reproducible():
    a = simulate_bookings(SimulationConfig(seed=7)).bookings
    b = simulate_bookings(SimulationConfig(seed=7)).bookings
    pd.testing.assert_frame_equal(a, b)


def test_simulated_data_passes_schema(sim):
    validate_bookings(sim.bookings)


def test_schema_rejects_bad_rows(sim):
    bad = sim.bookings.head(208).copy()
    bad.loc[0, "bookings"] = -1
    bad.loc[1, "week_start"] = pd.Timestamp("2024-01-08")  # a Monday
    with pytest.raises(pa_errors.SchemaErrors) as err:
        validate_bookings(bad)
    failures = err.value.failure_cases
    assert {"bookings", "week_start"} <= set(failures["column"].dropna())


def test_schema_rejects_duplicates(sim):
    df = sim.bookings[sim.bookings["customer_id"].isin(["C001", "C002"])]
    dup = pd.concat([df, df.head(1)])
    with pytest.raises(pa_errors.SchemaErrors):
        validate_bookings(dup)


def test_configs_forbid_unknown_keys():
    with pytest.raises(ValidationError):
        FitConfig(chians=4)
    with pytest.raises(ValidationError):
        SimulationConfig(break_window=(90, 200))


def test_model_data_uses_training_statistics(model_data, train_test):
    train, _ = train_test
    assert model_data.y.shape == (100, 78)
    assert model_data.fourier.shape == (78, 4)
    assert model_data.spread.mean() == pytest.approx(0.0, abs=1e-9)
    assert model_data.spread.std() == pytest.approx(1.0, abs=1e-9)
    assert model_data.transform.spread_mean == pytest.approx(train["spread"].mean())
    # time is centered on the training window
    assert model_data.t_years.mean() == pytest.approx(0.0, abs=0.01)
    assert set(np.unique(model_data.type_idx)) == {0, 1}
