"""Forecast, segmentation and drift logic, tested with hand-made posteriors (no MCMC)."""

import numpy as np
import pandas as pd
import pytest

from conftest import make_fake_fit
from fulfillcast import SegmentRules, forecast, segment_probabilities, surprise_report
from fulfillcast.drift import cusum_alarms, predictive_pit
from fulfillcast.evaluation import crps_from_samples
from fulfillcast.model import PARAMS

LOG_08, LOG_015 = np.log(0.8), np.log(0.15)


def _theta(C, alpha=LOG_08, b=0.0, log_s=LOG_015, beta=0.1):
    th = np.zeros((C, len(PARAMS)))
    th[:, 0], th[:, 1], th[:, 2], th[:, 3] = alpha, b, log_s, beta
    return th


def test_forecast_shapes_and_level(model_data, train_test):
    _, test = train_test
    fit = make_fake_fit(model_data, _theta(model_data.n_customers, beta=0.0), s=0.01)
    fc = forecast(fit, test, n_samples=300)
    assert fc.samples.shape == (300, 100, 26)
    summ = fc.fulfillment_summary()
    assert len(summ) == 100
    # with alpha = log 0.8, no trend, no price effect and tiny wobble, fulfillment is about 0.8
    assert summ["fulfillment_median"].median() == pytest.approx(0.8, abs=0.05)
    assert (summ["fulfillment_lower"] <= summ["fulfillment_median"]).all()
    assert (summ["fulfillment_median"] <= summ["fulfillment_upper"]).all()


def test_forecast_rejects_overlapping_weeks(model_data, train_test):
    train, _ = train_test
    fit = make_fake_fit(model_data, _theta(model_data.n_customers))
    with pytest.raises(ValueError, match="after the training window"):
        forecast(fit, train)


def test_negative_trend_lowers_forecast(model_data, train_test):
    _, test = train_test
    flat = forecast(make_fake_fit(model_data, _theta(100, b=0.0)), test, n_samples=200, seed=1)
    down = forecast(make_fake_fit(model_data, _theta(100, b=-0.5)), test, n_samples=200, seed=1)
    assert down.fulfillment_samples().mean() < flat.fulfillment_samples().mean()


def test_segments_follow_parameters(model_data):
    th = _theta(100)
    th[0, 3] = 1.0  # price driven
    th[1, 1] = -0.5  # declining
    th[2, 1] = 0.5  # growing
    th[3, 2] = np.log(0.6)  # volatile
    seg = segment_probabilities(make_fake_fit(model_data, th), SegmentRules())
    seg = seg.set_index("customer_id")
    ids = model_data.customer_ids
    assert seg.loc[ids[0], "segment"] == "price_driven"
    assert seg.loc[ids[1], "segment"] == "declining"
    assert seg.loc[ids[2], "segment"] == "growing"
    assert seg.loc[ids[4], "segment"] == "stable"
    prob_cols = [c for c in seg.columns if c.startswith("p_")]
    np.testing.assert_allclose(seg[prob_cols].sum(axis=1), 1.0)


def test_uncertain_customer_is_not_labelled(model_data):
    th = _theta(100, b=0.15)  # right on the growing threshold
    seg = segment_probabilities(make_fake_fit(model_data, th, noise=0.1))
    assert (seg["segment"] == "uncertain").mean() > 0.9


def test_pit_is_roughly_uniform_when_model_is_right():
    rng = np.random.default_rng(0)
    samples = rng.poisson(10, size=(2000, 50, 20))

    class FC:  # minimal stand-in for Forecast
        pass

    fc = FC()
    fc.samples = samples
    obs = rng.poisson(10, size=(50, 20))
    pit = predictive_pit(fc, obs)
    assert abs(pit.mean() - 0.5) < 0.03
    a_down, a_up, _ = cusum_alarms(pit)
    assert (a_down >= 0).mean() < 0.1


def test_cusum_catches_a_level_drop(model_data, train_test):
    _, test = train_test
    fit = make_fake_fit(model_data, _theta(100, beta=0.0), s=0.01)
    fc = forecast(fit, test, n_samples=500, seed=3)
    actual = test.copy()
    # customer 0 halves its bookings from the 5th forecast week
    first = model_data.customer_ids[0]
    dates = sorted(actual["week_start"].unique())
    mask = (actual["customer_id"] == first) & (actual["week_start"] >= dates[4])
    actual["bookings"] = np.rint(fc.samples.mean(axis=0).ravel()).astype(int)  # "as expected" for everyone
    actual.loc[mask, "bookings"] = (actual.loc[mask, "bookings"] * 0.3).astype(int)
    rep = surprise_report(fc, actual).set_index("customer_id")
    assert rep.loc[first, "alarm"] == "down"
    assert rep.loc[first, "alarm_week"] >= pd.Timestamp(dates[4])


def test_crps_matches_normal_closed_form():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20000, 1))
    y = np.array([0.0])
    # CRPS of N(0,1) at 0 is (sqrt(2) - 1) / sqrt(pi)
    assert crps_from_samples(x, y)[0] == pytest.approx((np.sqrt(2) - 1) / np.sqrt(np.pi), abs=0.01)


def test_fit_result_save_and_load_roundtrip(model_data, tmp_path):
    from fulfillcast import FitResult

    fit = make_fake_fit(model_data, _theta(100))
    path = fit.save(tmp_path / "fit.pkl")
    loaded = FitResult.load(path)
    np.testing.assert_array_equal(loaded.draws("theta"), fit.draws("theta"))
    assert loaded.n_divergences() == 0
    np.testing.assert_array_equal(loaded.data.y, model_data.y)


def test_both_drift_rules_flag_a_collapse(model_data, train_test):
    _, test = train_test
    fit = make_fake_fit(model_data, _theta(100, beta=0.0), s=0.01)
    fc = forecast(fit, test, spread_scenario="persistence", n_samples=500, seed=4)
    actual = test.copy()
    actual["bookings"] = np.rint(fc.samples.mean(axis=0).ravel()).astype(int)
    first = model_data.customer_ids[0]
    actual.loc[actual["customer_id"] == first, "bookings"] = 0
    for method in ("cumulative", "cusum"):
        rep = surprise_report(fc, actual, method=method).set_index("customer_id")
        assert rep.loc[first, "alarm"] == "down", method
        assert (rep["alarm"] != "none").mean() < 0.1, method


def test_ar1_spread_paths_revert_and_spread_out():
    from fulfillcast.forecast import simulate_spread_paths

    rng = np.random.default_rng(0)
    T, C = 200, 5
    x = np.zeros((C, T))
    for t in range(1, T):
        x[:, t] = 0.8 * x[:, t - 1] + rng.normal(0, 0.3, C)
    x[:, -1] = 2.0  # start far above the mean
    paths = simulate_spread_paths(x, horizon=30, n_paths=2000, rng=rng)
    assert paths.shape == (2000, C, 30)
    assert abs(paths[:, :, -1].mean()) < 0.3  # reverted toward the mean
    assert paths[:, :, -1].std() > paths[:, :, 0].std()  # uncertainty grows
