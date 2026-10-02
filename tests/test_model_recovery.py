"""Recovery test: simulate data with known parameters, fit, check we get them back.

Runs MCMC, so it takes a few minutes. Skip with:  pytest -m "not slow"
"""

import pytest

from fulfillcast import FitConfig, SimulationConfig, fit, prepare_model_data, simulate_bookings
from fulfillcast.config import ArchetypeMix
from fulfillcast.evaluation import parameter_recovery, true_parameters_on_model_scale


@pytest.mark.slow
def test_model_recovers_trend_and_price_sensitivity():
    cfg = SimulationConfig(
        n_weeks=78,
        break_window=(60, 70),
        bco_mix=ArchetypeMix(stable=10, growing=4, declining=4, opportunistic=4, **{"break": 0}),
        fwd_mix=ArchetypeMix(stable=2, growing=0, declining=0, opportunistic=2, **{"break": 0}),
        seed=11,
    )
    sim = simulate_bookings(cfg)
    data = prepare_model_data(sim.bookings)
    result = fit(data, FitConfig(draws=300, tune=500, chains=2, seed=1))

    assert int(result.idata.sample_stats["diverging"].values.sum()) <= 5

    truth = true_parameters_on_model_scale(result, sim.truth, sim.bookings)
    rec = parameter_recovery(result, truth).set_index("parameter")
    assert rec.loc["b", "correlation"] > 0.8
    assert rec.loc["beta", "correlation"] > 0.8
    assert rec.loc["b", "coverage_80"] > 0.6

    theta = result.draws("theta").mean(axis=0)
    opp = truth["archetype"].to_numpy() == "opportunistic"
    assert theta[opp, 3].mean() > theta[~opp, 3].mean() + 0.3
