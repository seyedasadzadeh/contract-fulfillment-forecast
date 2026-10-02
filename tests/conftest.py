from types import SimpleNamespace

import numpy as np
import pytest

from fulfillcast import FitConfig, SimulationConfig, prepare_model_data, simulate_bookings, split_by_date
from fulfillcast.model import PARAMS, FitResult


@pytest.fixture(scope="session")
def sim():
    return simulate_bookings(SimulationConfig())


@pytest.fixture(scope="session")
def train_test(sim):
    dates = sorted(sim.bookings["week_start"].unique())
    return split_by_date(sim.bookings, dates[77])


@pytest.fixture(scope="session")
def model_data(train_test):
    return prepare_model_data(train_test[0])


def make_fake_fit(data, theta_mean, s=0.15, rho=0.7, phi=50.0, n_draws=400, noise=0.01, seed=0):
    """A FitResult with hand-made posterior draws, so tests can run without MCMC.

    ``theta_mean`` has shape (C, 4) in the order of ``PARAMS``.
    """
    rng = np.random.default_rng(seed)
    C = data.n_customers
    K = data.fourier.shape[1]
    theta = theta_mean[None] + noise * rng.standard_normal((n_draws, C, len(PARAMS)))
    s_arr = np.full((n_draws, C), s)
    rho_arr = np.full((n_draws, 2), rho)
    post = {
        "theta": theta[None],
        "s": s_arr[None],
        "tau": (s_arr * np.sqrt(1 - rho**2))[None],
        "rho": rho_arr[None],
        "u_last": np.zeros((1, n_draws, C)),
        "phi": np.full((1, n_draws), phi),
        "delta": np.zeros((1, n_draws, 2, K)),
    }
    stats = {"diverging": np.zeros((1, n_draws), dtype=bool)}
    return FitResult(idata=SimpleNamespace(posterior=post, sample_stats=stats), data=data, config=FitConfig())
