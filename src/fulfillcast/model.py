"""Hierarchical Bayesian model of contract fulfillment (PyMC).

    bookings[c,t]   ~ NegBinomial(commitment[c,t] * lambda[c,t], phi)
    log lambda[c,t] = alpha_c + b_c * t + beta_c * spread[c,t] + season_k(t) + u[c,t]
    u[c,t]          = rho_k * u[c,t-1] + tau_c * eps[c,t]          (latent AR(1))
    tau_c           = s_c * sqrt(1 - rho_k**2)
    theta_c = (alpha_c, b_c, log s_c, beta_c) ~ StudentT(nu, mu_k, sigma_k)   (k = customer type;
              heavy tails for b and beta, effectively Normal for alpha and log s)

s_c is the *marginal* standard deviation of the deviation u (how far the rate
typically sits from its trend). Parameterizing by s_c instead of the innovation
sd tau_c makes s_c and rho nearly independent in the posterior, which mixes far
better: the data identify the total wobble well, but split it between "size of
shocks" and "how long shocks last" only weakly.

Customer parameters are written in non-centered form (theta = mu + sigma * z),
which avoids the "funnel" geometry that causes divergences in hierarchical models.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from .config import FitConfig
from .data import ModelData

PARAMS = ("alpha", "b", "log_s", "beta")
PRIOR_MU_MEAN = np.array([np.log(0.8), 0.0, np.log(0.15), 0.2])
PRIOR_MU_SD = np.array([0.5, 0.2, 0.5, 0.3])
PRIOR_SIGMA = np.array([0.5, 0.2, 0.5, 0.3])


SAVED_VARS = ("mu", "sigma", "theta", "s", "tau", "rho", "u_last", "phi", "delta")


@dataclass
class FitResult:
    """A fitted model: posterior draws plus everything needed to forecast."""

    idata: object  # arviz DataTree / InferenceData, or a light stand-in after load()
    data: ModelData
    config: FitConfig

    def draws(self, name: str) -> np.ndarray:
        """Posterior draws of a variable, chains stacked: shape (n_draws, ...)."""
        arr = np.asarray(self.idata.posterior[name])
        return arr.reshape((-1,) + arr.shape[2:])

    def n_divergences(self) -> int:
        return int(np.asarray(self.idata.sample_stats["diverging"]).sum())

    def save(self, path: str | Path) -> Path:
        """Save the posterior draws, data and config to one file (.pkl).

        Only the variables needed downstream are kept, as plain numpy arrays,
        so the file loads without ArviZ/NetCDF backends.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        posterior = {v: np.asarray(self.idata.posterior[v]) for v in SAVED_VARS if v in self.idata.posterior}
        diverging = np.asarray(self.idata.sample_stats["diverging"])
        payload = {
            "version": 1,
            "posterior": posterior,
            "diverging": diverging,
            "data": self.data,
            "config": self.config.model_dump(),
        }
        with path.open("wb") as f:
            pickle.dump(payload, f)
        return path

    @classmethod
    def load(cls, path: str | Path) -> FitResult:
        """Load a result written by :meth:`save` (only load files you trust: pickle)."""
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        idata = SimpleNamespace(posterior=payload["posterior"], sample_stats={"diverging": payload["diverging"]})
        return cls(idata=idata, data=payload["data"], config=FitConfig(**payload["config"]))


def _ar1_matrix(rho, T: int):
    """Lower-triangular matrix with entries rho**(t-s) for t >= s."""
    d = np.subtract.outer(np.arange(T), np.arange(T))
    mask = d >= 0
    return pt.switch(mask, rho ** np.maximum(d, 0), 0.0)


def centered_mask(data: ModelData, cfg: FitConfig) -> np.ndarray:
    """Boolean (C, P): True where a customer parameter uses the centered form.

    Non-centered (theta = mu + sigma * z) is best when a customer's own data are
    weak; centered (theta ~ Normal(mu, sigma)) is best when they are strong.
    Using the wrong one shows up as low ESS / high R-hat on the type means.
    """
    M = np.zeros((data.n_customers, len(PARAMS)), dtype=bool)
    for type_name, params in cfg.centered.items():
        if type_name not in data.type_names:
            continue
        rows = data.type_idx == data.type_names.index(type_name)
        for p in params:
            M[rows, PARAMS.index(p)] = True
    return M


def build_model(data: ModelData, cfg: FitConfig | None = None) -> pm.Model:
    """Build the PyMC model for the given data."""
    cfg = cfg or FitConfig()
    C, T = data.y.shape
    k = data.type_idx
    coords = {
        "customer": data.customer_ids,
        "week": data.dates,
        "type": list(data.type_names),
        "param": list(PARAMS),
        "fourier": [f"f{i}" for i in range(data.fourier.shape[1])],
    }

    with pm.Model(coords=coords) as model:
        # --- population (type) level ------------------------------------
        mu = pm.Normal("mu", PRIOR_MU_MEAN, PRIOR_MU_SD, dims=("type", "param"))

        if cfg.correlated_effects:
            z = pm.Normal("z", 0.0, 1.0, dims=("customer", "param"))
            chols = []
            for name in data.type_names:
                chol, _, _ = pm.LKJCholeskyCov(
                    f"chol_{name}",
                    n=len(PARAMS),
                    eta=2.0,
                    sd_dist=pm.HalfNormal.dist(PRIOR_SIGMA, shape=len(PARAMS)),
                    compute_corr=True,
                )
                chols.append(chol)
            offs = pt.stack([z @ L.T for L in chols])  # (type, C, P)
            theta = mu[k] + offs[k, np.arange(C)]
        else:
            sigma = pm.HalfNormal("sigma", PRIOR_SIGMA, dims=("type", "param"))
            # Mixed parameterization: centered where a customer's own data are
            # strong (large forwarders), non-centered where data are weak.
            M = centered_mask(data, cfg)
            # Heavy tails (Student-t) for trend and price sensitivity: a few
            # customers are genuinely extreme (fast growth, opportunism), and a
            # Normal population would shrink them too hard toward the type mean.
            nu = np.array([cfg.nu if p in cfg.heavy_tailed else 1e4 for p in PARAMS])
            z = pm.StudentT(
                "z",
                nu=np.broadcast_to(nu, (C, len(PARAMS))),
                mu=pt.switch(M, mu[k], 0.0),
                sigma=pt.switch(M, sigma[k], 1.0),
                dims=("customer", "param"),
            )
            theta = pt.switch(M, z, mu[k] + sigma[k] * z)

        theta = pm.Deterministic("theta", theta, dims=("customer", "param"))
        alpha, b, log_s, beta = (theta[:, i] for i in range(len(PARAMS)))
        s = pm.Deterministic("s", pt.exp(log_s), dims="customer")

        # --- seasonality (shared within type) -----------------------------
        if data.fourier.shape[1] > 0:
            delta = pm.Normal("delta", 0.0, 0.2, dims=("type", "fourier"))
            season = pt.dot(data.fourier, delta.T)  # (T, type)
            season_c = season[:, k].T  # (C, T)
        else:
            season_c = 0.0

        # --- latent AR(1) deviation ---------------------------------------
        rho = pm.Beta("rho", 8.0, 2.0, dims="type")
        eps = pm.Normal("eps", 0.0, 1.0, dims=("customer", "week"))
        tau = pm.Deterministic("tau", s * pt.sqrt(1.0 - rho[k] ** 2), dims="customer")
        scale = pt.repeat(tau[:, None], T, axis=1)
        scale = pt.set_subtensor(scale[:, 0], s)  # stationary start: sd of u[0] is s
        innov = eps * scale
        u_by_type = pt.stack([innov @ _ar1_matrix(rho[i], T).T for i in range(len(data.type_names))])
        u = u_by_type[k, np.arange(C)]  # (C, T)
        pm.Deterministic("u_last", u[:, -1], dims="customer")

        # --- observation model ---------------------------------------------
        log_lam = alpha[:, None] + b[:, None] * data.t_years[None, :] + beta[:, None] * data.spread + season_c + u
        inv_sqrt_phi = pm.HalfNormal("inv_sqrt_phi", 1.0)
        phi = pm.Deterministic("phi", 1.0 / inv_sqrt_phi**2)
        pm.NegativeBinomial(
            "bookings",
            mu=data.commitment * pt.exp(log_lam),
            alpha=phi,
            observed=data.y,
            dims=("customer", "week"),
        )
    return model


def fit(data: ModelData, cfg: FitConfig | None = None, progressbar: bool = False) -> FitResult:
    """Fit the model with NUTS and return a :class:`FitResult`."""
    cfg = cfg or FitConfig()
    model = build_model(data, cfg)
    with model:
        idata = pm.sample(
            draws=cfg.draws,
            tune=cfg.tune,
            chains=cfg.chains,
            target_accept=cfg.target_accept,
            nuts_sampler=cfg.nuts_sampler,
            random_seed=cfg.seed,
            progressbar=progressbar,
            nuts_sampler_kwargs=dict(cfg.sampler_kwargs),
            var_names=["mu", "theta", "s", "tau", "rho", "u_last", "phi", "inv_sqrt_phi", "z"]
            + (["sigma"] if not cfg.correlated_effects else [])
            + (["delta"] if data.fourier.shape[1] > 0 else []),
        )
    return FitResult(idata=idata, data=data, config=cfg)
