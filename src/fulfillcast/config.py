"""Validated configuration objects (Pydantic).

Configs live outside the code so they can be versioned, logged and changed
without editing the model. Unknown keys raise an error (``extra="forbid"``),
so a typo such as ``chians: 4`` fails loudly instead of silently using a default.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

ARCHETYPES = ("stable", "growing", "declining", "opportunistic", "break")


class ArchetypeMix(BaseModel):
    """How many customers of each behaviour archetype to simulate, per type."""

    model_config = ConfigDict(extra="forbid")

    stable: int = Field(ge=0)
    growing: int = Field(ge=0)
    declining: int = Field(ge=0)
    opportunistic: int = Field(ge=0)
    break_: int = Field(ge=0, alias="break")

    def total(self) -> int:
        return self.stable + self.growing + self.declining + self.opportunistic + self.break_

    def as_list(self) -> list[str]:
        counts = {
            "stable": self.stable,
            "growing": self.growing,
            "declining": self.declining,
            "opportunistic": self.opportunistic,
            "break": self.break_,
        }
        return [name for name, n in counts.items() for _ in range(n)]


class SimulationConfig(BaseModel):
    """Settings for the synthetic contract-booking dataset."""

    model_config = ConfigDict(extra="forbid")

    n_weeks: int = Field(default=104, ge=20)
    start_date: str = Field(default="2024-01-07", description="Must be a Sunday.")
    n_lanes: int = Field(default=4, ge=1)
    forwarder_share: float = Field(
        default=0.40, gt=0, lt=1, description="Target share of expected weekly bookings from forwarders."
    )
    bco_median_commitment: float = Field(default=8.0, gt=0, description="Containers per week.")
    phi: float = Field(default=12.0, gt=0, description="Negative Binomial dispersion (larger = less noise).")
    rho: dict[str, float] = Field(default_factory=lambda: {"BCO": 0.7, "FWD": 0.6})
    break_window: tuple[int, int] = Field(
        default=(82, 95), description="Week index range in which 'break' customers drop."
    )
    break_size: float = Field(default=-1.2, lt=0, description="Log-scale level drop for 'break' customers.")
    bco_mix: ArchetypeMix = Field(
        default_factory=lambda: ArchetypeMix(stable=55, growing=12, declining=10, opportunistic=9, **{"break": 4})
    )
    fwd_mix: ArchetypeMix = Field(
        default_factory=lambda: ArchetypeMix(stable=3, growing=2, declining=1, opportunistic=3, **{"break": 1})
    )
    seed: int = 42

    @model_validator(mode="after")
    def _check(self) -> SimulationConfig:
        lo, hi = self.break_window
        if not (0 <= lo < hi <= self.n_weeks):
            raise ValueError("break_window must lie inside [0, n_weeks]")
        return self

    @property
    def n_bco(self) -> int:
        return self.bco_mix.total()

    @property
    def n_fwd(self) -> int:
        return self.fwd_mix.total()


class FitConfig(BaseModel):
    """Sampler and model-structure settings."""

    model_config = ConfigDict(extra="forbid")

    draws: int = Field(default=1000, ge=100)
    tune: int = Field(default=1000, ge=100)
    chains: int = Field(default=4, ge=1)
    target_accept: float = Field(default=0.9, gt=0.5, lt=1)
    n_fourier: int = Field(default=2, ge=0, le=6, description="Annual Fourier harmonics.")
    correlated_effects: bool = Field(default=False, description="Use an LKJ correlation between customer parameters.")
    centered: dict[str, list[str]] = Field(
        default_factory=lambda: {"FWD": ["alpha", "b", "beta"]},
        description="Per customer type, parameters fitted in centered form (strong data). Others are non-centered.",
    )
    heavy_tailed: list[str] = Field(
        default_factory=lambda: ["b", "beta"], description="Parameters with a Student-t population distribution."
    )
    nu: float = Field(default=4.0, gt=1, description="Student-t degrees of freedom for heavy-tailed parameters.")
    nuts_sampler: str = Field(default="nutpie", description="'nutpie' (fast, recommended), 'pymc' or 'numpyro'.")
    sampler_kwargs: dict = Field(
        default_factory=dict,
        description="Extra arguments for the NUTS backend, e.g. {'adaptation': 'low_rank'} for nutpie.",
    )
    seed: int = 2026
