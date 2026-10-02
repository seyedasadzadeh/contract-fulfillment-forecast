"""Behaviour segmentation with probabilities.

Segment rules are applied to every posterior draw. The share of draws in which
a customer falls in a segment is the probability of that segment, so customers
with little data get an honest "uncertain" instead of a confident wrong label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .model import PARAMS, FitResult

SEGMENTS = ("price_driven", "declining", "growing", "volatile", "stable")


class SegmentRules(BaseModel):
    """Thresholds on the customer parameters (model scale)."""

    model_config = ConfigDict(extra="forbid")

    growing_b: float = Field(default=0.15, gt=0, description="trend per year on the log scale")
    declining_b: float = Field(default=-0.15, lt=0)
    price_driven_beta: float = Field(default=0.45, gt=0, description="log effect of +1 sd spread")
    volatile_s: float = Field(default=0.30, gt=0, description="marginal sd of the latent deviation")
    min_probability: float = Field(default=0.8, gt=0.5, lt=1)


def segment_draws(result: FitResult, rules: SegmentRules | None = None) -> np.ndarray:
    """Segment label index per draw and customer: (S, C).

    Labels are mutually exclusive, in priority order:
    price_driven > declining > growing > volatile > stable.
    """
    rules = rules or SegmentRules()
    theta = result.draws("theta")
    b = theta[..., PARAMS.index("b")]
    beta = theta[..., PARAMS.index("beta")]
    s = result.draws("s")
    conditions = [
        beta > rules.price_driven_beta,
        b < rules.declining_b,
        b > rules.growing_b,
        s > rules.volatile_s,
    ]
    return np.select(conditions, [0, 1, 2, 3], default=4)


def segment_probabilities(result: FitResult, rules: SegmentRules | None = None) -> pd.DataFrame:
    """Probability of each segment per customer, plus the assigned segment.

    A customer is assigned its most probable segment only if that probability
    reaches ``rules.min_probability``; otherwise it is ``"uncertain"``.
    """
    rules = rules or SegmentRules()
    labels = segment_draws(result, rules)
    probs = np.stack([(labels == i).mean(axis=0) for i in range(len(SEGMENTS))], axis=1)
    out = pd.DataFrame(probs, columns=[f"p_{s}" for s in SEGMENTS])
    best = probs.argmax(axis=1)
    out.insert(0, "customer_id", result.data.customer_ids)
    out.insert(1, "customer_type", np.asarray(result.data.type_names)[result.data.type_idx])
    out["segment"] = np.where(probs.max(axis=1) >= rules.min_probability, np.asarray(SEGMENTS)[best], "uncertain")
    out["segment_probability"] = probs.max(axis=1)
    return out


def parameter_summary(result: FitResult, interval: float = 0.8) -> pd.DataFrame:
    """Posterior mean and interval of each customer's behaviour parameters."""
    lo, hi = (1 - interval) / 2, 1 - (1 - interval) / 2
    theta = result.draws("theta")
    cols = {"customer_id": result.data.customer_ids}
    named = {p: theta[..., i] for i, p in enumerate(PARAMS)}
    named["level"] = np.exp(named["alpha"])  # typical fulfillment at the window midpoint
    named["s"] = result.draws("s")
    for name in ("level", "b", "beta", "s"):
        x = named[name]
        cols[f"{name}_mean"] = x.mean(axis=0)
        cols[f"{name}_lower"] = np.quantile(x, lo, axis=0)
        cols[f"{name}_upper"] = np.quantile(x, hi, axis=0)
    return pd.DataFrame(cols)
