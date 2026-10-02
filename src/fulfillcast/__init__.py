"""fulfillcast: Bayesian forecasting of contract fulfillment for container shipping.

Public API
----------
Data:      simulate_bookings, validate_bookings, prepare_model_data, split_by_date
Model:     fit, build_model, FitResult
Outputs:   forecast, segment_probabilities, parameter_summary
Drift:     surprise_report, deviation_report, compare_fits
Config:    SimulationConfig, FitConfig, SegmentRules
"""

from .config import FitConfig, SimulationConfig
from .data import ModelData, prepare_model_data, split_by_date
from .drift import compare_fits, deviation_report, surprise_report
from .forecast import Forecast, forecast
from .model import FitResult, build_model, fit
from .schemas import validate_bookings
from .segments import SegmentRules, parameter_summary, segment_probabilities
from .simulate import SimulatedData, simulate_bookings

__version__ = "0.1.0"

__all__ = [
    "FitConfig",
    "FitResult",
    "Forecast",
    "ModelData",
    "SegmentRules",
    "SimulatedData",
    "SimulationConfig",
    "build_model",
    "compare_fits",
    "deviation_report",
    "fit",
    "forecast",
    "parameter_summary",
    "prepare_model_data",
    "segment_probabilities",
    "simulate_bookings",
    "split_by_date",
    "surprise_report",
    "validate_bookings",
]
