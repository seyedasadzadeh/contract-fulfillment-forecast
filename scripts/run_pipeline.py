"""End-to-end run: simulate -> fit -> forecast -> segments -> drift -> evaluation.

Usage (from the project root, with the package installed):

    python scripts/run_pipeline.py                # full run
    python scripts/run_pipeline.py --quick        # fewer draws, for a smoke test

Outputs are written to ``outputs/``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from fulfillcast import (
    FitConfig,
    FitResult,
    SimulationConfig,
    deviation_report,
    fit,
    forecast,
    parameter_summary,
    prepare_model_data,
    segment_probabilities,
    simulate_bookings,
    split_by_date,
    surprise_report,
)
from fulfillcast.diagnostics import quality_gate
from fulfillcast.evaluation import forecast_evaluation, parameter_recovery, true_parameters_on_model_scale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train-weeks", type=int, default=78, help="weeks used for fitting; the rest is the forecast horizon"
    )
    ap.add_argument("--quick", action="store_true", help="short sampling run for a smoke test")
    ap.add_argument("--draws", type=int, default=None)
    ap.add_argument("--tune", type=int, default=None)
    ap.add_argument("--chains", type=int, default=None)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--load-fit", default=None, help="reuse a saved fit (.pkl) instead of running MCMC")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    # 1. Data --------------------------------------------------------------
    sim = simulate_bookings(SimulationConfig())
    Path("data").mkdir(exist_ok=True)
    sim.bookings.to_csv("data/bookings.csv", index=False)
    sim.truth.to_csv("data/customers_truth.csv", index=False)
    dates = sorted(sim.bookings["week_start"].unique())
    train_end = dates[args.train_weeks - 1]
    train, test = split_by_date(sim.bookings, train_end)
    print(
        f"Train: {train['week_start'].min().date()} .. {train_end.date()} ({args.train_weeks} weeks); "
        f"forecast {len(dates) - args.train_weeks} weeks"
    )

    # 2. Fit ---------------------------------------------------------------
    cfg = FitConfig(draws=300, tune=400, chains=2) if args.quick else FitConfig()
    overrides = {k: v for k, v in {"draws": args.draws, "tune": args.tune, "chains": args.chains}.items() if v}
    cfg = cfg.model_copy(update=overrides)
    data = prepare_model_data(train, n_fourier=cfg.n_fourier)
    if args.load_fit:
        result = FitResult.load(args.load_fit)
        print(f"Loaded fit from {args.load_fit}")
    else:
        t0 = time.time()
        result = fit(data, cfg, progressbar=True)
        print(f"Fitted in {time.time() - t0:.0f}s")
        result.save(out / "fit_result.pkl")
    n_div = result.n_divergences()
    if args.load_fit:
        gate = {"passed": None, "reasons": ["loaded fit: convergence not re-checked"]}
        print(f"Divergences: {n_div}")
    else:
        gate = quality_gate(result)
        print(f"Divergences: {n_div}; quality gate passed: {gate['passed']} {gate['reasons']}")
        print(f"max R-hat {gate['max_r_hat']:.3f}, min bulk ESS {gate['min_ess_bulk']:.0f}")

    # 3. Forecast ----------------------------------------------------------
    # Production forecast: the future spread is unknown, so it is simulated (AR(1)).
    fc = forecast(result, test, spread_scenario="ar1")
    fc.fulfillment_summary().to_csv(out / "fulfillment_forecast.csv", index=False)
    fc.weekly_summary().to_csv(out / "weekly_forecast.csv", index=False)

    # 4. Segments and parameters ------------------------------------------
    seg = segment_probabilities(result)
    seg.to_csv(out / "segments.csv", index=False)
    parameter_summary(result).to_csv(out / "customer_parameters.csv", index=False)

    # 5. Drift -------------------------------------------------------------
    # Monitoring happens after the fact, when the market is known: judge the
    # bookings against a forecast that uses the realized spread.
    fc_monitor = forecast(result, test, spread_scenario="observed")
    surprise = surprise_report(fc_monitor, test)
    surprise.to_csv(out / "drift_surprise.csv", index=False)
    deviation_report(result).to_csv(out / "drift_deviation.csv", index=False)

    # 6. Evaluation against the simulator's truth --------------------------
    truth_ms = true_parameters_on_model_scale(result, sim.truth, sim.bookings)
    rec = parameter_recovery(result, truth_ms)
    ev = forecast_evaluation(fc, test, train)
    scenario_eval = {
        name: forecast_evaluation(forecast(result, test, spread_scenario=name), test, train)["summary"]
        for name in ("persistence", "observed")
    }
    scenario_eval["ar1"] = ev["summary"]
    scenario_table = pd.DataFrame(scenario_eval).T.round(3)
    seg_vs_truth = pd.crosstab(
        sim.truth.set_index("customer_id").loc[seg["customer_id"], "archetype"].to_numpy(),
        seg["segment"].to_numpy(),
        rownames=["true archetype"],
        colnames=["assigned segment"],
    )
    drift_vs_truth = pd.crosstab(
        sim.truth.set_index("customer_id").loc[surprise["customer_id"], "archetype"].to_numpy(),
        surprise["alarm"].to_numpy(),
        rownames=["true archetype"],
        colnames=["drift alarm"],
    )
    ev["per_customer"].to_csv(out / "forecast_evaluation.csv", index=False)

    try:
        from fulfillcast.plotting import plot_customers

        t = sim.truth.set_index("customer_id")
        picks = [t.index[t["archetype"] == a][0] for a in ("stable", "growing", "declining", "opportunistic", "break")]
        titles = [f"{c} ({t.loc[c, 'customer_type']}, true: {t.loc[c, 'archetype']})" for c in picks]
        plot_customers(fc, train, test, picks, titles, path=str(out / "example_customers.png"))
    except ImportError:
        print("matplotlib not installed: skipping plot")

    print("\nParameter recovery\n", rec.round(3).to_string(index=False))
    print("\nForecast evaluation by spread scenario (observed = oracle, not available at forecast time)")
    print(scenario_table.to_string())
    print("\nSegments vs truth\n", seg_vs_truth)
    print("\nDrift alarms vs truth\n", drift_vs_truth)

    report = {
        "divergences": n_div,
        "quality_gate": gate,
        "recovery": rec.to_dict(orient="records"),
        "forecast": ev["summary"],
        "forecast_by_spread_scenario": scenario_eval,
        "segments_vs_truth": seg_vs_truth.to_dict(),
        "drift_vs_truth": drift_vs_truth.to_dict(),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
