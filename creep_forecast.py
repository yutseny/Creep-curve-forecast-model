"""creep_forecast.py - forecast an ongoing creep test with the H3 hybrid model.

`master_model.py` reproduces the manuscript's retrospective benchmark, in which
every curve is complete and the observation boundary is imposed artificially.
This module applies the same model to a test that is still running: you supply
the loading condition and the strain measured so far, and it returns the
forecast remaining trajectory, rupture life and split-conformal bands.

The protocol is the one validated in the manuscript (Methods 2.4-2.5):

  1. The screened cohort (output of `master_model.py --stage theta-fit`) is
     split, material-stratified, into proper-training (70%) and calibration
     (30%) rows.
  2. The observation boundary t_b is the last time recorded for the new test.
     It is expressed as the reveal fraction r = t_b / t_r,cond, where t_r,cond
     is the condition-only life estimate (Eq. 10), and every training curve is
     featurized at the same r so that the new test is in-distribution.
  3. H3 = condition-only physics baseline -> bounded P5 prefix update ->
     bounded ExtraTrees residual correction -> material-median shape (Eq. 9).
  4. Calibration rows give split-conformal 68% / 90% whole-suffix strain bands
     and a 90% interval on log rupture life.

Python use
----------
    from creep_forecast import CreepForecaster

    forecaster = CreepForecaster()                    # loads the cohort once
    result = forecaster.forecast(
        material="A617", temperature=900, stress=30,  # degC, MPa
        time=time_hours, strain=strain_percent,       # measured so far
    )
    print(result.summary)
    result.save("my_test_forecast")                   # CSV + JSON + PNG

Command line
------------
    python creep_forecast.py --material A617 --temperature 900 --stress 30 \\
        --curve my_test.csv --strain-units fraction --out my_test_forecast

See README.md ("Forecasting an ongoing test") and examples/worked_example.py.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import master_model as mm

# Reveal fractions covered by the manuscript's validation sweep (mm.REVEALS).
VALIDATED_REVEAL_RANGE = (min(mm.REVEALS), max(mm.REVEALS))
# Below this many prefix points the Omega and smoothed-rate features are empty.
RECOMMENDED_PREFIX_POINTS = 10
NEW_TEST_FILE = "__new_test__"


@dataclass
class ForecastResult:
    """Forecast for one ongoing test.

    Attributes
    ----------
    summary : dict
        Scalar results: predicted rupture life and terminal strain, the
        condition-only reference values, reveal fraction, P5/H3 corrections,
        conformal quantiles, support flags and warnings.
    curve : DataFrame
        Forecast on a time grid from 0 to the predicted rupture life:
        `time_h`, `strain_percent` (H3 center), `lower_68`, `upper_68`,
        `lower_90`, `upper_90`.  Strain is creep strain in percent, referenced
        to the first measurement (Eq. 1).
    observed : DataFrame
        The prefix as used by the model (`time_h`, `strain_percent`).
    """

    summary: dict
    curve: pd.DataFrame
    observed: pd.DataFrame
    warnings: list[str] = field(default_factory=list)

    def save(self, directory, plot=True) -> Path:
        """Write forecast_curve.csv, forecast_summary.json and forecast.png."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.curve.to_csv(directory / "forecast_curve.csv", index=False)
        self.observed.to_csv(directory / "observed_prefix.csv", index=False)
        with open(directory / "forecast_summary.json", "w") as handle:
            json.dump(_json_ready(self.summary), handle, indent=2)
        if plot:
            try:
                self.plot(directory / "forecast.png")
            except ImportError:
                print("matplotlib not installed; skipping forecast.png")
        return directory

    def plot(self, path=None, ax=None, actual=None, ylim=None):
        """Plot the prefix, the H3 forecast and its bands.

        `actual` may be a (time, strain_percent) pair to overlay the measured
        remainder when it is known (e.g. in a retrospective check).
        """
        import matplotlib.pyplot as plt

        own_figure = ax is None
        if own_figure:
            _, ax = plt.subplots(figsize=(6.4, 4.2))
        s = self.summary
        t_b = s["observation_boundary_h"]
        future = self.curve[self.curve["time_h"] >= t_b]
        ax.fill_between(future["time_h"], future["lower_90"], future["upper_90"],
                        color="tab:blue", alpha=0.12, lw=0, label="90% band")
        ax.fill_between(future["time_h"], future["lower_68"], future["upper_68"],
                        color="tab:blue", alpha=0.25, lw=0, label="68% band")
        ax.plot(self.curve["time_h"], self.curve["strain_percent"], color="tab:blue",
                lw=2, label="H3 forecast")
        if actual is not None:
            ax.plot(actual[0], actual[1], color="0.35", lw=1.2, ls="--",
                    label="measured (withheld)")
        ax.plot(self.observed["time_h"], self.observed["strain_percent"],
                color="k", lw=2.5, label="observed so far")
        ax.axvline(t_b, color="k", ls=":", lw=1)

        # Rupture life: H3 point and 90% interval drawn as an error bar near the
        # top of the axes, with the condition-only estimate for comparison.
        life = s["predicted_rupture_life_h"]
        lo, hi = s["rupture_life_interval_90_h"]
        top = ylim or 1.15 * max(float(future["upper_68"].max()),
                                 float(np.max(actual[1])) if actual is not None else 0.0)
        ax.set_ylim(0, top)
        y = 0.93 * top
        ax.errorbar([life], [y], xerr=[[life - lo], [hi - life]], fmt="o", color="tab:red",
                    ms=5, capsize=3, lw=1.2, label="H3 life, 90% interval")
        ax.plot([s["condition_only_life_h"]], [y], marker="x", color="0.3", ms=7, ls="none",
                label="condition-only life")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Creep strain (%)")
        ax.set_title(f"{s['material']}, {s['temperature_C']:g} °C, {s['stress_MPa']:g} MPa"
                     f"  (r = {s['reveal_fraction']:.2f})", fontsize=10)
        reference = max(life, s["condition_only_life_h"],
                        float(np.max(actual[0])) if actual is not None else 0.0)
        ax.set_xlim(0, 1.6 * reference)
        ax.legend(fontsize=7, loc="center right")
        if own_figure:
            ax.figure.tight_layout()
            if path is not None:
                ax.figure.savefig(path, dpi=150)
                plt.close(ax.figure)
        return ax


class CreepForecaster:
    """Forecast ongoing creep tests from the screened training cohort.

    Parameters
    ----------
    data, curve_cache : optional
        A modeling table and curve cache as returned by
        `master_model.load_modeling_data()`.  Loaded automatically when omitted
        (this requires `master_model.py --stage theta-fit` to have been run).
    exclude_conditions : iterable of str, optional
        `condition_id` values ("<material>_<T>_<stress>") to drop from training,
        e.g. to forecast a curve that is itself in the dataset without leakage.
    seed : int
        Seed for the proper / calibration split.
    """

    def __init__(self, data=None, curve_cache=None, exclude_conditions=(), seed=mm.SEED):
        if data is None or curve_cache is None:
            data, curve_cache = mm.load_modeling_data()
        exclude = set(exclude_conditions)
        unknown = exclude - set(data["condition_id"])
        if unknown:
            examples = sorted(data["condition_id"].unique())[:5]
            raise ValueError(
                f"exclude_conditions not found in the cohort: {sorted(unknown)}. "
                f"condition_id looks like {examples}"
            )
        data = data[~data["condition_id"].isin(exclude)].copy()

        self.data = data
        self.curve_cache = {idx: curve_cache[idx] for idx in data.index}
        proper_pos, calibration_pos = mm.stratified_proper_calibration_split(data, seed)
        self.proper = data.iloc[proper_pos].copy()
        self.calibration = data.iloc[calibration_pos].copy()
        self.materials = sorted(data["material"].unique())
        self.heat_treatments = sorted(
            data.loc[data["material"].eq("H282_w"), "heat_treatment"].dropna().unique()
        )

    # ------------------------------------------------------------------
    def condition_only_life(self, material, temperature, stress, heat_treatment=None) -> float:
        """Condition-only rupture-life estimate t_r,cond in hours (Eq. 10)."""
        row = self._new_row(material, temperature, stress, heat_treatment)
        state = mm.fit_state_targets(self.proper, row, mm.CONDITION_FEATURES, mm.SEED + 101)
        return mm.safe_power10(state[0, 0])

    def forecast(self, material, temperature, stress, time, strain,
                 heat_treatment=None, n_grid=301) -> ForecastResult:
        """Forecast the remaining trajectory of one ongoing test.

        Parameters
        ----------
        material : str
            One of the training materials ("A617", "Grade 91", "H282_w").
        temperature : float
            Test temperature in degC.
        stress : float
            Applied (initial engineering) stress in MPa.
        time, strain : array-like
            Measurements so far: time in hours, creep strain in PERCENT.
            Both are re-referenced to the first point (Eq. 1).
        heat_treatment : str, optional
            H282_w heat identifier (e.g. "HT_8354"); ignored for other alloys.
        n_grid : int
            Number of points in the returned forecast curve.
        """
        notes: list[str] = []
        t_obs, e_obs = _prepare_prefix(time, strain)
        t_b = float(t_obs[-1])

        if material not in self.materials:
            notes.append(
                f"Material '{material}' is not in the training set {self.materials}; "
                "the forecast is an unsupported extrapolation. Add completed curves "
                "for this material (see README) before relying on it."
            )
        if len(t_obs) < RECOMMENDED_PREFIX_POINTS:
            notes.append(
                f"Only {len(t_obs)} prefix points; prefix-rate features need at least "
                f"{RECOMMENDED_PREFIX_POINTS} for full use."
            )

        row = self._new_row(material, temperature, stress, heat_treatment)
        new_idx = row.index[0]
        if material == "H282_w" and row.loc[new_idx, "heat_treatment"] not in self.heat_treatments:
            notes.append(
                f"Heat '{row.loc[new_idx, 'heat_treatment']}' is not among the training "
                f"heats {self.heat_treatments}; heat-treatment effects cannot be inferred."
            )

        cache = dict(self.curve_cache)
        cache[new_idx] = {
            "time": t_obs,
            "shifted": e_obs,
            "actual_rupture_time": np.nan,
            "actual_final_shifted_strain": np.nan,
        }

        # Reveal fraction relative to the condition-only life, as in Eq. 10.
        # build_featured uses the same state model and seed, so the training
        # boundary for the new row reproduces t_b exactly.
        life_cond = self.condition_only_life(material, temperature, stress, heat_treatment)
        reveal = t_b / life_cond
        lo, hi = VALIDATED_REVEAL_RANGE
        if reveal >= 1.0:
            notes.append(
                f"The test has already run {reveal:.2f}x its condition-only predicted "
                "life; the bounded update (+/- 1 decade) may be saturated and the "
                "forecast should be treated as unreliable."
            )
        elif not lo <= reveal <= hi:
            notes.append(
                f"Reveal fraction r = {reveal:.3f} is outside the validated range "
                f"[{lo:.2f}, {hi:.2f}]; accuracy has not been benchmarked there."
            )

        proper, calibration = self.proper, self.calibration
        target = pd.concat([calibration, row])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            featured_train, featured_target = mm.build_featured(proper, target, cache, reveal)
            p5_train = mm.inner_oof_p5(proper, featured_train, cache)
            p5_target, _centers, shapes = mm.direct_p5(proper, target, featured_target, cache)
            h3_state, h3_centers = mm.fit_model_H3(
                proper, target, featured_train, featured_target,
                p5_train, p5_target, shapes, cache,
            )

        # Split-conformal calibration on the calibration rows (Section 11).
        boundaries = {idx: float(featured_target.loc[idx, "reveal_time"]) for idx in target.index}
        scores = mm.calibration_scores(calibration.index, h3_centers, cache, boundaries)
        q90 = mm.conformal_quantile(scores, mm.CONFORMAL_COVERAGE)
        q68 = mm.conformal_quantile(scores, mm.CONFORMAL_CENTRAL)
        life_errors = np.abs(
            h3_state.loc[calibration.index, "h3_log_rupture_time"].to_numpy(float)
            - calibration["log_rupture_time"].to_numpy(float)
        )
        q_life = mm.conformal_quantile(life_errors, mm.CONFORMAL_COVERAGE)

        state = h3_state.loc[new_idx]
        log_life = float(state["h3_log_rupture_time"])
        log_final = float(state["h3_log_final_shifted_strain"])
        life = mm.safe_power10(log_life)
        shape = shapes.get(material, shapes["__global__"])

        grid = np.unique(np.r_[np.linspace(0.0, life, n_grid), t_b])
        center = mm.normalized_shape_prediction(grid, shape, log_life, log_final)
        lower_90, upper_90 = mm.conformal_band(center, q90)
        lower_68, upper_68 = mm.conformal_band(center, q68)
        curve = pd.DataFrame({
            "time_h": grid,
            "strain_percent": center,
            "lower_68": lower_68,
            "upper_68": upper_68,
            "lower_90": lower_90,
            "upper_90": upper_90,
            "is_forecast": grid > t_b,
        })

        outside_hull = mm.hull_status(proper, row.loc[new_idx])
        if outside_hull:
            notes.append(
                "The condition lies outside the training convex hull for this material "
                "(standardized log stress vs 1/T); the manuscript reports larger errors "
                "for such extrapolated conditions."
            )
        p5 = p5_target.loc[new_idx]
        if not bool(p5["p5_fit_success"]):
            notes.append("The P5 prefix update did not converge; H3 fell back toward the "
                         "condition-only baseline.")

        summary = {
            "material": material,
            "temperature_C": float(temperature),
            "stress_MPa": float(stress),
            "heat_treatment": row.loc[new_idx, "heat_treatment"],
            "n_prefix_points": int(len(t_obs)),
            "observation_boundary_h": t_b,
            "strain_at_boundary_percent": float(e_obs[-1]),
            "condition_only_life_h": life_cond,
            "reveal_fraction": reveal,
            "predicted_rupture_life_h": life,
            "rupture_life_interval_90_h": [
                mm.safe_power10(log_life - q_life), mm.safe_power10(log_life + q_life)
            ],
            "predicted_remaining_life_h": max(life - t_b, 0.0),
            "predicted_final_strain_percent": mm.safe_power10(log_final),
            "predicted_log_minimum_creep_rate": float(state["h3_log_minimum_creep_rate"]),
            "p5_life_correction_decades": float(p5["p5_life_scale_correction"]),
            "p5_final_strain_correction_decades": float(p5["p5_final_scale_correction"]),
            "h3_life_residual_decades": float(state["h3_life_residual_correction"]),
            "h3_final_residual_decades": float(state["h3_final_residual_correction"]),
            "p5_prefix_fit_nrmse": float(p5["p5_prefix_fit_nrmse"]),
            "conformal_q68": q68,
            "conformal_q90": q90,
            "n_calibration_curves": int(len(scores)),
            "n_training_curves": int(len(proper)),
            "outside_training_hull": bool(outside_hull),
            "warnings": notes,
        }
        observed = pd.DataFrame({"time_h": t_obs, "strain_percent": e_obs})
        return ForecastResult(summary=summary, curve=curve, observed=observed, warnings=notes)

    def forecast_file(self, path, material, temperature, stress, strain_units,
                      heat_treatment=None, time_col=None, strain_col=None, **kwargs):
        """Read a CSV/XLSX record of an ongoing test and forecast it."""
        time, strain = read_test_record(path, strain_units, time_col, strain_col)
        return self.forecast(material, temperature, stress, time, strain,
                             heat_treatment=heat_treatment, **kwargs)

    # ------------------------------------------------------------------
    def _new_row(self, material, temperature, stress, heat_treatment) -> pd.DataFrame:
        temperature, stress = float(temperature), float(stress)
        if material == "H282_w":
            heat = heat_treatment or "unknown"
        else:
            heat = "not_applicable"
        row = {
            "material": material,
            "file": NEW_TEST_FILE,
            "relative_path": NEW_TEST_FILE,
            "_occurrence": 0,
            "temperature": temperature,
            "stress": stress,
            "temperature_K": temperature + 273.15,
            "inv_temperature": 1.0 / (temperature + 273.15),
            "log_stress": float(np.log10(stress)),
            "heat_treatment": heat,
            "condition_id": f"{material}_{temperature}_{round(stress, 6)}",
        }
        return pd.DataFrame([row], index=[int(self.data.index.max()) + 1])


# ----------------------------------------------------------------------
# Input helpers
# ----------------------------------------------------------------------


def read_test_record(path, strain_units, time_col=None, strain_col=None):
    """Read time (h) and strain from a CSV/XLSX file; return strain in percent.

    Column names are matched with the same priority rules used for the
    training data ("time", "Run Time", "strain (%)", "Creep (%)", ...) unless
    `time_col` / `strain_col` are given explicitly.
    """
    df = mm.read_table(Path(path))
    if time_col is None or strain_col is None:
        auto_time, auto_strain, df = mm.choose_time_and_strain_columns(df)
        time_col = time_col or auto_time
        strain_col = strain_col or auto_strain
    time = pd.to_numeric(df[time_col], errors="coerce").to_numpy(float)
    strain = pd.to_numeric(df[strain_col], errors="coerce").to_numpy(float)
    if strain_units == "fraction":
        strain = strain * 100.0
    elif strain_units != "percent":
        raise ValueError("strain_units must be 'percent' or 'fraction'")
    return time, strain


def _prepare_prefix(time, strain):
    frame = pd.DataFrame({"time": np.asarray(time, float),
                          "strain": np.asarray(strain, float)}).dropna()
    frame = frame[frame["time"] >= 0]
    frame = frame.groupby("time", as_index=False)["strain"].mean().sort_values("time")
    if len(frame) < mm.MIN_PREFIX_POINTS:
        raise ValueError(
            f"At least {mm.MIN_PREFIX_POINTS} distinct time points are required; got {len(frame)}"
        )
    time = frame["time"].to_numpy(float)
    strain = frame["strain"].to_numpy(float)
    time, strain = time - time[0], strain - strain[0]
    if time[-1] <= 0:
        raise ValueError("Observed time span is zero")
    if np.nanmax(np.abs(strain)) < 1e-6:
        raise ValueError("Observed strain does not change; check units and columns")
    return time, strain


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def format_summary(summary: dict) -> str:
    lo, hi = summary["rupture_life_interval_90_h"]
    lines = [
        f"{summary['material']}  {summary['temperature_C']:g} degC  {summary['stress_MPa']:g} MPa",
        f"  observed to            {summary['observation_boundary_h']:.1f} h "
        f"({summary['n_prefix_points']} points, strain {summary['strain_at_boundary_percent']:.3f} %)",
        f"  condition-only life    {summary['condition_only_life_h']:.1f} h  "
        f"-> reveal fraction r = {summary['reveal_fraction']:.3f}",
        f"  H3 rupture life        {summary['predicted_rupture_life_h']:.1f} h  "
        f"(90% interval {lo:.1f} - {hi:.1f} h)",
        f"  remaining life         {summary['predicted_remaining_life_h']:.1f} h",
        f"  H3 terminal strain     {summary['predicted_final_strain_percent']:.2f} %",
        f"  prefix correction      life {summary['p5_life_correction_decades']:+.3f}, "
        f"strain {summary['p5_final_strain_correction_decades']:+.3f} decades (P5)",
        f"  outside training hull  {summary['outside_training_hull']}",
    ]
    for note in summary["warnings"]:
        lines.append(f"  WARNING: {note}")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Forecast the remaining creep curve of an ongoing test with model H3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--curve", required=True, type=Path,
                        help="CSV/XLSX with the measurements so far (time in hours).")
    parser.add_argument("--material", required=True,
                        help='Training material: "A617", "Grade 91" or "H282_w".')
    parser.add_argument("--temperature", required=True, type=float, help="Temperature, degC.")
    parser.add_argument("--stress", required=True, type=float, help="Applied stress, MPa.")
    parser.add_argument("--strain-units", required=True, choices=["percent", "fraction"],
                        help="Units of the strain column in --curve.")
    parser.add_argument("--heat", default=None, help="H282_w heat identifier, e.g. HT_8354.")
    parser.add_argument("--time-col", default=None, help="Time column name (auto-detected).")
    parser.add_argument("--strain-col", default=None, help="Strain column name (auto-detected).")
    parser.add_argument("--out", type=Path, default=Path("forecast_output"),
                        help="Directory for forecast_curve.csv, forecast_summary.json, forecast.png.")
    parser.add_argument("--exclude-condition", action="append", default=[],
                        help='Drop a training condition, e.g. "A617_750_121.0" '
                             "(repeatable). Use when the test is itself in the dataset.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Override the training data root.")
    parser.add_argument("--model-output-dir", type=Path, default=None,
                        help="Directory holding theta_projection_summary.xlsx.")
    return parser.parse_args()


def main():
    args = parse_args()
    mm.set_paths(args.data_dir, args.model_output_dir)
    forecaster = CreepForecaster(exclude_conditions=args.exclude_condition)
    result = forecaster.forecast_file(
        args.curve, args.material, args.temperature, args.stress, args.strain_units,
        heat_treatment=args.heat, time_col=args.time_col, strain_col=args.strain_col,
    )
    print(format_summary(result.summary))
    print(f"\nWrote {result.save(args.out)}")


if __name__ == "__main__":
    main()
