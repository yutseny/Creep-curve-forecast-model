"""master_model.py - consolidated model and data-processing pipeline.

Paper
-----
"Early creep deformation constrains full-trajectory forecasting in
high-temperature alloys" (Yi, Zhang, Messner, Chen, Rollett).

Purpose
-------
The analysis for that manuscript was carried out across ~160 exploratory
scripts and notebooks in `new fit/` and the dated revision folders, chained
together at run time by `importlib` module loading.  This file collapses the
path that actually produced the published numbers into one self-contained
module, with no cross-script loading and no hidden dependency chain.

The original chain that this file replaces was:

    run_theta_h3_tertiary_test.py            (T0/T1 theta, scoring)
      -> run_2026_07_30_revision.py          (reveal sweep, hull support)
      -> run_locked_p5_m2_h3_benchmark.py    (P5, M2, H3)
        -> run_locked_p1_p2_benchmark.py     (Arrhenius power laws, exclusions)
        -> final_hybrid_nested_cv.py         (fold-safe priors)
          -> model_v1/final_hybrid_v1.py     (frozen V1 wiring)
            -> theta_center_uncertainty.py   (condition-only center = model M)
            -> refined_early_forecast_model.py  (smoothed prefix features)
              -> omega_material_early_forecast.py (Omega prefix features)
                -> normalized_shape_curve_model.py (S(x) normalization)
                  -> model_endpoint_scaling_and_physics.py (raw curve I/O)
    theta_fit_all_v2.ipynb                   (theta-projection fitting)
    theta_ML_derived.ipynb                   (R^2 >= 0.95 quality screen)

Layout
------
Sections follow the manuscript's Methods section:

    1  Configuration and constants
    2  Raw creep-record I/O and strain shifting        (Methods 2.1)
    3  Theta-projection fitting and quality screening  (Methods 2.2)
    4  Endpoint / shape decomposition                  (Methods 2.3)
    5  Condition-only state models and fold-safe priors
    6  Observed-prefix feature extraction
    7  Bounded physics prefix update (P5)
    8  Forecasting models M, M2, H3, T0, T1            (Methods 2.4)
    9  Cross-validation protocol                       (Methods 2.5)
    10 Scoring, bootstrap intervals, condition support
    11 Split-conformal predictive intervals
    12 Orchestration

Reproducibility
---------------
Seeds, feature lists, model hyperparameters, bounds and clipping limits are
carried over verbatim from the locked scripts so that this module reproduces
the published centers rather than merely re-implementing the method.  The
constants that matter are collected in Section 1 and annotated with the file
they came from.

Usage
-----
    python master_model.py --stage all
    python master_model.py --stage theta-fit     # Section 3 only
    python master_model.py --stage forecast      # Sections 5-10
    python master_model.py --reveals 0.20 0.30
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, lsq_linear
from scipy.spatial import Delaunay, QhullError
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from scipy.signal import savgol_filter
except ImportError:  # pragma: no cover - SciPy is a hard dependency in practice
    savgol_filter = None

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ==========================================================================
# 1  Configuration and constants
# ==========================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "new fit" / "data"
OUTPUT_DIR = BASE_DIR / "master_model_output"

# Theta fits produced by Section 3; consumed by Section 4 onwards.
THETA_SUMMARY_FILE = OUTPUT_DIR / "theta_projection_summary.xlsx"

# Material folders under DATA_DIR.  "HT" is the wrought H282 record set and is
# renamed to H282_w on load, matching the manuscript's material naming.
MATERIAL_FOLDERS = ["A617", "Grade 91", "HT"]
MATERIAL_RENAME = {"HT": "H282_w"}

EPS = 1e-12

# --- Cohort definition (Methods 2.2) --------------------------------------
# 333 assembled -> 325 at theta-fit R^2 >= 0.95 -> 323 after two locked
# bad-record exclusions.  The threshold is from theta_ML_derived.ipynb; the two
# exclusions are from run_locked_p1_p2_benchmark.py (EXCLUDED).
MIN_FIT_R2 = 0.95
EXCLUDED_RECORDS = {"HT/HT_8354/760_201.xlsx", "HT/HT_8389/593_624.xlsx"}

# --- Normalized-shape grid (Methods 2.3) ----------------------------------
# 201-point linear grid in normalized time x = t / t_r.
GRID = np.linspace(0.0, 1.0, 201)
MAX_EVAL_POINTS = 201

# --- Validation protocol (Methods 2.5) ------------------------------------
N_SPLITS = 5
SPLIT_SCHEME = "grouped_temperature_stress_5fold"
CALIBRATION_FRACTION = 0.30          # theta_postconstraint_conformal_v2.py
REVEALS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
PRIMARY_REVEALS = (0.20, 0.30)
MIN_PREFIX_POINTS = 5
MIN_SUFFIX_POINTS = 3

# --- Seeds (verbatim from the locked scripts) -----------------------------
SEED = 2718                          # final_hybrid_v1.py / run_locked_p1_p2
THETA_SEED = 20260814                # run_theta_h3_tertiary_test.py
CENTER_RANDOM_STATE = 1709           # theta_center_uncertainty.py
FROZEN_EXPORT_SCHEME_NUMBER = 1      # run_locked_p1_p2_benchmark.py
N_BOOT = 1000

# --- Condition-only state model (Section 5) -------------------------------
# theta_center_uncertainty.py: FEATURES / TARGETS / TARGET_MODELS / N_TREES.
CONDITION_FEATURES = [
    "material",
    "temperature",
    "stress",
    "log_stress",
    "inv_temperature",
]
STATE_TARGETS = [
    "log_rupture_time",
    "log_beta2_normalized",
    "log_beta4_normalized",
    "log_primary_final_percent",
    "log_tertiary_final_percent",
]
STATE_TARGET_MODELS = {
    "log_rupture_time": "gaussian_process",
    "log_beta2_normalized": "random_forest",
    "log_beta4_normalized": "random_forest",
    "log_primary_final_percent": "random_forest",
    "log_tertiary_final_percent": "extra_trees",
}
N_TREES = 300

# --- Fold-safe prior model (Section 5) ------------------------------------
# final_hybrid_nested_cv.py: FEATURES / TARGET_MODELS.
PRIOR_FEATURES = [
    "temperature",
    "stress",
    "temperature_K",
    "inv_temperature",
    "log_stress",
    "material",
    "heat_treatment",
]
PRIOR_TARGET_MODELS = {
    "log_beta2_normalized": "RandomForest",
    "log_beta4_normalized": "RandomForest",
    "log_primary_final_percent": "RandomForest",
    "log_tertiary_final_percent": "ExtraTrees",
    "log_rupture_time": "ExtraTrees",
    "log_rupture_strain_percent": "RandomForest",
}
PRIOR_TARGETS = list(PRIOR_TARGET_MODELS)
KEYS = ["material", "file", "_occurrence"]

# --- Prefix feature blocks (Section 6) ------------------------------------
# omega_material_early_forecast.py: BASE_FEATURES / OMEGA_FEATURES.
BASE_FEATURES = [
    "temperature",
    "stress",
    "log_stress",
    "inv_temperature",
    "previous_log_rupture_time",
    "previous_log_final_shifted_strain",
    "early_log_time",
    "early_log_strain",
    "early_log_secant_rate",
    "early_log_tail_rate",
    "early_strain_over_prior_final",
    "early_time_over_prior_time",
]
OMEGA_FEATURES = [
    *BASE_FEATURES,
    "omega_log_rate_vs_strain",
    "omega_log_rate_vs_time",
    "early_log_rate_start",
    "early_log_rate_end",
    "early_log_rate_ratio",
    "early_rate_curvature",
    "early_strain_fraction_slope",
]
# refined_early_forecast_model.py: SMOOTH_FEATURES.  This is the M2 feature set.
SMOOTH_FEATURES = [
    *OMEGA_FEATURES,
    "smooth_log_median_rate",
    "smooth_log_rate_cv",
    "smooth_log_rate_iqr",
    "smooth_log_accel_ratio",
    "smooth_omega_log_rate_vs_strain",
    "smooth_late_rate_slope",
    "smooth_strain_curvature",
    "early_fraction_of_prior_rupture_strain",
]
# run_locked_p5_m2_h3_benchmark.py: P5_FEATURES; H3 = SMOOTH_FEATURES + P5.
P5_FEATURES = [
    "p5_log_rupture_time",
    "p5_log_final_shifted_strain",
    "p5_life_scale_correction",
    "p5_final_scale_correction",
    "p5_prefix_fit_nrmse",
]
H3_FEATURES = [*SMOOTH_FEATURES, *P5_FEATURES]

# --- Bounded-update limits (Methods 2.4, "bounded in logarithmic space") ---
P5_DELTA_BOUND = 1.0        # least_squares bounds on (dlog t_r, dlog eps_f)
P5_PRIOR_WEIGHT = 0.15      # weak prior pulling delta toward zero
P5_PRIOR_SCALE = 0.50
H3_CORRECTION_BOUND = 1.0   # residual ML corrections clipped to +/- 1 decade

# --- Conformal calibration (Methods 2.5) ----------------------------------
CONFORMAL_COVERAGE = 0.90
CONFORMAL_CENTRAL = 0.68


# ==========================================================================
# 2  Raw creep-record I/O and strain shifting            (Methods 2.1)
# ==========================================================================
#
# Source: model_endpoint_scaling_and_physics.py::read_raw_curve, plus the
# column-selection priority from theta_fit_all_v2.ipynb.  Strain is referenced
# to the first recorded creep measurement, giving Eq. 1 of the manuscript:
#
#     eps(t) = eps_raw(t) - eps_raw(t_0)

RAW_CURVE_CACHE: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

TIME_COLUMN_PRIORITY = [
    lambda c: "run time" in c,
    lambda c: "e.t." in c and "hour" in c,
    lambda c: "elapsed" in c and "hour" in c,
    lambda c: c == "time",
    lambda c: "time" in c,
    lambda c: "hour" in c,
]
STRAIN_COLUMN_PRIORITY = [
    lambda c: "creep (%)" in c,
    lambda c: "strain (%)" in c,
    lambda c: "ave strain" in c,
    lambda c: "average strain" in c,
    lambda c: c == "strain",
    lambda c: "strain" in c,
    lambda c: "creep" in c,
]


def safe_log10(values):
    values = pd.to_numeric(values, errors="coerce")
    return np.log10(values.where(values > 0))


def safe_power10(value, lo=-12.0, hi=12.0):
    return float(10.0 ** float(np.clip(value, lo, hi)))


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported creep-record format: {path.suffix}")


def choose_time_and_strain_columns(df: pd.DataFrame) -> tuple[str, str, pd.DataFrame]:
    """Pick the time and strain columns by documented name priority.

    Experimental programs label these columns inconsistently ("Run Time",
    "E.T. (hours)", "Creep (%)", "Ave Strain", ...), so the priority lists are
    tried in order and a numeric-density fallback is used only if both fail.
    """
    columns = [str(column).strip() for column in df.columns]
    lower = [column.lower() for column in columns]
    df = df.copy()
    df.columns = columns

    def first_match(priority):
        for matcher in priority:
            for original, value in zip(columns, lower):
                if matcher(value):
                    return original
        return None

    time_col = first_match(TIME_COLUMN_PRIORITY)
    strain_col = first_match(STRAIN_COLUMN_PRIORITY)
    if time_col is None or strain_col is None:
        numeric = [
            column
            for column in columns
            if pd.to_numeric(df[column], errors="coerce").notna().sum() >= 10
        ]
        if len(numeric) < 2:
            raise ValueError("Could not identify time and strain columns")
        time_col = time_col or numeric[0]
        strain_col = strain_col or numeric[1]
    return time_col, strain_col, df


def read_raw_curve(relative_path: str, material: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (time in hours from the first record, shifted strain in percent).

    A617 and Grade 91 source files store fractional strain and are converted to
    percent; H282 records already use the percent convention.  The magnitude
    test catches mixed-convention files in either set.
    """
    key = (str(relative_path), str(material))
    if key in RAW_CURVE_CACHE:
        time, shifted = RAW_CURVE_CACHE[key]
        return time.copy(), shifted.copy()

    df = read_table(DATA_DIR / str(relative_path))
    time_col, strain_col, df = choose_time_and_strain_columns(df)
    clean = pd.DataFrame(
        {
            "time": pd.to_numeric(df[time_col], errors="coerce"),
            "strain": pd.to_numeric(df[strain_col], errors="coerce"),
        }
    ).dropna()
    clean = clean[clean["time"] >= 0].sort_values("time")
    clean = clean.groupby("time", as_index=False)["strain"].mean()

    time = clean["time"].to_numpy(float)
    time = time - np.nanmin(time)
    strain = clean["strain"].to_numpy(float)
    if material in {"A617", "Grade 91"} or np.nanmax(np.abs(strain)) <= 1.0:
        strain = strain * 100.0
    shifted = strain - strain[0]

    RAW_CURVE_CACHE[key] = (time.copy(), shifted.copy())
    return time, shifted


def parse_temperature_stress(filename: str) -> tuple[float, float]:
    """Record filenames encode the condition as '<temperature>_<stress>'."""
    numbers = re.findall(r"\d+\.?\d*", Path(filename).stem)
    if len(numbers) < 2:
        return np.nan, np.nan
    return float(numbers[0]), float(numbers[1])


def discover_records() -> pd.DataFrame:
    """Enumerate every creep record under the three material folders."""
    rows = []
    for folder in MATERIAL_FOLDERS:
        root = DATA_DIR / folder
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
                continue
            if path.name.startswith("~$"):
                continue
            relative = path.relative_to(DATA_DIR).as_posix()
            temperature, stress = parse_temperature_stress(path.name)
            # Record filenames encode the condition as '<temperature>_<stress>'.
            # Derived summary workbooks written alongside the records
            # (kachanov_fast_summary.xlsx, creep_fit_summary.xlsx, ...) carry no
            # such pair and are not creep records.
            if not (np.isfinite(temperature) and np.isfinite(stress)):
                continue
            rows.append(
                {
                    "material": MATERIAL_RENAME.get(folder, folder),
                    "source_folder": folder,
                    "file": path.name,
                    "relative_path": relative,
                    "temperature": temperature,
                    "stress": stress,
                }
            )
    return pd.DataFrame(rows)


def add_occurrence_index(df: pd.DataFrame) -> pd.DataFrame:
    """Disambiguate replicate records that share a material and filename."""
    out = df.copy()
    out["_occurrence"] = out.groupby(["material", "file"]).cumcount()
    return out


def prepare_curve(time, strain_percent) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame({"time": time, "strain": strain_percent}).dropna()
    frame = frame.groupby("time", as_index=False)["strain"].mean().sort_values("time")
    if len(frame) < 10:
        raise ValueError("Fewer than 10 unique time points")
    time = frame["time"].to_numpy(float)
    time = time - np.nanmin(time)
    if np.nanmax(time) <= 0:
        raise ValueError("Rupture time is not positive")
    return time, frame["strain"].to_numpy(float)


def downsample(time, values, max_points):
    if len(time) <= max_points:
        return time, values
    keep = np.unique(np.linspace(0, len(time) - 1, max_points).astype(int))
    return time[keep], values[keep]


def downsample_curve_cache(curve_cache: dict, max_points=MAX_EVAL_POINTS) -> dict:
    """Cap evaluation points per curve so densely sampled tests do not dominate.

    Sampling density ranges from a few dozen to tens of thousands of points
    across the three experimental programs (Methods 2.1).
    """
    result = {}
    for key, cache in curve_cache.items():
        if len(cache["time"]) <= max_points:
            result[key] = cache
            continue
        keep = np.unique(np.linspace(0, len(cache["time"]) - 1, max_points).astype(int))
        result[key] = {**cache, "time": cache["time"][keep], "shifted": cache["shifted"][keep]}
    return result


# ==========================================================================
# 3  Theta-projection fitting and quality screening      (Methods 2.2)
# ==========================================================================
#
# Source: theta_fit_all_v2.ipynb.  The four-parameter theta model (Eq. 2) is
# fitted in normalized time so that the rate parameters become the
# dimensionless beta2 = theta2 * t_r and beta4 = theta4 * t_r of Eq. 3.


def theta_model_normalized(x, theta1, beta2, theta3, beta4):
    """Theta model in normalized time x = t / t_r (Eq. 2 with Eq. 3 rates)."""
    x = np.asarray(x, float)
    return theta1 * (1.0 - np.exp(-beta2 * x)) + theta3 * np.expm1(beta4 * x)


def theta_model_hours(time, theta1, theta2, theta3, theta4):
    """Theta model in dimensional time (Eq. 2 as published)."""
    time = np.asarray(time, float)
    return theta1 * (1.0 - np.exp(-theta2 * time)) + theta3 * np.expm1(theta4 * time)


def _safe_positive(value, fallback):
    return float(value) if np.isfinite(value) and value > 0 else float(fallback)


def estimate_stage_guesses(x, shifted_strain):
    """Staged multi-start guesses for [theta1, beta2, theta3, beta4].

    The primary and tertiary terms trade off strongly, so a single start point
    routinely converges to a degenerate fit with one stage switched off.  These
    guesses are seeded from the early, mid and late portions of the curve.
    """
    strain_range = float(max(np.nanmax(shifted_strain) - np.nanmin(shifted_strain), 1e-8))
    final_strain = _safe_positive(shifted_strain[-1], strain_range)

    early_mask = x <= 0.25
    mid_mask = (x >= 0.15) & (x <= 0.45)
    late_mask = x >= 0.75

    early_level = _safe_positive(
        np.nanpercentile(shifted_strain[early_mask], 80)
        if early_mask.sum() >= 5
        else 0.25 * strain_range,
        0.25 * strain_range,
    )
    mid_level = _safe_positive(
        np.nanmedian(shifted_strain[mid_mask]) if mid_mask.sum() >= 5 else 0.4 * strain_range,
        0.4 * strain_range,
    )
    theta1_base = max(early_level, 0.25 * mid_level, 0.05 * strain_range)

    beta2_candidates = [0.3, 1.0, 3.0, 8.0, 30.0, 100.0, 1000.0]
    if early_mask.sum() >= 5:
        ratio = np.clip(early_level / max(theta1_base * 1.5, 1e-8), 1e-6, 0.95)
        x_early = _safe_positive(np.nanmedian(x[early_mask]), 0.15)
        beta2_candidates.insert(0, _safe_positive(-np.log(1.0 - ratio) / x_early, 1.0))

    late_gain = _safe_positive(final_strain - mid_level, 0.1 * strain_range)
    beta4_candidates = [0.3, 1.0, 3.0, 8.0, 16.0, 50.0]
    if late_mask.sum() >= 8:
        late_y = shifted_strain[late_mask]
        positive = late_y - np.nanmin(late_y) + 1e-8
        if np.isfinite(positive).sum() >= 8 and np.nanmax(positive) > np.nanmin(positive):
            slope = np.polyfit(x[late_mask], np.log(np.maximum(positive, 1e-8)), 1)[0]
            beta4_candidates.insert(0, float(np.clip(slope, 0.1, 25.0)))

    starts = []
    for theta1_factor in (0.5, 1.0, 2.0):
        theta1 = max(theta1_base * theta1_factor, 1e-8)
        for beta2 in beta2_candidates[:5]:
            for beta4 in beta4_candidates[:3]:
                theta3 = max(late_gain / max(np.expm1(beta4), 1e-8), 1e-8)
                starts.append([theta1, beta2, theta3, beta4])
    starts.extend(
        [
            [0.3 * strain_range, 1.0, 0.01 * strain_range, 1.0],
            [0.1 * strain_range, 3.0, 0.05 * strain_range, 3.0],
            [0.7 * strain_range, 0.5, 0.001 * strain_range, 8.0],
            [0.5 * strain_range, 100.0, 0.01 * strain_range, 1.0],
            [0.5 * strain_range, 1000.0, 0.01 * strain_range, 1.0],
        ]
    )

    unique, seen = [], set()
    for start in starts:
        key = tuple(np.round(np.log10(np.maximum(start, 1e-12)), 3))
        if key not in seen:
            seen.add(key)
            unique.append(start)
    return unique[:28]


def normalized_theta_residual(log_params, x, shifted_strain, scale, regularization_weight=0.002):
    """Scaled residual with a weak penalty against degenerate parameter scales."""
    theta1, beta2, theta3, beta4 = 10.0 ** np.asarray(log_params)
    prediction = theta_model_normalized(x, theta1, beta2, theta3, beta4)
    data_residual = (prediction - shifted_strain) / scale

    final_primary = theta1 * (1.0 - np.exp(-beta2))
    final_tertiary = theta3 * np.expm1(beta4)
    total = max(final_primary + final_tertiary, 1e-12)
    penalties = regularization_weight * np.array(
        [
            max(0.0, -8.0 - np.log10(theta1)),
            max(0.0, -8.0 - np.log10(theta3)),
            max(0.0, 0.02 - final_primary / total),
            max(0.0, 0.02 - final_tertiary / total),
        ]
    )
    return np.r_[data_residual, penalties]


def estimate_tc_from_strain_rate(time, strain_percent, threshold_multiplier=1.5):
    """Experimental tertiary onset: first time the rate exceeds 1.5x the minimum."""
    if len(time) < 15:
        return np.nan, np.nan, threshold_multiplier
    order = np.argsort(time)
    time = np.asarray(time, float)[order]
    strain_percent = np.asarray(strain_percent, float)[order]
    unique = (
        pd.DataFrame({"time": time, "strain": strain_percent})
        .groupby("time", as_index=False)
        .mean()
    )
    time = unique["time"].to_numpy(float)
    strain_percent = unique["strain"].to_numpy(float)
    if len(time) < 15:
        return np.nan, np.nan, threshold_multiplier

    window = min(31, len(strain_percent) // 2 * 2 - 1)
    smooth = (
        savgol_filter(strain_percent, window_length=window, polyorder=3)
        if window >= 7 and savgol_filter is not None
        else strain_percent.copy()
    )
    rate = np.gradient(smooth, time)
    valid = np.isfinite(rate) & (rate > 0)
    if valid.sum() < 5:
        return np.nan, np.nan, threshold_multiplier

    valid_indices = np.flatnonzero(valid)
    min_idx = valid_indices[np.argmin(rate[valid])]
    minimum_creep_rate = rate[min_idx]
    threshold = threshold_multiplier * minimum_creep_rate
    tc = np.nan
    for index in range(min_idx + 1, len(time)):
        if np.isfinite(rate[index]) and rate[index] > threshold:
            tc = time[index]
            break
    return tc, minimum_creep_rate, threshold_multiplier


def fit_theta_curve(time, strain_percent, max_fit_points=400) -> dict:
    """Fit one completed creep curve and return theta and endpoint quantities.

    Fitting is done on log10 parameters in normalized time with a soft_l1 loss
    over multiple starts, then polished with a plain least-squares pass.  The
    returned `fit_identifiable` flag records whether both stages are actually
    active and away from their bounds; it is diagnostic only and is not used as
    a cohort filter (the cohort filter is R^2 >= 0.95, applied downstream).
    """
    time, strain_percent = prepare_curve(time, strain_percent)
    rupture_time = float(np.nanmax(time))
    rupture_strain_percent = float(strain_percent[-1])
    initial_strain_percent = float(strain_percent[0])
    shifted_strain = strain_percent - initial_strain_percent
    strain_range = float(max(np.nanmax(shifted_strain) - np.nanmin(shifted_strain), 1e-8))
    scale = max(strain_range, 1e-6)

    fit_time, fit_strain = downsample(time, strain_percent, max_fit_points)
    fit_x = fit_time / rupture_time
    fit_shifted = fit_strain - initial_strain_percent

    lower_log = np.log10([1e-10, 1e-6, 1e-10, 1e-6])
    upper_log = np.log10([max(100.0 * strain_range, 10.0), 5000.0,
                          max(100.0 * strain_range, 10.0), 100.0])

    best_result, best_cost = None, np.inf
    for start in estimate_stage_guesses(fit_x, fit_shifted):
        x0 = np.clip(np.log10(np.maximum(start, 1e-12)), lower_log, upper_log)
        result = least_squares(
            normalized_theta_residual,
            x0=x0,
            bounds=(lower_log, upper_log),
            args=(fit_x, fit_shifted, scale),
            loss="soft_l1",
            f_scale=0.1,
            max_nfev=2500,
        )
        data_residual = normalized_theta_residual(
            result.x, fit_x, fit_shifted, scale, regularization_weight=0.0
        )[: len(fit_x)]
        cost = float(np.sum(data_residual**2))
        if cost < best_cost:
            best_cost, best_result = cost, result

    if best_result is None or not best_result.success:
        raise RuntimeError(
            "theta least_squares did not converge"
            if best_result is None
            else best_result.message
        )

    polished = least_squares(
        normalized_theta_residual,
        x0=best_result.x,
        bounds=(lower_log, upper_log),
        args=(fit_x, fit_shifted, scale, 0.0),
        loss="linear",
        max_nfev=3000,
    )
    if polished.success:
        best_result = polished

    theta1, beta2, theta3, beta4 = 10.0 ** best_result.x
    theta2 = beta2 / rupture_time
    theta4 = beta4 / rupture_time

    x_full = time / rupture_time
    fitted_percent = theta_model_normalized(x_full, theta1, beta2, theta3, beta4) + initial_strain_percent
    r2 = r2_score(strain_percent, fitted_percent)

    # Endpoint strain contributions at rupture (Eq. 4).
    final_primary = float(theta1 * (1.0 - np.exp(-beta2)))
    final_tertiary = float(theta3 * np.expm1(beta4))
    final_total = max(final_primary + final_tertiary, 1e-12)
    primary_fraction_final = final_primary / final_total
    tertiary_fraction_final = final_tertiary / final_total

    log_values = np.log10([theta1, theta2, theta3, theta4])
    fit_identifiable = bool(
        not (best_result.x - lower_log <= 0.05).any()
        and primary_fraction_final >= 0.03
        and tertiary_fraction_final >= 0.03
        and not ((log_values < -10).any() or (log_values > 6).any())
    )

    tc, minimum_creep_rate, tc_threshold = estimate_tc_from_strain_rate(time, strain_percent)

    return {
        "theta1": theta1,
        "theta2": theta2,
        "theta3": theta3,
        "theta4": theta4,
        "beta2_normalized": beta2,
        "beta4_normalized": beta4,
        "log_theta1": float(np.log10(theta1)),
        "log_theta2": float(np.log10(theta2)),
        "log_theta3": float(np.log10(theta3)),
        "log_theta4": float(np.log10(theta4)),
        "initial_strain_percent": initial_strain_percent,
        "rupture_time": rupture_time,
        "rupture_strain_percent": rupture_strain_percent,
        "primary_final_percent": final_primary,
        "tertiary_final_percent": final_tertiary,
        "modeled_final_strain_percent": final_primary + final_tertiary,
        "primary_fraction_final": primary_fraction_final,
        "tertiary_fraction_final": tertiary_fraction_final,
        "minimum_creep_rate_percent_per_hour": minimum_creep_rate,
        "tertiary_onset_time": tc,
        "tc_threshold_multiplier": tc_threshold,
        "r2": float(r2),
        "n_points": int(len(time)),
        "fit_identifiable": fit_identifiable,
    }


def fit_all_theta_curves(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every assembled record; return (summary, failures)."""
    rows, failures = [], []
    for _, record in records.iterrows():
        try:
            time, shifted = read_raw_curve(record["relative_path"], record["material"])
            # Fitting works on the record's own percent strain; the theta fit
            # re-derives its own offset, so the shifted series is sufficient.
            result = fit_theta_curve(time, shifted)
            rows.append({**record.to_dict(), **result})
        except Exception as exc:
            failures.append({**record.to_dict(), "error": repr(exc)})
    return pd.DataFrame(rows), pd.DataFrame(failures)


def apply_quality_screen(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the R^2 >= 0.95 representation screen and the two locked exclusions.

    This is a retrospective screen on completed-curve fits.  Neither the theta
    parameters nor the measured endpoints used here are supplied to any
    prospective forecast (Methods 2.2).
    """
    frame = summary.copy()
    frame["r2"] = pd.to_numeric(frame["r2"], errors="coerce")
    frame["relative_path"] = frame["relative_path"].astype(str).str.replace("\\", "/", regex=False)

    frame["excluded_reason"] = ""
    frame.loc[frame["r2"] < MIN_FIT_R2, "excluded_reason"] = f"theta_fit_r2_below_{MIN_FIT_R2}"
    frame.loc[frame["relative_path"].isin(EXCLUDED_RECORDS), "excluded_reason"] = "locked_bad_record"

    retained = frame[frame["excluded_reason"].eq("")].copy()
    excluded = frame[frame["excluded_reason"].ne("")].copy()
    return retained.drop(columns=["excluded_reason"]), excluded


def build_modeling_table(retained: pd.DataFrame) -> pd.DataFrame:
    """Derive the log-space targets and condition descriptors used downstream.

    Source: theta_ML_derived.ipynb and
    model_endpoint_scaling_and_physics.py::load_data.
    """
    data = add_occurrence_index(retained.copy())

    for column in [
        "temperature",
        "stress",
        "rupture_time",
        "rupture_strain_percent",
        "primary_final_percent",
        "tertiary_final_percent",
        "minimum_creep_rate_percent_per_hour",
        "beta2_normalized",
        "beta4_normalized",
    ]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["temperature_K"] = data["temperature"] + 273.15
    data["inv_temperature"] = 1.0 / data["temperature_K"]
    data["log_stress"] = safe_log10(data["stress"])
    data["log_rupture_time"] = safe_log10(data["rupture_time"])
    data["log_rupture_strain_percent"] = safe_log10(data["rupture_strain_percent"])

    # Terminal shifted strain (Eq. 6): eps_f = eps_p,f + eps_t,f
    data["final_shifted_strain_percent"] = (
        data["primary_final_percent"] + data["tertiary_final_percent"]
    )
    data["log_final_shifted_strain_percent"] = safe_log10(data["final_shifted_strain_percent"])
    data["log_final_shifted_strain"] = data["log_final_shifted_strain_percent"]
    data["log_beta2_normalized"] = safe_log10(data["beta2_normalized"])
    data["log_beta4_normalized"] = safe_log10(data["beta4_normalized"])
    data["log_primary_final_percent"] = safe_log10(data["primary_final_percent"])
    data["log_tertiary_final_percent"] = safe_log10(data["tertiary_final_percent"])
    data["log_minimum_creep_rate"] = safe_log10(data["minimum_creep_rate_percent_per_hour"])

    # H282 records carry a heat-treatment folder identifier used categorically.
    data["heat_treatment"] = np.where(
        data["material"].eq("H282_w"),
        data["relative_path"].astype(str).str.extract(r"HT[\\/]([^\\/]+)", expand=False),
        "not_applicable",
    )

    # Condition identity groups replicates for leakage-safe validation.
    data["condition_id"] = (
        data["material"].astype(str)
        + "_"
        + data["temperature"].astype(str)
        + "_"
        + data["stress"].round(6).astype(str)
    )

    data = data.replace([np.inf, -np.inf], np.nan)
    return data.loc[data[STATE_TARGETS].notna().all(axis=1)].copy()


# ==========================================================================
# 4  Endpoint / shape decomposition                      (Methods 2.3)
# ==========================================================================
#
# Source: normalized_shape_curve_model.py and
# model_endpoint_scaling_and_physics.py::derived_curve.
#
# Each trajectory is represented by rupture life t_r, terminal strain eps_f and
# a normalized shape S(x) on the 201-point grid, so that
#
#     eps_hat(t) = eps_f_hat * S_hat(t / t_r_hat)      (Eq. 9)


def derived_curve(time, rupture_time, beta2, beta4, primary_final, tertiary_final):
    """Endpoint-normalized theta curve (Eq. 5), evaluated at `time`.

    The beta4 >= 50 branch avoids overflow in expm1 by rewriting the tertiary
    fraction about x = 1, where it is numerically well behaved.
    """
    rupture_time = max(float(rupture_time), EPS)
    beta2 = max(float(beta2), EPS)
    beta4 = max(float(beta4), EPS)
    primary_final = max(float(primary_final), EPS)
    tertiary_final = max(float(tertiary_final), EPS)

    x = np.clip(np.asarray(time, float) / rupture_time, 0.0, 1.0)
    primary = primary_final * (1.0 - np.exp(-beta2 * x)) / max(1.0 - np.exp(-beta2), EPS)
    if beta4 < 50.0:
        tertiary_fraction = np.expm1(beta4 * x) / max(np.expm1(beta4), EPS)
    else:
        tertiary_fraction = np.exp(np.clip(beta4 * (x - 1.0), -700.0, 0.0))
        tertiary_fraction[x <= 0.0] = 0.0
    return primary + tertiary_final * tertiary_fraction


def normalized_curve_for_row(row) -> dict:
    """Interpolate one measured curve onto the normalized shape grid (Eq. 8).

    The shape is forced monotone and endpoint-normalized so that S(0) = 0 and
    S(1) = 1 by construction.
    """
    time, shifted = read_raw_curve(row["relative_path"], row["material"])
    rupture_time = float(row["rupture_time"])
    final_strain = float(shifted[-1])
    if not np.isfinite(rupture_time) or rupture_time <= 0:
        rupture_time = float(time[-1])
    if not np.isfinite(final_strain) or abs(final_strain) < EPS:
        final_strain = float(row["final_shifted_strain_percent"])
    if not np.isfinite(final_strain) or abs(final_strain) < EPS:
        raise ValueError("Invalid final shifted strain for normalization")

    x = np.clip(time / rupture_time, 0.0, 1.0)
    y = shifted / final_strain
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 5:
        raise ValueError("Too few normalized points")

    order = np.argsort(x)
    x, y = x[order], y[order]
    x = np.r_[0.0, x, 1.0]
    y = np.r_[0.0, y, 1.0]
    unique_x, inverse = np.unique(x, return_inverse=True)
    mean_y = np.array([np.nanmean(y[inverse == i]) for i in range(len(unique_x))])

    shape = np.interp(GRID, unique_x, mean_y)
    shape[0], shape[-1] = 0.0, 1.0
    shape = np.maximum.accumulate(np.clip(shape, -0.25, 1.75))
    if shape[-1] != 0:
        shape = shape / shape[-1]

    return {
        "time": time,
        "shifted": shifted,
        "actual_rupture_time": rupture_time,
        "actual_final_shifted_strain": final_strain,
        "shape": shape,
    }


def load_curve_cache(data: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    records, failed = {}, []
    for idx, row in data.iterrows():
        try:
            records[idx] = normalized_curve_for_row(row)
        except Exception as exc:
            failed.append({"index": idx, "material": row.get("material"),
                           "file": row.get("file"), "reason": str(exc)})
    return records, pd.DataFrame(failed)


def _renormalize_shape(shape: np.ndarray) -> np.ndarray:
    shape = np.nan_to_num(np.asarray(shape, float), nan=0.0, posinf=1.0, neginf=0.0)
    shape = np.maximum.accumulate(np.clip(shape, 0.0, np.inf))
    shape[0] = 0.0
    if shape[-1] <= EPS:
        return np.linspace(0.0, 1.0, len(shape))
    return shape / shape[-1]


def material_shapes(train: pd.DataFrame, curve_cache: dict) -> dict:
    """Material-median normalized shapes from the proper-training fold only.

    This is the shape template used by M and H3.  It is always derived from
    training data, never from the held-out specimen.
    """
    available = [idx for idx in train.index if idx in curve_cache]
    global_shape = np.nanmedian(np.vstack([curve_cache[idx]["shape"] for idx in available]), axis=0)
    output = {}
    for material, group in train.groupby("material"):
        indices = [idx for idx in group.index if idx in curve_cache]
        output[material] = (
            np.nanmedian(np.vstack([curve_cache[idx]["shape"] for idx in indices]), axis=0)
            if indices
            else global_shape.copy()
        )
    output["__global__"] = global_shape
    return {name: _renormalize_shape(shape) for name, shape in output.items()}


def condition_knn_shape(train_df, train_indices, curve_cache, row, k=5) -> np.ndarray:
    """Distance-weighted k-nearest-condition shape in (temperature, log stress).

    This is the shape assignment used by M2; it prefers same-material
    neighbours and only pools across materials when fewer than three exist.
    """
    material = row["material"]
    material_indices = [idx for idx in train_indices if train_df.loc[idx, "material"] == material]
    candidates = material_indices if len(material_indices) >= 3 else list(train_indices)

    train_features = train_df.loc[candidates, ["temperature", "log_stress"]].to_numpy(float)
    test_features = np.asarray([row["temperature"], row["log_stress"]], float)
    center = np.nanmean(train_features, axis=0)
    scale = np.nanstd(train_features, axis=0)
    scale[scale < EPS] = 1.0

    distances = np.sqrt(np.nansum(((train_features - test_features) / scale) ** 2, axis=1))
    order = np.argsort(distances)[: min(k, len(candidates))]
    weights = 1.0 / np.maximum(distances[order], 1e-6)
    selected = [curve_cache[candidates[pos]]["shape"] for pos in order]
    if not selected:
        return _renormalize_shape(
            np.nanmean(np.vstack([curve_cache[idx]["shape"] for idx in train_indices]), axis=0)
        )
    shape = np.average(np.vstack(selected), axis=0, weights=weights)
    shape[0], shape[-1] = 0.0, 1.0
    return _renormalize_shape(np.clip(shape, -0.25, 1.75))


def reconstruct_from_shape(time, shape, rupture_time, final_strain) -> np.ndarray:
    """Dimensional reconstruction from endpoints and shape (Eq. 9)."""
    rupture_time = max(float(rupture_time), EPS)
    final_strain = max(float(final_strain), EPS)
    x = np.clip(np.asarray(time, float) / rupture_time, 0.0, 1.0)
    return final_strain * np.interp(x, GRID, shape)


def normalized_shape_prediction(time, shape, log_life, log_final) -> np.ndarray:
    """Reconstruct from log-space endpoint predictions, with saturation bounds."""
    life = float(10 ** np.clip(log_life, -8, 12))
    final = float(10 ** np.clip(log_final, -8, 4))
    return reconstruct_from_shape(time, shape, life, final)


def theta_shape_from_state(beta2, beta4, primary_final, tertiary_final) -> np.ndarray:
    """Normalized shape implied by an endpoint-normalized theta state.

    Used by T0 and T1 in place of an empirical shape template.
    """
    beta2 = 10.0 ** np.clip(beta2, -8, 4)
    beta4 = 10.0 ** np.clip(beta4, -8, 2)
    primary_final = 10.0 ** np.clip(primary_final, -10, 5)
    tertiary_final = 10.0 ** np.clip(tertiary_final, -10, 5)

    primary = primary_final * (1.0 - np.exp(-beta2 * GRID)) / max(1.0 - np.exp(-beta2), EPS)
    if beta4 < 50.0:
        tertiary = tertiary_final * np.expm1(beta4 * GRID) / max(np.expm1(beta4), EPS)
    else:
        tertiary = tertiary_final * np.exp(np.clip(beta4 * (GRID - 1.0), -700, 0))
        tertiary[0] = 0.0

    shape = np.maximum.accumulate(np.maximum(primary + tertiary, 0.0))
    if not np.isfinite(shape).all() or shape[-1] <= EPS:
        return GRID.copy()
    shape = shape / shape[-1]
    shape[0], shape[-1] = 0.0, 1.0
    return shape


def material_normalized_shape_table(data: pd.DataFrame, curve_cache: dict) -> pd.DataFrame:
    """Per-specimen S(x) on the grid, for the material-shape comparison figure."""
    rows = []
    for idx, row in data.iterrows():
        if idx not in curve_cache:
            continue
        shape = curve_cache[idx]["shape"]
        rows.append(
            {
                "material": row["material"],
                "relative_path": row["relative_path"],
                **{f"S_{value:.3f}": float(shape[position]) for position, value in enumerate(GRID)},
            }
        )
    return pd.DataFrame(rows)


# ==========================================================================
# 5  Condition-only state models and fold-safe priors
# ==========================================================================
#
# Two distinct condition-only predictors are used and must not be conflated:
#
#   * `fit_state_targets` is the model M center (theta_center_uncertainty.py):
#     Gaussian-process rupture life plus random-forest / extra-trees for the
#     remaining four state quantities, reconstructed as an endpoint-normalized
#     theta curve.
#
#   * `fold_safe_priors` is the baseline the prefix-informed models correct
#     (final_hybrid_nested_cv.py).  Proper-training rows receive inner
#     out-of-fold priors and target rows receive proper-trained priors, so the
#     residual targets are never fitted against a prior that saw the same row.


def one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(features: list[str]) -> ColumnTransformer:
    categorical = [name for name in ("material", "heat_treatment") if name in features]
    numeric = [name for name in features if name not in categorical]
    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", one_hot_encoder())]),
                categorical,
            )
        )
    return ColumnTransformer(transformers)


def make_state_model(name: str, features: list[str], seed: int, n_trees=N_TREES) -> Pipeline:
    """Model M's per-target regressors (theta_center_uncertainty.py::make_model)."""
    if name == "random_forest":
        regressor = RandomForestRegressor(
            n_estimators=n_trees, min_samples_leaf=3, max_features=0.9,
            random_state=seed, n_jobs=-1,
        )
    elif name == "extra_trees":
        regressor = ExtraTreesRegressor(
            n_estimators=n_trees, min_samples_leaf=3, max_features=0.9,
            random_state=seed, n_jobs=-1,
        )
    elif name == "gaussian_process":
        kernel = (
            ConstantKernel(1.0, constant_value_bounds="fixed")
            * RBF(length_scale=1.0, length_scale_bounds="fixed")
            + WhiteKernel(noise_level=0.1, noise_level_bounds="fixed")
        )
        regressor = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed)
    else:
        raise ValueError(f"Unknown state model: {name}")
    return Pipeline([("preprocess", make_preprocessor(features)), ("model", regressor)])


def fit_state_targets(train, test, features, seed, n_trees=N_TREES) -> np.ndarray:
    """Predict the five-element theta state for `test`, in STATE_TARGETS order."""
    predictions = np.zeros((len(test), len(STATE_TARGETS)), float)
    for position, target in enumerate(STATE_TARGETS):
        model = make_state_model(
            STATE_TARGET_MODELS[target], features, seed + position, n_trees=n_trees
        )
        model.fit(train[features], train[target].to_numpy(float))
        predictions[:, position] = model.predict(test[features])
    return predictions


def state_to_curve(time, state: np.ndarray) -> np.ndarray:
    """Model M's reconstruction: endpoint-normalized theta from the state vector."""
    return derived_curve(
        time,
        safe_power10(state[0]),
        safe_power10(state[1]),
        safe_power10(state[2]),
        safe_power10(state[3]),
        safe_power10(state[4]),
    )


def make_prior_model(label: str, seed: int) -> Pipeline:
    """Prior regressors (final_hybrid_nested_cv.py::make_model)."""
    numeric = [name for name in PRIOR_FEATURES if name not in {"material", "heat_treatment"}]
    preprocess = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("onehot", one_hot_encoder())]), ["material", "heat_treatment"]),
        ]
    )
    if label == "RandomForest":
        regressor = RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2, random_state=seed, n_jobs=-1
        )
    else:
        regressor = ExtraTreesRegressor(
            n_estimators=400, min_samples_leaf=2, random_state=seed, n_jobs=-1
        )
    return Pipeline([("preprocess", preprocess), ("model", regressor)])


def fold_safe_priors(proper: pd.DataFrame, target: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Condition-only priors: inner-OOF for proper rows, direct for target rows.

    Returns one row per input row, in proper-then-target order, carrying a
    `previous_<target>` column for each prior target.
    """
    output = pd.concat(
        [proper[KEYS].reset_index(drop=True), target[KEYS].reset_index(drop=True)],
        ignore_index=True,
    )
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    prior_values = {}

    for number, (name, label) in enumerate(PRIOR_TARGET_MODELS.items()):
        train_mask = proper[name].notna() & proper[PRIOR_FEATURES].notna().all(axis=1)
        test_mask = target[PRIOR_FEATURES].notna().all(axis=1)
        proper_valid = proper.loc[train_mask].reset_index(drop=True)
        target_valid = target.loc[test_mask].reset_index(drop=True)
        if len(proper_valid) < 10:
            raise RuntimeError(f"Too few valid proper-training rows for {name}: {len(proper_valid)}")

        oof = np.full(len(proper_valid), np.nan, float)
        for inner, (fit_pos, val_pos) in enumerate(splitter.split(proper_valid), 1):
            model = make_prior_model(label, seed + 1000 * number + inner)
            model.fit(proper_valid.iloc[fit_pos][PRIOR_FEATURES], proper_valid.iloc[fit_pos][name])
            oof[val_pos] = model.predict(proper_valid.iloc[val_pos][PRIOR_FEATURES])

        model = make_prior_model(label, seed + 1000 * number + 99)
        model.fit(proper_valid[PRIOR_FEATURES], proper_valid[name])
        direct = np.full(len(target), np.nan, float)
        if len(target_valid):
            direct[test_mask.to_numpy()] = model.predict(target_valid[PRIOR_FEATURES])

        proper_prior = pd.Series(np.nan, index=range(len(proper)), dtype=float)
        proper_prior.loc[np.flatnonzero(train_mask.to_numpy())] = oof
        prior_values[f"previous_{name}"] = np.concatenate(
            [proper_prior.to_numpy(), direct]
        )

    priors = pd.DataFrame(prior_values)
    priors["previous_log_final_shifted_strain"] = np.log10(
        10.0 ** priors["previous_log_primary_final_percent"]
        + 10.0 ** priors["previous_log_tertiary_final_percent"]
    )
    return pd.concat([output, priors], axis=1)


# --- Arrhenius / power-law baselines used by the P5 bounded update ---------
# Source: run_locked_p1_p2_benchmark.py.  The sign constraints encode the
# physical expectation that life falls with stress and rises with 1/T, while
# the minimum creep rate does the reverse.


def design_matrix(frame, materials, material_offsets):
    columns, names = [], []
    if material_offsets:
        for material in materials:
            columns.append(frame["material"].eq(material).to_numpy(float))
            names.append(f"intercept[{material}]")
    else:
        columns.append(np.ones(len(frame)))
        names.append("intercept")
    columns.extend(
        [
            pd.to_numeric(frame["log_stress"], errors="coerce").to_numpy(float),
            1000.0 * pd.to_numeric(frame["inv_temperature"], errors="coerce").to_numpy(float),
        ]
    )
    names.extend(["log_stress", "1000_over_temperature_K"])
    return np.column_stack(columns), names


def fit_power_law(train, test, target, kind, material_offsets=True):
    """Sign-constrained log-linear fit in (log stress, 1000/T)."""
    materials = sorted(train["material"].astype(str).unique())
    x_train, names = design_matrix(train, materials, material_offsets)
    x_test, _ = design_matrix(test, materials, material_offsets)
    y = pd.to_numeric(train[target], errors="coerce").to_numpy(float)
    valid = np.isfinite(y) & np.all(np.isfinite(x_train), axis=1)
    if np.count_nonzero(valid) <= x_train.shape[1]:
        raise RuntimeError(f"Insufficient proper-training rows for {target}")

    lower = np.full(x_train.shape[1], -np.inf)
    upper = np.full(x_train.shape[1], np.inf)
    stress_position, temperature_position = x_train.shape[1] - 2, x_train.shape[1] - 1
    if kind == "life":
        upper[stress_position] = 0.0      # life decreases with stress
        lower[temperature_position] = 0.0  # life increases with 1000/T
    elif kind == "rate":
        lower[stress_position] = 0.0
        upper[temperature_position] = 0.0
    else:
        raise ValueError(kind)

    result = lsq_linear(x_train[valid], y[valid], bounds=(lower, upper),
                        lsmr_tol="auto", max_iter=2000)
    predicted = np.clip(x_test @ result.x, -18.0, 12.0)
    coefficients = dict(zip(names, result.x))
    coefficients["fit_cost"] = float(result.cost)
    coefficients["fit_N"] = int(np.count_nonzero(valid))
    return predicted, coefficients


def material_final_prediction(train, test) -> np.ndarray:
    """Material-median terminal strain baseline."""
    target = "log_final_shifted_strain_percent"
    global_value = float(pd.to_numeric(train[target], errors="coerce").median())
    medians = (
        train.assign(_target=pd.to_numeric(train[target], errors="coerce"))
        .groupby("material")["_target"]
        .median()
        .to_dict()
    )
    return np.asarray([medians.get(material, global_value) for material in test["material"]], float)


def base_predictions(train, test):
    """Condition-only physics baselines (life, minimum creep rate, terminal strain)."""
    life, _ = fit_power_law(train, test, "log_rupture_time", "life", True)
    rate, _ = fit_power_law(train, test, "log_minimum_creep_rate", "rate", True)
    final = material_final_prediction(train, test)
    return life, rate, final


# ==========================================================================
# 6  Observed-prefix feature extraction
# ==========================================================================
#
# All features are computed from measurements at or before the observation
# boundary t_b only (Methods 2.5).  The boundary itself is set from the
# condition-only life estimate, never from the specimen's measured rupture
# life, so nothing here is retrospective.


def early_features_at_horizon(row, cache, horizon) -> dict:
    """Accumulated strain, secant rate and tail rate over the observed prefix.

    Source: run_locked_p5_m2_h3_benchmark.py::early_features_at_horizon.
    """
    time = np.asarray(cache["time"], float)
    shifted = np.asarray(cache["shifted"], float)
    horizon = float(np.clip(horizon, np.nanmin(time), np.nanmax(time)))

    keep = np.isfinite(time) & np.isfinite(shifted) & (time <= horizon)
    if np.count_nonzero(keep) < 3:
        finite = np.flatnonzero(np.isfinite(time) & np.isfinite(shifted))
        keep[finite[np.argsort(time[finite])[: min(3, len(finite))]]] = True

    t_obs, e_obs = time[keep], shifted[keep]
    order = np.argsort(t_obs, kind="stable")
    t_obs, e_obs = t_obs[order], e_obs[order]
    t_last = max(float(t_obs[-1]), EPS)
    e_last = max(float(e_obs[-1]), EPS)

    if len(t_obs) >= 5:
        tail_n = max(3, int(np.ceil(0.35 * len(t_obs))))
        tail_t, tail_e = t_obs[-tail_n:], e_obs[-tail_n:]
        tail_rate = np.polyfit(tail_t, tail_e, 1)[0] if np.ptp(tail_t) > 0 else np.nan
    else:
        tail_rate = np.nan

    prior_final = safe_power10(row["previous_log_final_shifted_strain"])
    prior_time = safe_power10(row["previous_log_rupture_time"])
    return {
        "reveal_time": t_last,
        "reveal_horizon": horizon,
        "reveal_fraction_of_actual_life": t_last / max(float(cache["actual_rupture_time"]), EPS),
        "early_log_time": np.log10(t_last),
        "early_log_strain": np.log10(e_last),
        "early_log_secant_rate": np.log10(max(e_last / t_last, EPS)),
        "early_log_tail_rate": np.log10(max(float(tail_rate), EPS)) if np.isfinite(tail_rate) else np.nan,
        "early_strain_over_prior_final": e_last / max(prior_final, EPS),
        "early_time_over_prior_time": t_last / max(prior_time, EPS),
    }


def robust_rate(time, strain) -> np.ndarray:
    """Positive-floored numerical creep rate."""
    time = np.asarray(time, float)
    strain = np.asarray(strain, float)
    if len(time) < 3 or np.ptp(time) <= 0:
        return np.full(len(time), np.nan)
    rate = np.gradient(strain, time)
    positive = rate[np.isfinite(rate) & (rate > 0)]
    floor = np.nanpercentile(positive, 5) * 0.05 if len(positive) else EPS
    return np.clip(rate, max(floor, EPS), np.inf)


EMPTY_OMEGA_FEATURES = {
    "omega_log_rate_vs_strain": np.nan,
    "omega_log_rate_vs_time": np.nan,
    "early_log_rate_start": np.nan,
    "early_log_rate_end": np.nan,
    "early_log_rate_ratio": np.nan,
    "early_rate_curvature": np.nan,
    "early_strain_fraction_slope": np.nan,
}


def omega_features_for_row(row, cache) -> dict:
    """Omega-style rate descriptors over the observed prefix.

    `omega_log_rate_vs_strain` is the slope of log10(rate) against strain - the
    Omega-method state variable - evaluated on the prefix alone.
    """
    time = np.asarray(cache["time"], float)
    shifted = np.asarray(cache["shifted"], float)
    reveal_time = float(row["reveal_time"])
    keep = time <= reveal_time
    if keep.sum() < 5:
        keep[np.argsort(time)[: min(5, len(time))]] = True

    t, e = time[keep], shifted[keep]
    valid = np.isfinite(t) & np.isfinite(e)
    t, e = t[valid], e[valid]
    if len(t) < 5:
        return dict(EMPTY_OMEGA_FEATURES)

    rate = robust_rate(t, e)
    log_rate = np.log10(np.clip(rate, EPS, np.inf))
    shifted_e = e - np.nanmin(e)
    shifted_t = t - np.nanmin(t)

    omega = log_time_slope = curvature = strain_slope = np.nan
    if np.nanstd(shifted_e) > EPS:
        try:
            omega = np.polyfit(shifted_e, log_rate, 1)[0]
        except Exception:
            omega = np.nan
    if np.nanstd(np.log10(np.clip(t, EPS, np.inf))) > EPS:
        try:
            log_time_slope = np.polyfit(np.log10(np.clip(t, EPS, np.inf)), log_rate, 1)[0]
        except Exception:
            log_time_slope = np.nan
    if len(t) >= 6 and np.ptp(shifted_t) > 0:
        try:
            scaled = shifted_t / max(np.nanmax(shifted_t), EPS)
            curvature = np.polyfit(scaled, e, 2)[0]
            prior_final = safe_power10(row["previous_log_final_shifted_strain"])
            strain_slope = np.polyfit(scaled, e / max(prior_final, EPS), 1)[0]
        except Exception:
            curvature = strain_slope = np.nan

    head_n = tail_n = max(2, int(np.ceil(0.25 * len(log_rate))))
    start_rate = float(np.nanmedian(log_rate[:head_n]))
    end_rate = float(np.nanmedian(log_rate[-tail_n:]))
    return {
        "omega_log_rate_vs_strain": omega,
        "omega_log_rate_vs_time": log_time_slope,
        "early_log_rate_start": start_rate,
        "early_log_rate_end": end_rate,
        "early_log_rate_ratio": end_rate - start_rate,
        "early_rate_curvature": curvature,
        "early_strain_fraction_slope": strain_slope,
    }


EMPTY_SMOOTH_FEATURES = {
    "smooth_log_median_rate": np.nan,
    "smooth_log_rate_cv": np.nan,
    "smooth_log_rate_iqr": np.nan,
    "smooth_log_accel_ratio": np.nan,
    "smooth_omega_log_rate_vs_strain": np.nan,
    "smooth_late_rate_slope": np.nan,
    "smooth_strain_curvature": np.nan,
    "early_fraction_of_prior_rupture_strain": np.nan,
}


def smooth_strain(time, strain) -> np.ndarray:
    strain = np.asarray(strain, float)
    if len(strain) < 7 or savgol_filter is None:
        return strain.copy()
    window = min(21, len(strain) // 2 * 2 - 1)
    if window < 7:
        return strain.copy()
    try:
        return savgol_filter(strain, window_length=window, polyorder=2)
    except Exception:
        return strain.copy()


def smoothed_features_for_row(row, cache) -> dict:
    """Savitzky-Golay-smoothed rate statistics over the observed prefix.

    These capture rate dispersion and acceleration, which distinguish a
    specimen already entering tertiary creep from one still in primary.
    """
    time = np.asarray(cache["time"], float)
    strain = np.asarray(cache["shifted"], float)
    reveal_time = float(row["reveal_time"])
    keep = time <= reveal_time
    if keep.sum() < 7:
        keep[np.argsort(time)[: min(7, len(time))]] = True

    t, e = time[keep], strain[keep]
    valid = np.isfinite(t) & np.isfinite(e)
    t, e = t[valid], e[valid]
    if len(t) < 5 or np.ptp(t) <= 0:
        return dict(EMPTY_SMOOTH_FEATURES)

    es = smooth_strain(t, e)
    rate = np.gradient(es, t)
    positive = rate[np.isfinite(rate) & (rate > 0)]
    if len(positive) == 0:
        return dict(EMPTY_SMOOTH_FEATURES)
    floor = max(np.nanpercentile(positive, 5) * 0.05, EPS)
    log_rate = np.log10(np.clip(rate, floor, np.inf))
    log_rate_finite = log_rate[np.isfinite(log_rate)]
    if len(log_rate_finite) == 0:
        return dict(EMPTY_SMOOTH_FEATURES)

    early_n = late_n = max(2, int(np.ceil(0.25 * len(log_rate))))
    early_rate = float(np.nanmedian(log_rate[:early_n]))
    late_rate = float(np.nanmedian(log_rate[-late_n:]))

    shifted_e = es - np.nanmin(es)
    scaled_t = (t - np.nanmin(t)) / max(np.ptp(t), EPS)
    smooth_omega = late_slope = curvature = np.nan
    if np.nanstd(shifted_e) > EPS:
        try:
            smooth_omega = np.polyfit(shifted_e, log_rate, 1)[0]
        except Exception:
            smooth_omega = np.nan
    if len(t) >= 5 and np.nanstd(scaled_t[-late_n:]) > EPS:
        try:
            late_slope = np.polyfit(scaled_t[-late_n:], log_rate[-late_n:], 1)[0]
        except Exception:
            late_slope = np.nan
    if len(t) >= 6:
        try:
            curvature = np.polyfit(scaled_t, es, 2)[0]
        except Exception:
            curvature = np.nan

    prior_rupture = 10.0 ** np.clip(
        float(row.get("previous_log_rupture_strain_percent", np.nan)), -12, 12
    )
    if not np.isfinite(prior_rupture) or prior_rupture <= 0:
        prior_rupture = 10.0 ** np.clip(
            float(row.get("previous_log_final_shifted_strain", 0.0)), -12, 12
        )

    return {
        "smooth_log_median_rate": float(np.nanmedian(log_rate_finite)),
        "smooth_log_rate_cv": float(
            np.nanstd(log_rate_finite) / max(abs(np.nanmean(log_rate_finite)), 1e-6)
        ),
        "smooth_log_rate_iqr": float(
            np.nanpercentile(log_rate_finite, 75) - np.nanpercentile(log_rate_finite, 25)
        ),
        "smooth_log_accel_ratio": late_rate - early_rate,
        "smooth_omega_log_rate_vs_strain": smooth_omega,
        "smooth_late_rate_slope": late_slope,
        "smooth_strain_curvature": curvature,
        "early_fraction_of_prior_rupture_strain": float(
            max(es[-1], EPS) / max(prior_rupture, EPS)
        ),
    }


def build_featured(proper, target, curve_cache, reveal_fraction, seed=SEED):
    """Assemble the prefix feature table for one fold and reveal fraction.

    Source: run_locked_p5_m2_h3_benchmark.py::build_featured.  The observation
    boundary for each row is `reveal_fraction * condition-predicted life`,
    which is Eq. 10 of the manuscript.
    """
    combined = pd.concat([proper, target], axis=0)
    priors = fold_safe_priors(proper, target, seed + int(10000 * reveal_fraction))

    prior_map = priors.set_index(pd.MultiIndex.from_frame(priors[KEYS]))
    combined_keys = pd.MultiIndex.from_frame(combined[KEYS])
    featured_source = combined.copy()
    for column in [name for name in priors if name.startswith("previous_")]:
        featured_source[column] = prior_map.loc[combined_keys, column].to_numpy()
    featured_source["log_final_shifted_strain"] = np.log10(
        10.0 ** featured_source["log_primary_final_percent"]
        + 10.0 ** featured_source["log_tertiary_final_percent"]
    )

    # Observation boundary from the condition-only life estimate (Eq. 10).
    proper_state = fit_state_targets(proper, proper, CONDITION_FEATURES, seed + 101)
    target_state = fit_state_targets(proper, target, CONDITION_FEATURES, seed + 101)
    condition_life = pd.concat(
        [
            pd.Series(proper_state[:, 0], index=proper.index),
            pd.Series(target_state[:, 0], index=target.index),
        ]
    )

    records = []
    for idx, row in featured_source.iterrows():
        horizon = reveal_fraction * safe_power10(condition_life.loc[idx])
        early = early_features_at_horizon(row, curve_cache[idx], horizon)
        enriched = row.copy()
        for name, value in early.items():
            enriched[name] = value
        omega = omega_features_for_row(enriched, curve_cache[idx])
        for name, value in omega.items():
            enriched[name] = value
        smooth = smoothed_features_for_row(enriched, curve_cache[idx])
        records.append({"idx": idx, **early, **omega, **smooth})

    featured = featured_source.join(pd.DataFrame(records).set_index("idx"))
    featured["resid_log_rupture_time"] = (
        featured["log_rupture_time"] - featured["previous_log_rupture_time"]
    )
    featured["resid_log_final_shifted_strain"] = (
        featured["log_final_shifted_strain"] - featured["previous_log_final_shifted_strain"]
    )
    featured = featured.replace([np.inf, -np.inf], np.nan)
    return featured.loc[proper.index].copy(), featured.loc[target.index].copy()


def make_residual_model() -> Pipeline:
    """Residual regressor used by M2 and H3 (refined_early_forecast_model.py)."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=240, min_samples_leaf=3, max_features=0.9,
                    random_state=SEED, n_jobs=-1,
                ),
            ),
        ]
    )


def fit_predict_residual(train, test, target, features) -> np.ndarray:
    subset = (
        train[[*features, "material", target]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=[target])
    )
    model = make_residual_model()
    model.fit(subset[features], subset[target].to_numpy(float))
    return np.asarray(model.predict(test[features]), float)


# ==========================================================================
# 7  Bounded physics prefix update (P5)
# ==========================================================================
#
# The observed prefix is used to rescale only the characteristic time and
# strain scales of the condition-only forecast.  Both corrections are bounded
# to +/- 1 decade and weakly regularized toward zero, so a short or noisy
# prefix cannot move an endpoint by an unsupported amount (Methods 2.4).


def fit_prefix_state(cache, shape, base_log_life, base_log_final, reveal_time) -> dict:
    """Least-squares rescaling of (log t_r, log eps_f) to match the prefix."""
    time = np.asarray(cache["time"], float)
    strain = np.asarray(cache["shifted"], float)
    keep = np.isfinite(time) & np.isfinite(strain) & (time <= float(reveal_time))

    if np.count_nonzero(keep) < MIN_PREFIX_POINTS:
        return {
            "p5_log_rupture_time": float(base_log_life),
            "p5_log_final_shifted_strain": float(base_log_final),
            "p5_life_scale_correction": 0.0,
            "p5_final_scale_correction": 0.0,
            "p5_prefix_fit_nrmse": np.nan,
            "p5_fit_success": False,
        }

    observed_time, observed_strain = time[keep], strain[keep]
    scale = max(
        float(np.ptp(observed_strain)),
        0.10 * abs(float(observed_strain[-1])),
        0.02 * float(10 ** np.clip(base_log_final, -8, 4)),
        1e-6,
    )

    def residual(delta):
        predicted = normalized_shape_prediction(
            observed_time, shape, base_log_life + delta[0], base_log_final + delta[1]
        )
        data_residual = (predicted - observed_strain) / scale
        regularization = P5_PRIOR_WEIGHT * np.asarray(delta) / P5_PRIOR_SCALE
        return np.r_[data_residual, regularization]

    try:
        result = least_squares(
            residual,
            np.zeros(2),
            bounds=(-P5_DELTA_BOUND, P5_DELTA_BOUND),
            loss="soft_l1",
            f_scale=0.20,
            max_nfev=250,
        )
        delta_life, delta_final = result.x
        predicted = normalized_shape_prediction(
            observed_time, shape, base_log_life + delta_life, base_log_final + delta_final
        )
        prefix_nrmse = float(np.sqrt(np.mean((predicted - observed_strain) ** 2)) / scale)
        success = bool(result.success)
    except Exception:
        delta_life = delta_final = 0.0
        prefix_nrmse = np.nan
        success = False

    return {
        "p5_log_rupture_time": float(base_log_life + delta_life),
        "p5_log_final_shifted_strain": float(base_log_final + delta_final),
        "p5_life_scale_correction": float(delta_life),
        "p5_final_scale_correction": float(delta_final),
        "p5_prefix_fit_nrmse": prefix_nrmse,
        "p5_fit_success": success,
    }


def inner_oof_p5(proper, featured, curve_cache) -> pd.DataFrame:
    """Inner grouped-OOF P5 state for proper rows, so H3 residuals are honest."""
    groups = proper["temperature"].astype(str) + "_" + proper["stress"].round(6).astype(str)
    records = []
    for inner, (train_pos, val_pos) in enumerate(GroupKFold(N_SPLITS).split(proper, groups=groups), 1):
        train, validation = proper.iloc[train_pos], proper.iloc[val_pos]
        life, rate, final = base_predictions(train, validation)
        shapes = material_shapes(train, curve_cache)
        for position, (idx, row) in enumerate(validation.iterrows()):
            shape = shapes.get(row["material"], shapes["__global__"])
            state = fit_prefix_state(
                curve_cache[idx], shape, life[position], final[position],
                featured.loc[idx, "reveal_time"],
            )
            state["idx"] = idx
            state["p5_log_minimum_creep_rate"] = float(
                rate[position]
                + state["p5_final_scale_correction"]
                - state["p5_life_scale_correction"]
            )
            state["inner_fold"] = inner
            records.append(state)

    result = pd.DataFrame(records).set_index("idx")
    if set(result.index) != set(proper.index):
        raise RuntimeError("Inner P5 predictions did not cover proper training exactly")
    return result.loc[proper.index]


def direct_p5(proper, target, featured, curve_cache):
    """Proper-trained P5 state and centers for the held-out rows."""
    life, rate, final = base_predictions(proper, target)
    shapes = material_shapes(proper, curve_cache)
    records, centers = [], {}
    for position, (idx, row) in enumerate(target.iterrows()):
        shape = shapes.get(row["material"], shapes["__global__"])
        state = fit_prefix_state(
            curve_cache[idx], shape, life[position], final[position],
            featured.loc[idx, "reveal_time"],
        )
        state["idx"] = idx
        state["p5_log_minimum_creep_rate"] = float(
            rate[position]
            + state["p5_final_scale_correction"]
            - state["p5_life_scale_correction"]
        )
        centers[idx] = normalized_shape_prediction(
            curve_cache[idx]["time"], shape,
            state["p5_log_rupture_time"], state["p5_log_final_shifted_strain"],
        )
        records.append(state)
    return pd.DataFrame(records).set_index("idx").loc[target.index], centers, shapes


# ==========================================================================
# 8  Forecasting models                                  (Methods 2.4)
# ==========================================================================
#
#   M   condition-only reference: material, temperature, stress only.
#   T0  theta state from conditions only.
#   T1  theta state from conditions plus the observed prefix.
#   M2  matched prefix ML: prior endpoints plus learned residuals, kNN shape.
#   H3  physics-guided hybrid: P5 bounded update plus bounded residual ML,
#       with a material-median training shape.


def fit_model_M(proper, target, curve_cache, seed=SEED):
    """Condition-only reference forecast (model M).

    Predicts the five-element theta state from loading conditions and
    reconstructs the endpoint-normalized theta curve.  No measurement from the
    held-out specimen is used.
    """
    state = fit_state_targets(proper, target, CONDITION_FEATURES, seed + 101)
    centers, lives, records = {}, {}, {}
    for position, (idx, _row) in enumerate(target.iterrows()):
        centers[idx] = state_to_curve(curve_cache[idx]["time"], state[position])
        lives[idx] = safe_power10(state[position][0])
        records[idx] = {
            "m_log_rupture_time": float(state[position][0]),
            "m_log_final_shifted_strain": float(
                np.log10(
                    max(safe_power10(state[position][3]) + safe_power10(state[position][4]), EPS)
                )
            ),
        }
    return pd.DataFrame.from_dict(records, orient="index"), centers, lives


def make_theta_ml_model(features: list[str], seed: int) -> Pipeline:
    """Multi-output ExtraTrees over the endpoint-normalized theta state (T0/T1)."""
    return Pipeline(
        [
            ("preprocess", make_preprocessor(features)),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=240, min_samples_leaf=3, max_features=0.9,
                    random_state=seed, n_jobs=-1,
                ),
            ),
        ]
    )


def fit_theta_oof_and_target(proper, target, proper_features, target_features, features, seed):
    """Grouped inner-OOF theta state for proper rows; proper-trained for target.

    T1 predicts rupture life and endpoint-normalized quantities rather than the
    dimensional theta1-theta4, which span several decades and are strongly
    coupled (Methods 2.2).
    """
    y = proper[STATE_TARGETS].to_numpy(float)
    oof = np.full_like(y, np.nan)
    groups = proper["temperature"].astype(str) + "_" + proper["stress"].round(6).astype(str)

    for inner, (train_pos, val_pos) in enumerate(GroupKFold(N_SPLITS).split(proper, groups=groups), 1):
        model = make_theta_ml_model(features, seed + inner)
        model.fit(proper_features.iloc[train_pos][features], y[train_pos])
        oof[val_pos] = model.predict(proper_features.iloc[val_pos][features])
    if not np.isfinite(oof).all():
        raise RuntimeError("Theta inner OOF predictions are incomplete")

    model = make_theta_ml_model(features, seed + 100)
    model.fit(proper_features[features], y)
    predicted = model.predict(target_features[features])

    names = [f"theta_pred_{name}" for name in STATE_TARGETS]
    return (
        pd.DataFrame(oof, index=proper.index, columns=names),
        pd.DataFrame(predicted, index=target.index, columns=names),
    )


def stable_expm1(value):
    value = np.asarray(value, float)
    return np.where(value < 50.0, np.expm1(value), np.exp(np.clip(value, -700, 700)))


def add_theta_descriptors(predictions: pd.DataFrame) -> pd.DataFrame:
    """Map the predicted normalized state back to dimensional theta1-theta4.

    This is the inverse of Eqs. 3-4, plus late-life descriptors evaluated at
    x = 0.8 that summarize the predicted tertiary response.
    """
    result = predictions.copy()
    life = 10.0 ** np.clip(result["theta_pred_log_rupture_time"].to_numpy(float), -8, 12)
    beta2 = 10.0 ** np.clip(result["theta_pred_log_beta2_normalized"].to_numpy(float), -8, 4)
    beta4 = 10.0 ** np.clip(result["theta_pred_log_beta4_normalized"].to_numpy(float), -8, 2)
    primary_final = 10.0 ** np.clip(result["theta_pred_log_primary_final_percent"].to_numpy(float), -10, 5)
    tertiary_final = 10.0 ** np.clip(result["theta_pred_log_tertiary_final_percent"].to_numpy(float), -10, 5)

    theta1 = primary_final / np.maximum(-np.expm1(-beta2), EPS)
    theta2 = beta2 / np.maximum(life, EPS)
    theta3 = tertiary_final / np.maximum(stable_expm1(beta4), EPS)
    theta4 = beta4 / np.maximum(life, EPS)
    for number, values in enumerate((theta1, theta2, theta3, theta4), 1):
        result[f"theta{number}_pred"] = values
        result[f"theta_log_theta{number}"] = np.log10(np.maximum(values, EPS))

    result["theta_predicted_life"] = life
    result["theta_predicted_final"] = primary_final + tertiary_final

    x = 0.80
    exponent = np.clip(theta4 * life * x, -700, 700)
    tertiary_contribution = theta3 * np.expm1(exponent)
    primary_contribution = theta1 * (1.0 - np.exp(-theta2 * life * x))
    result["theta_log_tertiary_contribution_80"] = np.log10(np.maximum(tertiary_contribution, EPS))
    result["theta_log_tertiary_rate_80"] = np.log10(
        np.maximum(theta3 * theta4 * np.exp(exponent), EPS)
    )
    result["theta_log_tertiary_accel_80"] = np.log10(
        np.maximum(theta3 * theta4**2 * np.exp(exponent), EPS)
    )
    result["theta_tertiary_fraction_80"] = tertiary_contribution / np.maximum(
        primary_contribution + tertiary_contribution, EPS
    )

    onset = []
    for a, b, c, d, tr in zip(theta1, theta2, theta3, theta4, life):
        time = GRID * tr
        primary = a * (1.0 - np.exp(np.clip(-b * time, -700, 50)))
        tertiary = c * np.expm1(np.clip(d * time, -50, 50))
        fraction = tertiary / np.maximum(primary + tertiary, EPS)
        hit = np.flatnonzero(fraction >= 0.10)
        onset.append(float(GRID[hit[0]]) if len(hit) else 1.0)
    result["theta_onset_10pct"] = onset

    return result.replace([np.inf, -np.inf], np.nan)


def theta_center(row: pd.Series, time: np.ndarray) -> np.ndarray:
    """Dimensional theta forecast from a predicted normalized state (T0/T1)."""
    shape = theta_shape_from_state(
        float(row["theta_pred_log_beta2_normalized"]),
        float(row["theta_pred_log_beta4_normalized"]),
        float(row["theta_pred_log_primary_final_percent"]),
        float(row["theta_pred_log_tertiary_final_percent"]),
    )
    return reconstruct_from_shape(
        time, shape,
        float(row["theta_predicted_life"]),
        float(row["theta_predicted_final"]),
    )


def fit_model_M2(proper, target, featured_train, featured_target, curve_cache):
    """Matched prefix ML: prior endpoints plus unbounded learned residuals.

    M2 uses the same information as H3 but no bounded physics update and a
    condition-kNN shape rather than a material-median template.  It is the
    architecture control for H3.
    """
    life_correction = fit_predict_residual(
        featured_train, featured_target, "resid_log_rupture_time", SMOOTH_FEATURES
    )
    final_correction = fit_predict_residual(
        featured_train, featured_target, "resid_log_final_shifted_strain", SMOOTH_FEATURES
    )

    combined = pd.concat([featured_train, featured_target])
    train_indices = proper.index.to_numpy()
    states, centers = {}, {}
    for position, (idx, row) in enumerate(featured_target.iterrows()):
        log_life = float(row["previous_log_rupture_time"] + life_correction[position])
        log_final = float(row["previous_log_final_shifted_strain"] + final_correction[position])
        shape = condition_knn_shape(combined, train_indices, curve_cache, row, k=5)
        centers[idx] = normalized_shape_prediction(curve_cache[idx]["time"], shape, log_life, log_final)
        states[idx] = {
            "m2_log_rupture_time": log_life,
            "m2_log_final_shifted_strain": log_final,
            "m2_life_residual_correction": float(life_correction[position]),
            "m2_final_residual_correction": float(final_correction[position]),
        }
    return pd.DataFrame.from_dict(states, orient="index"), centers


def fit_model_H3(proper, target, featured_train, featured_target,
                 p5_train, p5_target, shapes, curve_cache):
    """Physics-guided hybrid forecast (model H3).

    Layers, in order:
      1. condition-only physics baselines for life and terminal strain;
      2. the bounded P5 prefix update of those two scales (Section 7);
      3. ExtraTrees residual corrections on top of the P5 state, clipped to
         +/- 1 decade so learned corrections cannot dominate the physics;
      4. reconstruction against the material-median training shape (Eq. 9).
    """
    train = featured_train.join(p5_train)
    test = featured_target.join(p5_target)
    train["h3_life_residual"] = train["log_rupture_time"] - train["p5_log_rupture_time"]
    train["h3_final_residual"] = (
        train["log_final_shifted_strain"] - train["p5_log_final_shifted_strain"]
    )

    life_correction = np.clip(
        fit_predict_residual(train, test, "h3_life_residual", H3_FEATURES),
        -H3_CORRECTION_BOUND, H3_CORRECTION_BOUND,
    )
    final_correction = np.clip(
        fit_predict_residual(train, test, "h3_final_residual", H3_FEATURES),
        -H3_CORRECTION_BOUND, H3_CORRECTION_BOUND,
    )

    output, centers = {}, {}
    for position, (idx, row) in enumerate(target.iterrows()):
        log_life = float(p5_target.loc[idx, "p5_log_rupture_time"] + life_correction[position])
        log_final = float(
            p5_target.loc[idx, "p5_log_final_shifted_strain"] + final_correction[position]
        )
        log_rate = float(
            p5_target.loc[idx, "p5_log_minimum_creep_rate"]
            + final_correction[position]
            - life_correction[position]
        )
        shape = shapes.get(row["material"], shapes["__global__"])
        centers[idx] = normalized_shape_prediction(
            curve_cache[idx]["time"], shape, log_life, log_final
        )
        output[idx] = {
            "h3_log_rupture_time": log_life,
            "h3_log_final_shifted_strain": log_final,
            "h3_log_minimum_creep_rate": log_rate,
            "h3_life_residual_correction": float(life_correction[position]),
            "h3_final_residual_correction": float(final_correction[position]),
        }
    return pd.DataFrame.from_dict(output, orient="index"), centers


# ==========================================================================
# 9  Cross-validation protocol                           (Methods 2.5)
# ==========================================================================
#
# Five-fold GroupKFold over material-temperature-stress conditions, so
# replicates at one nominal condition never straddle the train/test boundary.
# Within each outer-training fold, a material-stratified split separates the
# rows used to fit point forecasts and shape templates ("proper") from those
# reserved for uncertainty calibration ("calibration").


def split_schemes(data: pd.DataFrame):
    schemes = [
        ("random_curve_5fold",
         KFold(N_SPLITS, shuffle=True, random_state=CENTER_RANDOM_STATE), None)
    ]
    group_specs = [
        ("grouped_stress_5fold", data["stress"].round(6).astype(str)),
        ("grouped_temperature_5fold", data["temperature"].astype(str)),
        (SPLIT_SCHEME,
         data["temperature"].astype(str) + "_" + data["stress"].round(6).astype(str)),
    ]
    for name, values in group_specs:
        if values.nunique() >= N_SPLITS:
            schemes.append((name, GroupKFold(N_SPLITS), values.to_numpy()))
    return schemes


def stratified_proper_calibration_split(train: pd.DataFrame, seed: int):
    """Material-stratified proper / calibration split by row position."""
    rng = np.random.default_rng(seed)
    proper, calibration = [], []
    materials = train["material"].to_numpy(object)
    for material in pd.unique(materials):
        positions = rng.permutation(np.flatnonzero(materials == material))
        n_calibration = int(np.ceil(CALIBRATION_FRACTION * len(positions)))
        n_calibration = min(max(n_calibration, 1), len(positions) - 1)
        calibration.extend(positions[:n_calibration])
        proper.extend(positions[n_calibration:])
    return np.sort(proper), np.sort(calibration)


# ==========================================================================
# 10  Scoring, bootstrap intervals, condition support
# ==========================================================================


def suffix_score(actual, predicted, time, boundary) -> dict | None:
    """Curve-level NRMSE over the withheld suffix (Eq. 11).

    Normalized by the full measured strain range so that curves with different
    ductilities are comparable, and returned once per curve so that densely
    sampled tests do not dominate the pooled statistics.  Returns None when the
    eligibility rule (>= 5 prefix, >= 3 suffix points) is not met.
    """
    actual = np.asarray(actual, float)
    predicted = np.asarray(predicted, float)
    time = np.asarray(time, float)

    keep = np.isfinite(actual) & np.isfinite(predicted) & np.isfinite(time) & (time > float(boundary))
    prefix = np.isfinite(actual) & np.isfinite(time) & (time <= float(boundary))
    if np.count_nonzero(keep) < MIN_SUFFIX_POINTS or np.count_nonzero(prefix) < MIN_PREFIX_POINTS:
        return None

    full_range = float(np.ptp(actual[np.isfinite(actual)]))
    suffix_range = float(np.ptp(actual[keep]))
    if full_range <= EPS or suffix_range <= EPS:
        return None

    return {
        "n_prefix_points": int(np.count_nonzero(prefix)),
        "n_suffix_points": int(np.count_nonzero(keep)),
        "full_strain_range": full_range,
        "withheld_strain_range": suffix_range,
        "remainder_nrmse": float(
            np.sqrt(np.mean((actual[keep] - predicted[keep]) ** 2)) / full_range
        ),
    }


def region_nrmse(actual, predicted, time, mask, denominator) -> float:
    mask = (
        np.asarray(mask, bool)
        & np.isfinite(actual) & np.isfinite(predicted) & np.isfinite(time)
    )
    if np.count_nonzero(mask) < 3:
        return np.nan
    return float(
        np.sqrt(np.mean((np.asarray(predicted)[mask] - np.asarray(actual)[mask]) ** 2))
        / denominator
    )


def late_life_scores(actual, predicted, time, boundary, actual_life) -> dict:
    """Late-life trajectory error over the final 30% of measured creep life."""
    actual = np.asarray(actual, float)
    predicted = np.asarray(predicted, float)
    time = np.asarray(time, float)
    denominator = max(float(np.ptp(actual[np.isfinite(actual)])), EPS)
    suffix = time > boundary

    result = {
        "tertiary_nrmse_70": region_nrmse(
            actual, predicted, time,
            suffix & (time / max(actual_life, EPS) >= 0.70), denominator,
        ),
        "tertiary_nrmse_80": region_nrmse(
            actual, predicted, time,
            suffix & (time / max(actual_life, EPS) >= 0.80), denominator,
        ),
    }
    for x in (0.70, 0.80, 0.90):
        token = int(100 * x)
        actual_value = float(np.interp(x * actual_life, time, actual))
        predicted_value = float(np.interp(x * actual_life, time, predicted))
        result[f"late_strain_actual_{token}"] = actual_value
        result[f"late_strain_predicted_{token}"] = predicted_value
        result[f"late_strain_error_{token}"] = predicted_value - actual_value
    return result


def clustered_bootstrap_median(values, clusters, n_boot=N_BOOT, seed=THETA_SEED):
    """Percentile bootstrap on the median, resampling whole conditions.

    Replicates at one material-temperature-stress condition are not
    independent, so the cluster - not the curve - is the resampling unit.
    """
    frame = pd.DataFrame({"value": values, "cluster": clusters}).dropna()
    pools = {key: group["value"].to_numpy(float) for key, group in frame.groupby("cluster")}
    keys = np.asarray(list(pools), object)
    if len(keys) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    estimates = [
        np.median(np.concatenate([pools[key] for key in rng.choice(keys, len(keys), replace=True)]))
        for _ in range(n_boot)
    ]
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def hull_status(train: pd.DataFrame, row: pd.Series) -> bool:
    """True when the condition lies OUTSIDE the material training convex hull.

    Computed in standardized (log stress, 1/T) space per material.  Degenerate
    training sets (fewer than three distinct conditions) count as outside.
    """
    group = train[train["material"].eq(row["material"])]
    x = group[["log_stress", "inv_temperature"]].to_numpy(float)
    z = row[["log_stress", "inv_temperature"]].to_numpy(float)
    scale = np.nanstd(x, axis=0)
    scale[scale < EPS] = 1.0
    xs = (x - np.nanmean(x, axis=0)) / scale
    zs = (z - np.nanmean(x, axis=0)) / scale
    unique = np.unique(xs, axis=0)
    if len(unique) < 3:
        return True
    try:
        return bool(Delaunay(unique).find_simplex(zs) < 0)
    except QhullError:
        return True


def summarize_model(frame: pd.DataFrame, group_columns=("model", "reveal_fraction")) -> pd.DataFrame:
    """Median / P95 / severe-error summary, one row per model and reveal."""
    rows = []
    for keys, group in frame.groupby(list(group_columns), sort=True):
        values = group["remainder_nrmse"].to_numpy(float)
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        record = dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,)))
        record.update(
            {
                "N": int(len(values)),
                "median_nrmse": float(np.median(values)),
                "mean_nrmse": float(np.mean(values)),
                "p95_nrmse": float(np.percentile(values, 95)),
                "severe_n": int(np.count_nonzero(values > 1.0)),
                "severe_rate": float(np.mean(values > 1.0)),
                "fraction_below_0.20": float(np.mean(values < 0.20)),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def paired_comparison(frame: pd.DataFrame, reference: str, comparator: str) -> pd.DataFrame:
    """Paired comparator-minus-H3 differences on identical curve-reveal rows.

    Positive values favour `reference`.  Intervals are condition-clustered.
    """
    rows = []
    key = ["fold", "relative_path", "reveal_fraction"]
    for reveal, group in frame.groupby("reveal_fraction"):
        left = group[group["model"].eq(comparator)].set_index(key)
        right = group[group["model"].eq(reference)].set_index(key)
        shared = left.index.intersection(right.index)
        if not len(shared):
            continue
        difference = (
            left.loc[shared, "remainder_nrmse"].to_numpy(float)
            - right.loc[shared, "remainder_nrmse"].to_numpy(float)
        )
        clusters = right.loc[shared, "condition_id"].to_numpy()
        low, high = clustered_bootstrap_median(difference, clusters)
        rows.append(
            {
                "reveal_fraction": reveal,
                "reference": reference,
                "comparator": comparator,
                "N": int(len(shared)),
                "median_paired_difference": float(np.median(difference)),
                "ci_low": low,
                "ci_high": high,
                "reference_improvement_fraction": float(np.mean(difference > 0)),
            }
        )
    return pd.DataFrame(rows)


# ==========================================================================
# 11  Split-conformal predictive intervals               (Methods 2.5)
# ==========================================================================
#
# Calibrated on the held-out calibration rows of each outer-training fold,
# which are disjoint from the outer-test curves.  A central 68% residual band
# and an outer interval targeting 90% whole-suffix coverage are reported.
#
# SCOPE NOTE.  This is plain split-conformal calibration.  The interval numbers
# published in the manuscript (87.8% coverage at 20% reveal, 91.7% at 30%) came
# from a heavier composite - a material-stratified Bayesian residual posterior
# combined with a support-adaptive conformal quantile and tail weighting - in
# `new fit/run_three_model_uncertainty_study.py`, via
# `run_bayesian_curve_uncertainty.py` and
# `run_bayesian_conformal_support_uncertainty.py`.  That construction is not
# reproduced here.  The bands below are the correct nonconformity score and
# calibration split, and they reproduce the qualitative finding - coverage near
# the nominal target with wide intervals - but they should not be quoted as the
# published coverage figures.


def conformal_quantile(values, coverage=CONFORMAL_COVERAGE) -> float:
    """Finite-sample conformal quantile with the (n+1) correction."""
    values = np.sort(np.asarray(values, float)[np.isfinite(values)])
    if not len(values):
        return np.nan
    rank = min(int(np.ceil((len(values) + 1) * coverage)), len(values))
    return float(values[rank - 1])


def calibration_scores(calibration_indices, centers, curve_cache, boundaries) -> np.ndarray:
    """Whole-suffix nonconformity: max scaled |residual| over the withheld part."""
    scores = []
    for idx in calibration_indices:
        if idx not in centers:
            continue
        cache = curve_cache[idx]
        center = np.asarray(centers[idx], float)
        time = np.asarray(cache["time"], float)
        actual = np.asarray(cache["shifted"], float)
        keep = time > float(boundaries[idx])
        if np.count_nonzero(keep) < MIN_SUFFIX_POINTS:
            continue
        scale = max(float(np.ptp(center)), abs(float(center[-1])), EPS)
        scores.append(float(np.max(np.abs(center[keep] - actual[keep])) / scale))
    return np.asarray(scores, float)


def conformal_band(center, quantile) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric band scaled by the forecast's own magnitude, floored at zero."""
    center = np.asarray(center, float)
    scale = max(float(np.ptp(center)), abs(float(center[-1])), EPS)
    half_width = quantile * scale
    return np.maximum(center - half_width, 0.0), center + half_width


def interval_metrics(actual, lower, upper, time, boundary, denominator) -> dict:
    keep = np.asarray(time, float) > float(boundary)
    if np.count_nonzero(keep) < MIN_SUFFIX_POINTS:
        return {}
    actual = np.asarray(actual, float)[keep]
    lower = np.asarray(lower, float)[keep]
    upper = np.asarray(upper, float)[keep]
    inside = (actual >= lower) & (actual <= upper)
    return {
        "pointwise_coverage": float(np.mean(inside)),
        "whole_suffix_covered": bool(np.all(inside)),
        "relative_width": float(np.mean(upper - lower) / max(denominator, EPS)),
    }


# ==========================================================================
# 12  Orchestration
# ==========================================================================


MODEL_LABELS = {
    "condition_only": "M (conditions only)",
    "theta_t0_condition": "T0 (theta, conditions only)",
    "theta_t1_prefix": "T1 (endpoint-normalized theta, prefix)",
    "m2_matched_ml": "M2 (matched prefix ML)",
    "h3_prefix_hybrid": "H3 (physics-guided hybrid)",
}


def run_theta_fit_stage() -> pd.DataFrame:
    """Sections 2-3: assemble records, fit theta, screen, build modeling table."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    records = discover_records()
    print(f"Assembled creep records: {len(records)}")
    print(records.groupby("material").size().to_string())

    summary, failures = fit_all_theta_curves(records)
    print(f"Theta fits completed: {len(summary)}; failures: {len(failures)}")

    retained, excluded = apply_quality_screen(summary)
    print(f"Retained after R2 >= {MIN_FIT_R2} and locked exclusions: {len(retained)}")
    print(retained.groupby("material").size().to_string())

    with pd.ExcelWriter(THETA_SUMMARY_FILE) as writer:
        summary.to_excel(writer, sheet_name="theta_summary_all", index=False)
        retained.to_excel(writer, sheet_name="all_quality_data", index=False)
        excluded.to_excel(writer, sheet_name="excluded", index=False)
        if len(failures):
            failures.to_excel(writer, sheet_name="fit_failures", index=False)
    print(f"Wrote {THETA_SUMMARY_FILE}")
    return retained


def load_modeling_data() -> tuple[pd.DataFrame, dict]:
    """Load the screened cohort and build its normalized-shape curve cache."""
    if not THETA_SUMMARY_FILE.exists():
        raise FileNotFoundError(
            f"{THETA_SUMMARY_FILE} not found - run `--stage theta-fit` first."
        )
    retained = pd.read_excel(THETA_SUMMARY_FILE, sheet_name="all_quality_data")
    data = build_modeling_table(retained)

    curve_cache, failed = load_curve_cache(data)
    if len(failed):
        print(f"Curves dropped during shape normalization: {len(failed)}")
    data = data.loc[list(curve_cache.keys())].copy()
    curve_cache = downsample_curve_cache(curve_cache)

    print(f"Modeling cohort: {len(data)} curves")
    print(data.groupby("material").size().to_string())
    return data, curve_cache


def run_forecast_stage(reveals=PRIMARY_REVEALS) -> dict[str, pd.DataFrame]:
    """Sections 5-11: the full grouped cross-validated forecast comparison."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    data, curve_cache = load_modeling_data()

    curve_rows: list[dict] = []
    conformal_rows: list[dict] = []

    for scheme, splitter, groups in split_schemes(data):
        if scheme != SPLIT_SCHEME:
            continue

        for fold, (outer_pos, test_pos) in enumerate(splitter.split(data, groups=groups), 1):
            outer, test = data.iloc[outer_pos].copy(), data.iloc[test_pos].copy()
            proper_pos, calibration_pos = stratified_proper_calibration_split(
                outer, SEED + 100 * FROZEN_EXPORT_SCHEME_NUMBER + fold
            )
            proper = outer.iloc[proper_pos].copy()
            calibration = outer.iloc[calibration_pos].copy()
            target = pd.concat([calibration, test])
            print(f"fold {fold}: proper={len(proper)} calibration={len(calibration)} test={len(test)}",
                  flush=True)

            # Model M and T0 use loading conditions only and do not depend on
            # the reveal fraction, so they are fitted once per fold.
            _m_state, m_centers, _m_lives = fit_model_M(proper, target, curve_cache)
            t0_oof, t0_target = fit_theta_oof_and_target(
                proper, target, proper, target, CONDITION_FEATURES, THETA_SEED + 1000 * fold
            )
            t0_target = add_theta_descriptors(t0_target)

            for reveal in reveals:
                featured_train, featured_target = build_featured(
                    proper, target, curve_cache, reveal
                )
                p5_train = inner_oof_p5(proper, featured_train, curve_cache)
                p5_target, _p5_centers, shapes = direct_p5(
                    proper, target, featured_target, curve_cache
                )

                _h3_state, h3_centers = fit_model_H3(
                    proper, target, featured_train, featured_target,
                    p5_train, p5_target, shapes, curve_cache,
                )
                _m2_state, m2_centers = fit_model_M2(
                    proper, target, featured_train, featured_target, curve_cache
                )

                t1_features = list(dict.fromkeys(CONDITION_FEATURES + H3_FEATURES))
                _t1_oof, t1_target = fit_theta_oof_and_target(
                    proper, target,
                    featured_train.join(p5_train), featured_target.join(p5_target),
                    t1_features,
                    THETA_SEED + 10000 * fold + int(1000 * reveal),
                )
                t1_target = add_theta_descriptors(t1_target)

                centers_by_model = {
                    "condition_only": m_centers,
                    "m2_matched_ml": m2_centers,
                    "h3_prefix_hybrid": h3_centers,
                    "theta_t0_condition": {
                        idx: theta_center(t0_target.loc[idx], curve_cache[idx]["time"])
                        for idx in target.index
                    },
                    "theta_t1_prefix": {
                        idx: theta_center(t1_target.loc[idx], curve_cache[idx]["time"])
                        for idx in target.index
                    },
                }
                boundaries = {
                    idx: float(featured_target.loc[idx, "reveal_time"]) for idx in target.index
                }

                # Score the outer-test curves only.
                for idx, row in test.iterrows():
                    cache = curve_cache[idx]
                    boundary = boundaries[idx]
                    outside_hull = hull_status(proper, row)
                    for model, centers in centers_by_model.items():
                        score = suffix_score(
                            cache["shifted"], centers[idx], cache["time"], boundary
                        )
                        if score is None:
                            continue
                        late = late_life_scores(
                            cache["shifted"], centers[idx], cache["time"],
                            boundary, cache["actual_rupture_time"],
                        )
                        curve_rows.append(
                            {
                                "fold": fold,
                                "reveal_fraction": float(reveal),
                                "model": model,
                                "model_label": MODEL_LABELS[model],
                                "relative_path": row["relative_path"],
                                "material": row["material"],
                                "condition_id": row["condition_id"],
                                "temperature": row["temperature"],
                                "stress": row["stress"],
                                "actual_rupture_time": cache["actual_rupture_time"],
                                "observation_boundary_hours": boundary,
                                "observed_fraction_of_actual_life":
                                    boundary / max(cache["actual_rupture_time"], EPS),
                                "outside_convex_hull": outside_hull,
                                **score,
                                **late,
                            }
                        )

                # Split-conformal calibration on the calibration rows only.
                for model in ("h3_prefix_hybrid", "theta_t1_prefix"):
                    scores = calibration_scores(
                        calibration.index, centers_by_model[model], curve_cache, boundaries
                    )
                    if not len(scores):
                        continue
                    q_outer = conformal_quantile(scores, CONFORMAL_COVERAGE)
                    q_central = conformal_quantile(scores, CONFORMAL_CENTRAL)
                    for idx, row in test.iterrows():
                        cache = curve_cache[idx]
                        denominator = max(float(np.ptp(cache["shifted"])), EPS)
                        lower, upper = conformal_band(centers_by_model[model][idx], q_outer)
                        metrics = interval_metrics(
                            cache["shifted"], lower, upper, cache["time"],
                            boundaries[idx], denominator,
                        )
                        if not metrics:
                            continue
                        conformal_rows.append(
                            {
                                "fold": fold,
                                "reveal_fraction": float(reveal),
                                "model": model,
                                "relative_path": row["relative_path"],
                                "material": row["material"],
                                "condition_id": row["condition_id"],
                                "q_outer": q_outer,
                                "q_central": q_central,
                                **metrics,
                            }
                        )

    curve_level = pd.DataFrame(curve_rows)
    conformal = pd.DataFrame(conformal_rows)

    # Restrict to the matched cohort: curves scored by every model at a reveal.
    matched = []
    for reveal, group in curve_level.groupby("reveal_fraction"):
        counts = group.groupby("relative_path")["model"].nunique()
        complete = counts[counts.eq(len(MODEL_LABELS))].index
        matched.append(group[group["relative_path"].isin(complete)])
    matched_level = pd.concat(matched) if matched else curve_level

    summary = summarize_model(matched_level)
    paired = pd.concat(
        [
            paired_comparison(matched_level, "h3_prefix_hybrid", comparator)
            for comparator in ("condition_only", "m2_matched_ml", "theta_t1_prefix")
        ],
        ignore_index=True,
    ) if len(matched_level) else pd.DataFrame()

    support = (
        matched_level[matched_level["model"].eq("h3_prefix_hybrid")]
        .groupby(["reveal_fraction", "outside_convex_hull"])
        .agg(
            N=("remainder_nrmse", "size"),
            median_nrmse=("remainder_nrmse", "median"),
            p95_nrmse=("remainder_nrmse", lambda v: float(np.percentile(v, 95))),
            severe_rate=("remainder_nrmse", lambda v: float(np.mean(v > 1.0))),
        )
        .reset_index()
        if len(matched_level) else pd.DataFrame()
    )

    coverage = (
        conformal.groupby(["reveal_fraction", "model"])
        .agg(
            N=("whole_suffix_covered", "size"),
            whole_suffix_coverage=("whole_suffix_covered", "mean"),
            pointwise_coverage=("pointwise_coverage", "mean"),
            median_relative_width=("relative_width", "median"),
        )
        .reset_index()
        if len(conformal) else pd.DataFrame()
    )

    tables = {
        "curve_level": curve_level,
        "matched_curve_level": matched_level,
        "model_summary": summary,
        "paired_comparisons": paired,
        "condition_support": support,
        "conformal_coverage": coverage,
    }

    book = OUTPUT_DIR / "master_model_results.xlsx"
    with pd.ExcelWriter(book) as writer:
        for name, frame in tables.items():
            if len(frame):
                frame.to_excel(writer, sheet_name=name[:31], index=False)
    print(f"\nWrote {book}")

    if len(summary):
        print("\nModel summary:")
        print(summary.to_string(index=False))
    if len(paired):
        print("\nPaired comparisons (positive favours H3):")
        print(paired.to_string(index=False))

    return tables


def run_shape_stage() -> pd.DataFrame:
    """Section 4: export per-specimen normalized shapes and material medians."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    data, curve_cache = load_modeling_data()
    shapes = material_normalized_shape_table(data, curve_cache)

    shape_columns = [column for column in shapes.columns if column.startswith("S_")]
    medians = shapes.groupby("material")[shape_columns].median().reset_index()

    shapes.to_csv(OUTPUT_DIR / "material_normalized_shapes.csv.gz", index=False)
    medians.to_csv(OUTPUT_DIR / "material_normalized_shape_medians.csv", index=False)

    print("\nMedian accumulated strain fraction S(x) by material:")
    for x in (0.2, 0.5, 0.8):
        column = f"S_{x:.3f}"
        print(f"  x = {x:.1f}: " + ", ".join(
            f"{row['material']} {row[column]:.3f}" for _, row in medians.iterrows()
        ))
    return shapes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--stage",
        choices=["all", "theta-fit", "shapes", "forecast"],
        default="all",
        help="Which part of the pipeline to run.",
    )
    parser.add_argument(
        "--reveals",
        type=float,
        nargs="+",
        default=list(PRIMARY_REVEALS),
        help="Nominal reveal fractions to evaluate (default: 0.20 0.30).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage in {"all", "theta-fit"}:
        run_theta_fit_stage()
    if args.stage in {"all", "shapes"}:
        run_shape_stage()
    if args.stage in {"all", "forecast"}:
        run_forecast_stage(tuple(args.reveals))


if __name__ == "__main__":
    main()
