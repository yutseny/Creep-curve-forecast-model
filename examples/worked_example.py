"""Worked example: updating a forecast as an ongoing creep test progresses.

A completed Alloy 617 test (750 degC, 121 MPa) is replayed as if it were still
running.  Every replicate at that condition is removed from training, so the
model has never seen this condition, which matches the grouped cross-validation
in the manuscript.  The test is then "checked in" on three times, and at each
check-in only the strain measured so far is given to the forecaster.  Because
the test is in fact complete, each forecast can be scored against the withheld
remainder of the curve.

Run from the repository root, after `python master_model.py --stage theta-fit`:

    python examples/worked_example.py

It takes a few minutes: about 1-2 min to load the cohort, then about 1 min per
check-in.  Outputs go to examples/output/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import master_model as mm  # noqa: E402
from creep_forecast import CreepForecaster, format_summary  # noqa: E402

RECORD = "A617/750_121_1.csv"
# Check-ins as fractions of the condition-only predicted life, inside the
# validated reveal range 0.10-0.50.
CHECK_INS = (0.15, 0.30, 0.50)
OUT = REPO / "examples" / "output"


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. Load the screened cohort once; it is reused for every forecast.
    data, curve_cache = mm.load_modeling_data()
    record = data[data["relative_path"].eq(RECORD)].iloc[0]
    material = record["material"]
    temperature, stress = float(record["temperature"]), float(record["stress"])
    heat = record["heat_treatment"] if material == "H282_w" else None

    # 2. Hold out every replicate at this condition.
    forecaster = CreepForecaster(data, curve_cache,
                                 exclude_conditions=[record["condition_id"]])
    print(f"Held out condition {record['condition_id']}: "
          f"{len(data) - len(forecaster.data)} curves; "
          f"training on {len(forecaster.proper)} + calibrating on "
          f"{len(forecaster.calibration)}")

    # 3. The full measured curve, used only to replay the test and to score it.
    time, strain = mm.read_raw_curve(record["relative_path"], material)
    life_cond = forecaster.condition_only_life(material, temperature, stress, heat)
    print(f"Condition-only life estimate: {life_cond:.0f} h "
          f"(measured: {record['rupture_time']:.0f} h)\n")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(CHECK_INS), figsize=(4.6 * len(CHECK_INS), 4.4),
                             sharey=True)
    rows = []
    for ax, fraction in zip(axes, CHECK_INS):
        # 4. What the lab has measured by this check-in.
        observed = time <= fraction * life_cond
        result = forecaster.forecast(material, temperature, stress,
                                     time[observed], strain[observed],
                                     heat_treatment=heat)
        s = result.summary
        print(f"--- check-in at {s['observation_boundary_h']:.0f} h")
        print(format_summary(s))

        # 5. Score against the withheld remainder (possible only in hindsight).
        predicted = np.interp(time, result.curve["time_h"], result.curve["strain_percent"])
        score = mm.suffix_score(strain, predicted, time, s["observation_boundary_h"])
        lower = np.interp(time, result.curve["time_h"], result.curve["lower_90"])
        upper = np.interp(time, result.curve["time_h"], result.curve["upper_90"])
        suffix = time > s["observation_boundary_h"]
        covered = float(np.mean((strain[suffix] >= lower[suffix]) & (strain[suffix] <= upper[suffix])))
        print(f"  remainder NRMSE (Eq. 11): {score['remainder_nrmse']:.3f}; "
              f"90% band covers {100 * covered:.0f}% of withheld points\n")

        lo, hi = s["rupture_life_interval_90_h"]
        rows.append({
            "check_in_h": s["observation_boundary_h"],
            "reveal_fraction": s["reveal_fraction"],
            "strain_so_far_percent": s["strain_at_boundary_percent"],
            "predicted_life_h": s["predicted_rupture_life_h"],
            "life_90_low_h": lo,
            "life_90_high_h": hi,
            "measured_life_h": float(record["rupture_time"]),
            "predicted_final_strain_percent": s["predicted_final_strain_percent"],
            "measured_final_strain_percent": float(strain[-1]),
            "remainder_nrmse": score["remainder_nrmse"],
            "band90_pointwise_coverage": covered,
        })
        result.save(OUT / f"check_in_{int(round(100 * fraction)):02d}", plot=False)
        result.plot(ax=ax, actual=(time, strain), ylim=2.0 * float(strain[-1]))
        ax.set_title(f"Check-in at {s['observation_boundary_h']:.0f} h "
                     f"(r = {s['reveal_fraction']:.2f})", fontsize=10)
        ax.get_legend().remove()
        if ax is not axes[0]:
            ax.set_ylabel("")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8,
               frameon=False)
    xmax = max(ax.get_xlim()[1] for ax in axes)
    for ax in axes:
        ax.set_xlim(0, xmax)
    fig.suptitle(f"{material}, {temperature:g} °C, {stress:g} MPa - condition held out "
                 "of training", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "worked_example.png", dpi=150)

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "worked_example_summary.csv", index=False)
    print(table.round(3).to_string(index=False))
    print(f"\nWrote {OUT / 'worked_example.png'} and {OUT / 'worked_example_summary.csv'}")


if __name__ == "__main__":
    main()
