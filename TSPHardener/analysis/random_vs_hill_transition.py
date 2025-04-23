"""
plot_combined.py
Combine the trend line from control_vs_hard_iter
with the scatter from phase_transition.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from analysis_util.load_json import (
    load_all_hard_instances,
    load_phase_transition_iterations,
)

def plot_combined(
    dist: str = "uniform",
    city_size: int = 20,
    tsp_type: str = "euclidean",
    mutation_types: list[str] | None = None,  # None ➜ use *all* mutation types
) -> None:
    # --- load & filter -------------------------------------------------------
    hard_df  = load_all_hard_instances()
    phase_df = load_phase_transition_iterations()

    hard_df  = hard_df[
        (hard_df["distribution"] == dist)
        & hard_df["range"].notna()
        & (hard_df["iterations"] > 0)
        & (hard_df["city_size"] == city_size)
        & (hard_df["generation_type"] == tsp_type)
    ]
    if mutation_types:
        hard_df = hard_df[hard_df["mutation_type"].isin(mutation_types)]

    phase_df = phase_df[
        (phase_df["distribution"] == dist)
        & phase_df["range"].notna()
        & (phase_df["iteration"] > 0)
        & (phase_df["city_size"] == city_size)
        & (phase_df["generation_type"] == tsp_type)
    ]

    if hard_df.empty or phase_df.empty:
        print("No data left after filtering – adjust parameters?")
        return

    # --- trend line (from hardest‑instance data) ----------------------------
    stats_df = (
        hard_df.groupby("range")["iterations"]
        .agg(["max", "median", "mean", "std"])
        .reset_index()
        .sort_values("range")
    )
    
    coeffs = np.polyfit(hard_df["range"], hard_df["iterations"], 1)
    x_line = np.linspace(hard_df["range"].min(), hard_df["range"].max(), 200)
    y_line = np.polyval(coeffs, x_line)

    # --- plotting -----------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.scatter(
        phase_df["range"],
        phase_df["iteration"],
        alpha=0.35,
        color="grey",
        label="Phase‑transition instances",
    )
    plt.plot(
        stats_df["range"],
        stats_df["mean"],
        marker="o",
        linestyle="",
        markersize=6,
        label="Mean of hill-climbed hardest instances",
    )
    #    –– error bars (± 1 std); comment out if you only want the points
    plt.errorbar(
        stats_df["range"],
        stats_df["mean"],
        yerr=stats_df["std"],
        fmt="none",
        capsize=3,
        elinewidth=1,
        alpha=0.8,
    )
    plt.plot(
        x_line,
        y_line,
        linewidth=2,
        color="red",
        label="Linear trend (hill-climbed hardest)",
    )
    
    xlabel = r"$rand_{max}$" if dist == "uniform" else r"$\sigma$"
    plt.xlabel(xlabel)
    plt.ylabel("Iterations")
    plt.title(
        f"Trend vs Phase Transition  •  size={city_size}, {tsp_type}, {dist}"
    )
    plt.grid(True)
    plt.legend()

    out_dir  = "./plot/random_sampling_vs_hill_climber"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/trend_vs_phase_{dist}_{tsp_type}_{city_size}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_path}")

# quick demo:
if __name__ == "__main__":
    plot_combined()                          # default: uniform / 20 / EUC
    plot_combined("uniform", 30, "asymmetric")
    plot_combined("lognormal", 20, "euclidean")
    plot_combined("lognormal", 30, "euclidean")
