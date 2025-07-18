#!/usr/bin/env python3
"""
Create violin plots of iteration counts for phase‐transition experiments:
  1. For each TSP type, compare across city sizes.
  2. For each city size, compare across TSP types.
Saves each plot under ./plot/violin_random_sampling/.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from util.load_experiment import load_phase_transition_iterations

def preprocess_iterations(df, norm_method='per_city', robust=False):
    """
    Filter out bad data, then add a new column 'iteration_scaled' that is:
      - if norm_method=='per_city': iteration / city_size
      - if norm_method=='per_edge': iteration / (city_size*(city_size-1)/2)
      - if norm_method=='none': raw iteration
    If robust=True, then standardize via (x - median)/(IQR).
    Otherwise, if norm_method!='none' and robust=False, leave as-is after division.
    """
    # drop non-positive
    df = df[df['iteration'] > 0].copy()

    # choose base scaling
    if norm_method == 'per_city':
        df['iteration_scaled'] = df['iteration'] / df['city_size']
    elif norm_method == 'per_edge':
        df['iteration_scaled'] = df['iteration'] / (df['city_size']*(df['city_size']-1)/2)
    else:
        df['iteration_scaled'] = df['iteration'].astype(float)

    # optional robust standardization
    if robust:
        def rob_std(x):
            med = x.median()
            iqr = x.quantile(0.75) - x.quantile(0.25)
            return (x - med) / (iqr if iqr>0 else x.std())
        df['iteration_scaled'] = df.groupby('city_size')['iteration_scaled'] \
                                     .transform(rob_std)
    return df


def plot_violin_by_city_size(df, tsp_type, city_sizes, dist, output_dir):
    """
    For a given TSP type, draw a violin plot of 'iteration' vs. 'city_size',
    add a quartile legend with actual values, and save both full and zoomed-to-95th-percentile versions.
    """
    subset = df[df['generation_type'] == tsp_type]
    if subset.empty:
        print(f"[!] No data for TSP type: {tsp_type}")
        return

    # Compute quartiles
    q1 = np.percentile(subset['iteration'], 25)
    median = np.percentile(subset['iteration'], 50)
    q3 = np.percentile(subset['iteration'], 75)

    plt.figure(figsize=(10, 6))
    ax = sns.violinplot(
        data=subset,
        x='city_size',
        y='iteration',
        inner='quartile',
        palette='muted',
        cut=0
    )
    # Add quartile legend with computed values
    q1_line = mlines.Line2D([], [], color='k', linestyle='-', label=f'25th percentile ({int(q1)})')
    median_line = mlines.Line2D([], [], color='k', linestyle='-', linewidth=2, label=f'50th percentile ({int(median)})')
    q3_line = mlines.Line2D([], [], color='k', linestyle='--', label=f'75th percentile ({int(q3)})')
    ax.legend(handles=[q1_line, median_line, q3_line], title='Quartiles')

    plt.title(f"Violin plot of iterations for {tsp_type.capitalize()} TSP\nDistribution: {dist}")
    plt.xlabel("City Size")
    plt.ylabel("Number of Iterations")
    plt.tight_layout()
    # Save full range
    fname = os.path.join(output_dir, f"violin_by_size_{dist}_{tsp_type}.png")
    plt.savefig(fname)
    print(f"  → Saved {fname}")

    # Zoom to 95th percentile
    p95 = np.percentile(subset['iteration'], 95)
    plt.ylim(0, p95)
    plt.title(f"Violin plot of iterations for {tsp_type.capitalize()} TSP\nDistribution: {dist} (Zoom to 95th percentile)")
    fname_zoom = os.path.join(output_dir, f"violin_by_size_{dist}_{tsp_type}_zoom95.png")
    plt.savefig(fname_zoom)
    plt.close()
    print(f"  → Saved {fname_zoom}")


def plot_violin_by_tsp_type(df, size, tsp_types, dist, output_dir):
    """
    For a given city size, draw a violin plot of 'iteration' vs. 'generation_type',
    add a quartile legend with actual values, and save both full and zoomed-to-95th-percentile versions.
    """
    subset = df[df['city_size'] == size]
    if subset.empty:
        print(f"[!] No data for city size: {size}")
        return

    # Compute quartiles
    q1 = np.percentile(subset['iteration'], 25)
    median = np.percentile(subset['iteration'], 50)
    q3 = np.percentile(subset['iteration'], 75)

    plt.figure(figsize=(10, 6))
    ax = sns.violinplot(
        data=subset,
        x='generation_type',
        y='iteration',
        inner='quartile',
        palette='pastel',
        cut=0
    )
    # Add quartile legend with computed values
    q1_line = mlines.Line2D([], [], color='k', linestyle='-', label=f'25th percentile ({int(q1)})')
    median_line = mlines.Line2D([], [], color='k', linestyle='-', linewidth=2, label=f'50th percentile ({int(median)})')
    q3_line = mlines.Line2D([], [], color='k', linestyle='--', label=f'75th percentile ({int(q3)})')
    ax.legend(handles=[q1_line, median_line, q3_line], title='Quartiles')

    plt.title(f"Violin plot of iterations for city size {size}\nDistribution: {dist}")
    plt.xlabel("TSP Type")
    plt.ylabel("Number of Iterations")
    plt.tight_layout()
    # Save full range
    fname = os.path.join(output_dir, f"violin_by_tsp_{dist}_size{size}.png")
    plt.savefig(fname)
    print(f"  → Saved {fname}")

    # Zoom to 95th percentile
    p95 = np.percentile(subset['iteration'], 95)
    plt.ylim(0, p95)
    plt.title(f"Violin plot of iterations for city size {size}\nDistribution: {dist} (Zoom to 95th percentile)")
    fname_zoom = os.path.join(output_dir, f"violin_by_tsp_{dist}_size{size}_zoom95.png")
    plt.savefig(fname_zoom)
    plt.close()
    print(f"  → Saved {fname_zoom}")


def main(dist: str = 'uniform'):
    # styling
    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={
            "figure.figsize": (12, 7),
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
        }
    )

    # load and filter
    df = load_phase_transition_iterations()
    filtered = df[
        (df['distribution'] == dist) &
        (df['range'].notna()) &
        (df['iteration'] > 0)
    ]
    if filtered.empty:
        print(f"[!] No data found for distribution '{dist}'")
        return

    city_sizes = [20, 30]
    tsp_types = ['euclidean', 'asymmetric']
    output_dir = "./plot/violin_random_sampling/"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating violin plots for distribution '{dist}'…")
    # 1) By city size for each TSP type
    for tsp in tsp_types:
        plot_violin_by_city_size(filtered, tsp, city_sizes, dist, output_dir)

    # 2) By TSP type for each city size
    for size in city_sizes:
        plot_violin_by_tsp_type(filtered, size, tsp_types, dist, output_dir)


if __name__ == "__main__":
    main()            # uniform
    main('lognormal')
