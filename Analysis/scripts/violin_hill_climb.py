#!/usr/bin/env python3
"""
Create violin plots of iteration counts for phase‐transition experiments,
annotating both median and mean on each violin.
Saves each plot under ./plot/violin_random_sampling/.
"""
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from util.load_experiment import load_hill_climb_iterations

def plot_violin_by_city_size(df, tsp_type, city_sizes, dist, output_dir):
    """
    For a given TSP type, draw a violin plot of 'iteration' vs. 'city_size',
    marking median and mean.
    """
    subset = df[df['generation_type'] == tsp_type]
    if subset.empty:
        print(f"[!] No data for TSP type: {tsp_type}")
        return

    plt.figure(figsize=(10, 6))
    ax = sns.violinplot(
        data=subset,
        x='city_size',
        y='iteration',
        inner=None,        # hide default quartiles
        palette='muted',
        cut=0
    )

    plt.title(f"Iterations for {tsp_type.capitalize()} TSP by City Size\n{dist.capitalize()} Distribution")
    plt.xlabel("City Size")
    plt.ylabel("Number of Iterations")
    plt.tight_layout()

    fname = os.path.join(output_dir, f"violin_by_size_{dist}_{tsp_type}.png")
    plt.savefig(fname)
    plt.close()
    print(f"  → Saved {fname}")

def plot_violin_by_tsp_type(df, size, tsp_types, dist, output_dir):
    """
    For a given city size, draw a violin plot of 'iteration' vs. 'generation_type',
    marking median and mean.
    """
    subset = df[df['city_size'] == size]
    if subset.empty:
        print(f"[!] No data for city size: {size}")
        return

    plt.figure(figsize=(10, 6))
    ax = sns.violinplot(
        data=subset,
        x='generation_type',
        y='iteration',
        inner=None,
        palette='pastel',
        cut=0
    )

    # Because x is categorical, seaborn maps them to positions 0,1,...
    xticks = ax.get_xticks()
    for i, t in enumerate(tsp_types):
        # compute and annotate at the numeric x position
        y = subset[subset['generation_type'] == t]['iteration']
        if y.empty:
            continue
        med = np.median(y)
        mean = np.mean(y)
        ax.scatter(xticks[i], med,
                   color='white', edgecolor='black',
                   s=100, zorder=3, marker='o')
        ax.scatter(xticks[i], mean,
                   color='black',
                   s=100, zorder=3, marker='D')

    # Add legend for median and mean
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Median',
                   markerfacecolor='white', markeredgecolor='black', markersize=10),
        plt.Line2D([0], [0], marker='D', color='w', label='Mean',
                   markerfacecolor='black', markersize=10)
    ]
    ax.legend(handles=legend_elements, loc='upper right')  # Adjust location as needed

    plt.title(f"Frequency of Lital Iterations for City Size {size} by TSP Type\n{dist.capitalize()} Distribution")
    plt.xlabel("TSP Type")
    plt.ylabel("Number of Iterations")
    plt.tight_layout()

    fname = os.path.join(output_dir, f"violin_by_tsp_{dist}_size{size}.png")
    plt.savefig(fname)
    plt.close()
    print(f"  => Saved {fname}")

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
    df = load_hill_climb_iterations()
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

    print(f"Generating violin plots (with mean & median) for distribution '{dist}'…")
    # 1) By city size for each TSP type
    for tsp in tsp_types:
        plot_violin_by_city_size(filtered, tsp, city_sizes, dist, output_dir)

    # 2) By TSP type for each city size
    for size in city_sizes:
        plot_violin_by_tsp_type(filtered, size, tsp_types, dist, output_dir)

if __name__ == "__main__":
    main()            # uniform
    main('lognormal')
