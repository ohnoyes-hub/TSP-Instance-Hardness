import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from util.load_experiment import load_hill_climb_iterations

def histogram_hill_climbing(dist='uniform'):
    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={
            "figure.figsize": (14, 9),
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
        }
    )

    # Replace this with the actual function to load hill climbing data
    df = load_hill_climb_iterations()  # <-- update this line

    # Filter for the relevant distribution and valid iterations
    filtered_df = df[
        (df['distribution'] == dist) &
        (df['iteration'] > 0)
    ]

    if filtered_df.empty:
        print("No data found for distribution:", dist)
        return

    city_sizes = [20, 30]
    tsp_types = ['euclidean', 'asymmetric']
    mutation_types = filtered_df['mutation_type'].unique()
    output_dir = "./plot/histograms_hill_climbing/"
    os.makedirs(output_dir, exist_ok=True)

    for tsp_type in tsp_types:
        for size in city_sizes:
            for mutation in mutation_types:
                subset = filtered_df[
                    (filtered_df['generation_type'] == tsp_type) &
                    (filtered_df['city_size'] == size) &
                    (filtered_df['mutation_type'] == mutation)
                ]
                if subset.empty:
                    continue

                subset['log_iteration'] = np.log(subset['iteration'])
                mean_iter = subset['log_iteration'].mean()
                median_iter = subset['log_iteration'].median()

                plt.figure()
                ax = sns.histplot(
                    data=subset,
                    x='log_iteration',
                    bins='auto',
                    # kde=True,
                    element='bars',
                    alpha=0.7,
                    stat='count'
                )
                plt.title(
                    f"Log Lital Iterations for {tsp_type[0].capitalize()}TSP ({mutation.title()})\n"
                    f"City size {size}, {dist.capitalize()} Distribution"
                )
                plt.xlabel("Log (Lital Iterations)")
                plt.ylabel("Frequency")
                plt.axvline(mean_iter, color='r', linestyle='--', label=f'Log-Mean: {mean_iter:.2f}')
                plt.axvline(median_iter, color='g', linestyle='-', label=f'Log-Median: {median_iter:.2f}')
                ax.legend(loc='best')
                plt.tight_layout()
                plt.savefig(
                    os.path.join(output_dir, f"hist_{dist}_{tsp_type}_size{size}_{mutation}.png")
                )
                plt.close()

    print("All histograms saved to:", output_dir)

def histogram_hill_climbing_side_by_side(dist='uniform'):
    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={
            "figure.figsize": (14, 9),
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
        }
    )

    df = load_hill_climb_iterations()

    # Filter for the relevant distribution and valid iterations
    filtered_df = df[
        (df['distribution'] == dist) &
        (df['iteration'] > 0)
    ]

    if filtered_df.empty:
        print("No data found for distribution:", dist)
        return

    city_sizes = [20, 30]
    tsp_types = ['euclidean', 'asymmetric']
    mutation_types = filtered_df['mutation_type'].unique()

    output_dir = "./plot/histograms_hill_climbing_side_by_side/"
    os.makedirs(output_dir, exist_ok=True)

    for tsp_type in tsp_types:
        for mutation in mutation_types:
            data_frames = []
            for size in city_sizes:
                subset = filtered_df[
                    (filtered_df['generation_type'] == tsp_type) &
                    (df['city_size'] == size) &
                    (df['mutation_type'] == mutation)
                ].copy()
                if not subset.empty:
                    subset['log_iteration'] = np.log(subset['iteration'])
                    subset['city_size'] = str(size)
                    data_frames.append(subset)
            
            if not data_frames:
                continue

            combined_df = pd.concat(data_frames, ignore_index=True) 

            plt.figure()
            ax = sns.histplot(
                data=combined_df,
                x='log_iteration',
                hue='city_size',
                bins='auto',
                # kde=True,
                element='step',
                stat='count',
                common_norm=False,
                alpha=0.6
            )
            plt.title(
                f"Log Lital Iterations for {tsp_type[0].capitalize()}TSP ({mutation.title()})\n"
                f"{dist.capitalize()} Distribution"
            )
            plt.xlabel("Log (Lital Iterations)")
            plt.ylabel("Frequency")
            plt.legend(title="City Size", loc='best')
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"hist_{dist}_{tsp_type}_{mutation}_side_by_side.png")
            )
            plt.close()

    print("All side-by-side histograms saved to:", output_dir)

def histogram_hill_climbing_combined(dist='uniform'):
    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={
            "figure.figsize": (24, 16),
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )
    df = load_hill_climb_iterations()
    filtered_df = df[
        (df['distribution'] == dist) &
        (df['iteration'] > 0)
    ]
    if filtered_df.empty:
        print("No data found for distribution:", dist)
        return

    city_sizes = [20, 30]
    tsp_types = ['euclidean', 'asymmetric']
    mutation_types = list(filtered_df['mutation_type'].unique())
    output_dir = "./plot/histograms_hill_climbing_combined/"
    os.makedirs(output_dir, exist_ok=True)

    for tsp_type in tsp_types:
        fig, axes = plt.subplots(
            nrows=2, ncols=len(mutation_types), 
            figsize=(7 * len(mutation_types), 12),
            sharex=True, sharey=True
        )
        if len(mutation_types) == 1:
            axes = np.array([[axes[0]], [axes[1]]])  # fix shape for single mut type

        for row, size in enumerate(city_sizes):
            for col, mutation in enumerate(mutation_types):
                ax = axes[row, col]
                subset = filtered_df[
                    (filtered_df['generation_type'] == tsp_type) &
                    (filtered_df['city_size'] == size) &
                    (filtered_df['mutation_type'] == mutation)
                ].copy()
                if subset.empty:
                    ax.set_visible(False)
                    continue

                subset['log_iteration'] = np.log(subset['iteration'])
                sns.histplot(
                    data=subset,
                    x='log_iteration',
                    bins='auto',
                    # kde=True,
                    ax=ax,
                    stat='count',
                    alpha=0.65,
                    element='bars'
                )

                mean_iter = subset['log_iteration'].mean()
                median_iter = subset['log_iteration'].median()
                ax.axvline(mean_iter, color='r', linestyle='--', label=f'Log-Mean: {mean_iter:.2f}')
                ax.axvline(median_iter, color='g', linestyle='-', label=f'Log-Median: {median_iter:.2f}')
                
                # Inset label (top left)
                inset_text = f"City: {size}\nMutation: {mutation.title()}"
                ax.text(
                    0.05, 0.95, inset_text,
                    ha='left', va='top',
                    transform=ax.transAxes,
                    fontsize=15,
                    fontweight='bold',
                    bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.2', alpha=0.7),
                    zorder=10
                )

                # Remove x-label for top row
                if row == 0:
                    ax.set_xlabel("")
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel("Log (Lital Iterations)", fontsize=16)
                
                # Remove y-label for all but leftmost
                if col == 0:
                    ax.set_ylabel("Frequency", fontsize=16)
                else:
                    ax.set_ylabel("")

                # Legend only in bottom-right
                if row == 1 and col == len(mutation_types) - 1:
                    ax.legend(loc='best', fontsize=12)
                else:
                    ax.get_legend().remove() if ax.get_legend() else None

        # Suptitle and layout
        fig.suptitle(f"{tsp_type[0].capitalize()}TSP Hill Climbing\n{dist.capitalize()} Distribution", fontsize=24, y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        save_path = os.path.join(output_dir, f"hist_combined_{dist}_{tsp_type}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print("Saved:", save_path)

if __name__ == "__main__":
    # histogram_hill_climbing('uniform')
    # histogram_hill_climbing('lognormal')
    histogram_hill_climbing_side_by_side('uniform')
    histogram_hill_climbing_side_by_side('lognormal')
    histogram_hill_climbing_combined('uniform')
    histogram_hill_climbing_combined('lognormal')

