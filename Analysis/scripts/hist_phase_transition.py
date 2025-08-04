"""
The following creates histogram of Lital iterations under the random sampling experiment.
The purpose is to find the underlying "hardness" frequency of random sampler.
A comparison of city sizes and TSP type is also made to see differences in how TSP formulations affect hardness.
"""
import scipy.stats as stats
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from icecream import ic
from util.load_experiment import load_phase_transition_iterations


def plot_compare_city_sizes(df, tsp_type, city_sizes, dist, output_dir):
    """
    Overlay density histograms of 'iteration' for different city sizes for a given TSP type,
    using shared bins and consistent density scaling, including mean and median lines.
    """
    subset = df[df['generation_type'] == tsp_type]
    if subset.empty:
        ic("No data for TSP type:", tsp_type)
        return

    # shared bin edges across all sizes
    bins = np.histogram_bin_edges(subset['iteration'], bins='auto')
    plt.figure()

    for size in city_sizes:
        data = subset[subset['city_size'] == size]['iteration']
        if data.empty:
            ic(f"Skipping size {size}: no data")
            continue
        # plot density histogram as steps
        plt.hist(
            data,
            bins=bins,
            density=True,
            histtype='step',
            linewidth=2,
            label=f"Size {size}"
        )
        # add mean and median lines
        mean_val = data.mean()
        median_val = data.median()
        plt.axvline(
            mean_val,
            linestyle='--',
            linewidth=1.5,
            label=f"Mean size {size}: {mean_val:.2f}"
        )
        plt.axvline(
            median_val,
            linestyle=':',
            linewidth=1.5,
            label=f"Median size {size}: {median_val:.2f}"
        )

    plt.title(f"Density comparison of iterations for {tsp_type[0].capitalize()}TSP\nDistribution: {dist}")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Density")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"hist_compare_size_{dist}_{tsp_type}.png"))
    plt.close()

    # violin plot
    sns.violinplot(
        data=subset,
        x='city_size',
        y='iteration',
        inner='quartile',
        palette='muted',
        cut=0
    )
    plt.title(f"Violin plot of iterations for {tsp_type[0].capitalize()}TSP\nDistribution: {dist}")
    plt.xlabel("City Size")
    plt.ylabel("Number of Iterations")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"violin_{dist}_{tsp_type}.png"))
    plt.close()


def plot_compare_tsp_types(df, size, tsp_types, dist, output_dir):
    """
    Overlay density histograms of 'iteration' for different TSP types for a given city size,
    using shared bins and consistent density scaling, including mean and median lines.
    """
    subset = df[df['city_size'] == size]
    if subset.empty:
        ic("No data for city_size:", size)
        return

    # shared bin edges across all tsp types
    bins = np.histogram_bin_edges(subset['iteration'], bins='auto')
    plt.figure()

    for tsp in tsp_types:
        data = subset[subset['generation_type'] == tsp]['iteration']
        if data.empty:
            ic(f"Skipping tsp type {tsp}: no data")
            continue
        plt.hist(
            data,
            bins=bins,
            density=True,
            histtype='step',
            linewidth=2,
            label=f"{tsp.capitalize()}"
        )
        mean_val = data.mean()
        median_val = data.median()
        plt.axvline(
            mean_val,
            linestyle='--',
            linewidth=1.5,
            label=f"Mean {tsp}: {mean_val:.2f}"
        )
        plt.axvline(
            median_val,
            linestyle=':',
            linewidth=1.5,
            label=f"Median {tsp}: {median_val:.2f}"
        )

    plt.title(f"Density comparison of TSP types at city size {size}\nDistribution: {dist}")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Density")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"hist_compare_tsp_{dist}_size{size}.png"))
    plt.close()


def histogram_phase_transition(dist: str = 'uniform', log_scale: bool = True):
    # apply consistent styling
    sns.set_theme(
        style="whitegrid",
        context="talk",
        # palette="viridis",
        rc={
            "figure.figsize": (14, 9),
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
        }
    )

    df = load_phase_transition_iterations()
    # filter data
    filtered_df = df[
        (df['distribution'] == dist) &
        (df['range'].notna()) &
        (df['iteration'] > 0)
    ]

    if filtered_df.empty:
        ic("No data found for distribution:", dist)
        return

    city_sizes = [20, 30]
    tsp_types = ['euclidean', 'asymmetric', 'symmetric']
    output_dir = "./plot/histograms_random_sampling/"
    os.makedirs(output_dir, exist_ok=True)

    # individual histograms
    for tsp_type in tsp_types:
        for size in city_sizes:
            subset = filtered_df[
                (filtered_df['generation_type'] == tsp_type) &
                (filtered_df['city_size'] == size)
            ]
            if subset.empty:
                ic("No data for:", tsp_type, size)
                continue

            # mean_iter = subset['iteration'].mean()
            # median_iter = subset['iteration'].median()

            plt.figure()
            if log_scale:
                subset['plot_iteration'] = np.log(subset['iteration'])
                xlabel = "Log (Lital Iterations)"
            else:
                subset['plot_iteration'] = subset['iteration']
                xlabel = "Lital Iterations"

            mean_iter = subset['plot_iteration'].mean()
            median_iter = subset['plot_iteration'].median()

            ax = sns.histplot(
                data=subset,
                x='plot_iteration',
                bins='auto',
                # kde
                element='bars',
                alpha=0.7,
                stat='count'
            )
            plt.title(f"Frequency of {'Log ' if log_scale else ''}Lital Iterations for {tsp_type[0].capitalize()}TSP\n City size {size}, {dist.capitalize()} Distributed")            
            plt.ylabel("Frequency")
            plt.xlabel(xlabel)
            plt.axvline(mean_iter, color='r', linestyle='--', label=f'Log-Mean: {mean_iter:.2f}')
            plt.axvline(median_iter, color='g', linestyle='-', label=f'Log-Median: {median_iter:.2f}')
            ax.legend(loc='best')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"hist_{dist}_{tsp_type}_size{size}.png"))
            plt.close()

    # for tsp_type in tsp_types:
    #     plot_compare_city_sizes(filtered_df, tsp_type, city_sizes, dist, output_dir)

    # for size in city_sizes:
    #     plot_compare_tsp_types(filtered_df, size, tsp_types, dist, output_dir)

    ic("All histograms saved to:", output_dir)

def histogram_phase_transition_combined(dist: str = 'uniform', log_scale: bool = True):
    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={
            "figure.figsize": (18, 12),
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )

    df = load_phase_transition_iterations()
    filtered_df = df[
        (df['distribution'] == dist) &
        (df['range'].notna()) &
        (df['iteration'] > 0)
    ]
    if filtered_df.empty:
        print("No data found for distribution:", dist)
        return

    city_sizes = [20, 30]
    tsp_types = ['euclidean', 'asymmetric', 'symmetric']
    output_dir = "./plot/histograms_random_sampling_combined/"
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3,  sharex=True, sharey=True)
    # """figsize=(16, 12),"""

    for row, size in enumerate(city_sizes):
        for col, tsp_type in enumerate(tsp_types):
            ax = axes[row, col]
            subset = filtered_df[
                (filtered_df['generation_type'] == tsp_type) &
                (filtered_df['city_size'] == size)
            ].copy()
            if subset.empty:
                ax.set_visible(False)
                continue

            if log_scale:
                subset['plot_iteration'] = np.log(subset['iteration'])
                xlabel = "Log (Lital Iterations)"
            else:
                subset['plot_iteration'] = subset['iteration']
                xlabel = "Lital Iterations"
            # subset['log_iteration'] = np.log(subset['iteration'])
            sns.histplot(
                data=subset,
                x='plot_iteration',
                bins='auto',
                ax=ax,
                stat='count',
                alpha=0.7,
                element='bars'
            )

            mean_iter = subset['plot_iteration'].mean()
            median_iter = subset['plot_iteration'].median()
            ax.axvline(mean_iter, color='r', linestyle='--', label=f'{"Log-" if log_scale else ""}Mean: {mean_iter:.2f}')
            ax.axvline(median_iter, color='g', linestyle='-', label=f'{"Log-" if log_scale else ""}Median: {median_iter:.2f}')

            # Inset title (top left)
            inset_text = f"Random sampling\n{tsp_type[0].capitalize()}TSP\nCity: {size}\n{dist.capitalize()}"
            ax.text(
                0.05, 0.95, inset_text,
                ha='left', va='top',
                transform=ax.transAxes,
                fontsize=15,
                fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.2', alpha=0.7),
                zorder=10
            )

            # Remove x label for top row
            if row == 0:
                ax.set_xlabel("")
            else:
                ax.set_xlabel(xlabel, fontsize=16)


            # Remove y label for right column
            if col == 0:
                ax.set_ylabel("Frequency", fontsize=16)
            else:
                ax.set_ylabel("")

            ax.legend(loc='upper right', fontsize=10, frameon=True)
            # Only keep legend in bottom-right subplot
            # if row == 1 and col == 1:
            #     ax.legend(loc='best', fontsize=12)
            # else:
            #     ax.get_legend().remove() if ax.get_legend() else None

    # fig.suptitle(f"Random Sampling: Frequency of Log Lital Iterations\n{dist.capitalize()} Distribution", fontsize=24, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(output_dir, f"hist_combined_{dist}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print("Saved:", save_path)

def histogram_with_normal_fit(dist: str = 'uniform', log_scale: bool = True):
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

    df = load_phase_transition_iterations()

    filtered_df = df[
        (df['distribution'] == dist) &
        (df['range'].notna()) &
        (df['iteration'] > 0)
    ]

    city_sizes = [20, 30]
    tsp_types = ['euclidean', 'asymmetric', 'symmetric']
    output_dir = "./plot/histograms_normal_fit/"
    os.makedirs(output_dir, exist_ok=True)

    for tsp_type in tsp_types:
        for size in city_sizes:
            subset = filtered_df[
                (filtered_df['generation_type'] == tsp_type) &
                (filtered_df['city_size'] == size)
            ].copy()

            if subset.empty:
                print("No data for:", tsp_type, size)
                continue
            if log_scale:
                subset['plot_iteration'] = np.log(subset['iteration'])
                xlabel = "Log (Lital Iterations)"
            else:
                subset['plot_iteration'] = subset['iteration']
                xlabel = "Lital Iterations"

            plt.figure()

            # Plot histogram
            sns.histplot(
                subset['plot_iteration'],
                bins='auto',
                kde=False,
                stat='density',
                color='skyblue',
                alpha=0.6
            )

            # Fit normal distribution
            mu, std = stats.norm.fit(subset['plot_iteration'])

            # Plot normal distribution curve
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = stats.norm.pdf(x, mu, std)
            plt.plot(x, p, 'k', linewidth=2, label=f'Normal fit\n$\mu$={mu:.2f}, $\sigma$={std:.2f}')

            # Mean and median lines
            mean_iter = subset['plot_iteration'].mean()
            median_iter = subset['plot_iteration'].median()

            plt.axvline(mean_iter, color='r', linestyle='--', linewidth=1.5, label=f'Log-Mean: {mean_iter:.2f}')
            plt.axvline(median_iter, color='g', linestyle=':', linewidth=1.5, label=f'Log-Median: {median_iter:.2f}')

            plt.title(f"{xlabel} with Normal Fit\n{tsp_type.capitalize()} TSP, City Size: {size}, {dist.capitalize()} Distribution")
            plt.xlabel(xlabel)
            plt.ylabel("Density")
            plt.legend(loc='best')

            plt.tight_layout()
            save_path = os.path.join(output_dir, f"hist_normal_fit_{dist}_{tsp_type}_size{size}.png")
            plt.savefig(save_path)
            plt.close()

            print("Saved:", save_path)

def histogram_with_normal_fit_combined(dist: str = 'uniform', log_scale: bool = True):
    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={
            "figure.figsize": (16, 12),
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )

    df = load_phase_transition_iterations()

    filtered_df = df[
        (df['distribution'] == dist) &
        (df['range'].notna()) &
        (df['iteration'] > 0)
    ]

    city_sizes = [20, 30]
    tsp_types = ['euclidean', 'asymmetric', 'symmetric']
    output_dir = "./plot/histograms_normal_fit_combined/"
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)

    for row, size in enumerate(city_sizes):
        for col, tsp_type in enumerate(tsp_types):
            ax = axes[row, col]
            subset = filtered_df[
                (filtered_df['generation_type'] == tsp_type) &
                (filtered_df['city_size'] == size)
            ].copy()

            if subset.empty:
                ax.set_visible(False)
                continue

            if log_scale:
                subset['plot_iteration'] = np.log(subset['iteration'])
                xlabel = "Log (Lital Iterations)"
            else:
                subset['plot_iteration'] = subset['iteration']
                xlabel = "Lital Iterations"

            sns.histplot(
                subset['plot_iteration'],
                bins='auto',
                kde=False,
                stat='density',
                color='skyblue',
                alpha=0.8,
                ax=ax
            )

            mu, std = stats.norm.fit(subset['plot_iteration'])

            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = stats.norm.pdf(x, mu, std)
            ax.plot(x, p, 'k', linewidth=2, label=f'Normal fit\n$\mu$={mu:.2f}, $\sigma$={std:.2f}')

            mean_iter = subset['plot_iteration'].mean()
            median_iter = subset['plot_iteration'].median()

            ax.axvline(mean_iter, color='r', linestyle='--', linewidth=1.5, label=f'Log-Mean: {mean_iter:.2f}')
            ax.axvline(median_iter, color='g', linestyle=':', linewidth=1.5, label=f'Log-Median: {median_iter:.2f}')

            inset_text = f"{tsp_type[0].capitalize()}TSP\nCity: {size}\n{dist.capitalize()}"
            ax.text(
                0.05, 0.95, inset_text,
                ha='left', va='top',
                transform=ax.transAxes,
                fontsize=13,
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.2', alpha=0.8)
            )

            if row == 1:
                ax.set_xlabel(xlabel)
            else:
                ax.set_xlabel("")

            if col == 0:
                ax.set_ylabel("Density")
            else:
                ax.set_ylabel("")

            ax.legend(loc='best', fontsize=13)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"hist_normal_fit_combined_{dist}.png")
    plt.savefig(save_path)
    plt.close()

    print("Saved combined plot:", save_path)

if __name__ == "__main__":
    # histogram_phase_transition('uniform', log_scale=True)    # Log-scale
    # histogram_phase_transition('uniform', log_scale=False)   # Raw iterations

    # histogram_phase_transition_combined('uniform', log_scale=True)
    # histogram_phase_transition_combined('uniform', log_scale=False)

    # histogram_with_normal_fit('uniform', log_scale=True)
    # histogram_with_normal_fit('uniform', log_scale=False)

    histogram_with_normal_fit_combined('uniform', log_scale=True)
    # histogram_with_normal_fit_combined('uniform', log_scale=False)
    histogram_phase_transition_combined('lognormal', log_scale=True)
    # histogram_with_normal_fit_combined('lognormal', log_scale=False)


# if __name__ == "__main__":
#     # histogram_phase_transition()
#     # histogram_phase_transition('lognormal')
#     # histogram_phase_transition_combined('uniform')
#     # histogram_phase_transition_combined('lognormal')
#     # histogram_with_normal_fit('uniform')
#     # histogram_with_normal_fit('lognormal')
#     histogram_with_normal_fit_combined('uniform')
#     histogram_with_normal_fit_combined('lognormal')
