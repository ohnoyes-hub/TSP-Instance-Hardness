import matplotlib.pyplot as plt
import seaborn as sns
from util.load_experiment import load_hill_climb_iterations
from icecream import ic
import numpy as np
import os
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pandas as pd
import itertools
from matplotlib.lines import Line2D

def fit_hc_nb_glm(df):
    df = df[df['iteration'] > 0]
    model = smf.glm(
        formula="iteration ~ C(mutation_type) + C(generation_type) + city_size + range + C(distribution)",
        data=df,
        family=sm.families.NegativeBinomial()
    ).fit()
    return model

def fill_hc_missing_with_synthetic(df, min_count=20):
    EXPECTED_RANGES = {
        "uniform": np.arange(5, 101, 5),
        "lognormal": np.round(np.arange(0.2, 5.01, 0.2), 2),
    }
    distributions = df['distribution'].unique()
    tsp_types = df['generation_type'].unique()
    city_sizes = df['city_size'].unique()
    mutation_types = df['mutation_type'].unique()
    combs = list(itertools.product(distributions, tsp_types, city_sizes, mutation_types))

    model = fit_hc_nb_glm(df)
    synthetic_rows = []
    for d, t, s, m in combs:
        for r in EXPECTED_RANGES[d]:
            mask = (
                (df['distribution'] == d) &
                (df['generation_type'] == t) &
                (df['city_size'] == s) &
                (df['range'] == r) &
                (df['mutation_type'] == m)
            )
            count = mask.sum()
            n_needed = max(0, min_count - count)
            if n_needed > 0:
                synth = pd.DataFrame({
                    'distribution': [d] * n_needed,
                    'generation_type': [t] * n_needed,
                    'city_size': [s] * n_needed,
                    'range': [r] * n_needed,
                    'mutation_type': [m] * n_needed,
                })
                expected = model.predict(synth)
                alpha = model.scale
                mu = expected
                size = 1 / alpha
                prob = size / (size + mu)
                synth['iteration'] = np.random.negative_binomial(size, prob)
                synth['iteration'] = synth['iteration'].clip(lower=1)
                synth['synthetic'] = True
                synthetic_rows.append(synth)
    if synthetic_rows:
        df_synth = pd.concat(synthetic_rows, ignore_index=True)
        df_real = df.copy()
        df_real['synthetic'] = False
        df_synth = df_synth[df_synth['iteration'] > 0]
        df_all = pd.concat([df_real, df_synth], ignore_index=True)
        return df_all
    else:
        df['synthetic'] = False
        return df

def plot_combined_hill_transition():
    sns.set_theme(
        style="whitegrid",
        context="talk",
        palette="viridis",
        rc={
            "figure.figsize": (27, 24),
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14
        }
    )

    df = load_hill_climb_iterations()
    df = fill_hc_missing_with_synthetic(df, min_count=20)
    distributions = ['uniform', 'lognormal']
    tsp_types = ['asymmetric', 'euclidean']
    city_sizes = [20, 30]
    mutation_types = sorted(df['mutation_type'].unique())
    
    # For 4 rows: each is a unique (city_size, tsp_type) pair, columns: uniform (0), lognormal (1)
    row_combinations = [(city_size, tsp_type) for city_size in city_sizes for tsp_type in tsp_types]
    n_rows = 4
    n_cols = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 24), sharey='row')

    # For legend
    custom_handles = None
    legend_labels = [
        'Median (log)', 'Mean (log)', '±0.5 Std Dev (log)',
        'Log Lital iteration (real)', 'Log Lital iteration (synthetic)'
    ]

    for row_idx, (city_size, tsp_type) in enumerate(row_combinations):
        for col_idx, distribution in enumerate(distributions):
            ax = axes[row_idx, col_idx]
            # Select ALL mutation types together for each subplot (as in your original grid)
            filtered_df = df[
                (df['distribution'] == distribution) &
                (df['range'].notna()) &
                (df['iteration'] > 0) &
                (df['generation_type'] == tsp_type) &
                (df['city_size'] == city_size)
            ]
            if filtered_df.empty:
                ic("No data found", distribution, tsp_type, city_size)
                ax.set_visible(False)
                continue
            subset = filtered_df.copy()
            subset['log_iteration'] = np.log(subset['iteration'])
            stats = subset.groupby('range')['log_iteration'].agg(['median', 'mean', 'std']).reset_index()
            stats = stats.sort_values('range')
            x = stats['range'].values
            y_median = stats['median'].values
            if len(x) > 1:
                coeffs = np.polyfit(x, y_median, 1)
                trend = np.poly1d(coeffs)(x)
                trend_label = f"$y={coeffs[0]:.2f}x+{coeffs[1]:.2f}$"
            else:
                trend_label = ""
            # Plot lines
            median_line = sns.lineplot(
                x='range', y='median', data=stats, marker="s",
                ax=ax, linewidth=1.5, label='Median (log)'
            )
            mean_line = sns.lineplot(
                x='range', y='mean', data=stats, marker="^",
                ax=ax, linewidth=1.5, label='Mean (log)'
            )
            ribbon = ax.fill_between(
                stats['range'],
                stats['median'] - 0.5 * stats['std'],
                stats['median'] + 0.5 * stats['std'],
                color='gray', alpha=0.2, label='±0.5 Std Dev (log)'
            )
            real_df = subset[~subset['synthetic']]
            synth_df = subset[subset['synthetic']]
            scatter_real = sns.scatterplot(
                x='range', y='log_iteration', data=real_df,
                color='k', alpha=0.35, edgecolor=None, s=20,
                ax=ax, label='Log Lital iteration (real)'
            )
            scatter_synth = sns.scatterplot(
                x='range', y='log_iteration', data=synth_df,
                color='orange', marker='X', edgecolor='black', s=60, alpha=0.6,
                ax=ax, label='Log Lital iteration (synthetic)'
            )
            # Plot trendline
            if len(x) > 1:
                trend_line, = ax.plot(
                    x, trend, linewidth=2, color='red', linestyle='--',
                    label=trend_label
                )
                ax.legend(
                    handles=[trend_line],
                    labels=[trend_label],
                    loc='upper right',
                    frameon=True,
                    fontsize=14
                )
            # Collect handles for unified legend only once
            if row_idx == 0 and col_idx == 0:
                custom_handles = [
                    median_line.lines[0],
                    mean_line.lines[0],
                    ribbon,
                    Line2D([], [], linestyle="none", marker='o', color='k', alpha=0.35, markersize=8, label='Log Lital iteration (real)'),
                    Line2D([], [], linestyle="none", marker='X', color='orange', markeredgecolor='black', markersize=10, label='Log Lital iteration (synthetic)'),
                ]
            # Inset: titles
            tsp_abbrev = {'euclidean': 'E', 'asymmetric': 'A'}[tsp_type]
            inset_text = f"Hill-Climbing\n{tsp_abbrev}TSP\n{city_size}-City\n{distribution.capitalize()}"
            ax.text(
                0.05, 0.95, inset_text,
                ha='left', va='top',
                transform=ax.transAxes,
                fontsize=14, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.7),
                zorder=10
            )
            # X labels only for bottom row
            if row_idx == n_rows - 1:
                ax.set_xlabel(r"$rand_{max}$" if distribution == 'uniform' else r"$\sigma$", fontsize=14)
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])
            # Y labels only for first column
            if col_idx == 0:
                ax.set_ylabel("Log Lital Iterations", fontsize=14)
            else:
                ax.set_ylabel("")
    # Unified legend at bottom
    fig.legend(
        handles=custom_handles,
        labels=legend_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.01),
        ncol=5,
        fontsize=14,
        frameon=True
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    os.makedirs('./plot/hill_climb_transition', exist_ok=True)
    save_path = os.path.join('./plot/hill_climb_transition', 'hill_climb_transition_combined.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    ic("Saved:", save_path)

if __name__ == "__main__":
    plot_combined_hill_transition()
