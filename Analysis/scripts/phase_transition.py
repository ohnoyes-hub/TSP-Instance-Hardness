import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from matplotlib.lines import Line2D
import statsmodels.formula.api as smf
import statsmodels.api as sm
import itertools

from util.load_experiment import load_phase_transition_iterations
from icecream import ic

def fit_nb_glm(df):
    # Only use rows with valid iterations for fitting
    df = df[df['iteration'] > 0]
    # Fit a single model to all
    model = smf.glm(
        formula="iteration ~ C(generation_type) + city_size + range + C(distribution)",
        data=df,
        family=sm.families.NegativeBinomial()
    ).fit()
    return model

def fill_missing_with_synthetic(df, min_count=100):
    EXPECTED_RANGES = {
        "uniform": np.arange(5, 101, 5),
        "lognormal": np.round(np.arange(0.2, 5.01, 0.2), 2),
    }
    distributions = df['distribution'].unique()
    tsp_types = df['generation_type'].unique()
    city_sizes = df['city_size'].unique()

    combs = list(itertools.product(distributions, tsp_types, city_sizes))

    model = fit_nb_glm(df)
    synthetic_rows = []

    for d, t, s in combs:
        for r in EXPECTED_RANGES[d]:
            mask = (
                (df['distribution'] == d) &
                (df['generation_type'] == t) &
                (df['city_size'] == s) &
                (df['range'] == r)
            )
            count = mask.sum()
            n_needed = max(0, min_count - count)
            if n_needed > 0:
                synth = pd.DataFrame({
                    'distribution': [d] * n_needed,
                    'generation_type': [t] * n_needed,
                    'city_size': [s] * n_needed,
                    'range': [r] * n_needed,
                })
                expected = model.predict(synth)

                # More realistic negative binomial sampling with variability
                alpha = model.scale  # dispersion parameter from NBGLM
                mu = expected
                size = 1 / alpha
                prob = size / (size + mu)
                synth['iteration'] = np.random.negative_binomial(size, prob)

                # Ensure positive iterations
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

def plot_combined_phase_transition():
    sns.set_theme(
        style="whitegrid",
        context="talk",
        palette="viridis",
        rc={
            "figure.figsize": (28, 16),
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "font.weight": "bold",
        }
    )

    df = load_phase_transition_iterations()
    df = fill_missing_with_synthetic(df, min_count=100)

    distributions = ['uniform', 'lognormal']
    tsp_types = ['euclidean', 'asymmetric']
    city_sizes = [20, 30]

    for tsp_type in tsp_types:
        fig, axes = plt.subplots(2, 2, figsize=(24, 16), sharey=True)
        legend_labels = ['Median (log)', 'Mean (log)', '±0.5 Std Dev (log)', 'log(Lital iteration) (real)', 'log(Lital iteration) (synthetic)']
        custom_handles = []

        for dist_idx, dist in enumerate(distributions):
            for size_idx, size in enumerate(city_sizes):
                ax = axes[size_idx, dist_idx]
                
                # Filter data
                filtered_df = df[
                    (df['distribution'] == dist) &
                    (df['range'].notna()) &
                    (df['iteration'] > 0) &
                    (df['generation_type'] == tsp_type) &
                    (df['city_size'] == size)
                ]
                if filtered_df.empty:
                    # No real or synthetic data, so skip (should not happen with correct synthetic fill)
                    ic("No data found at all", dist, tsp_type, size)
                    continue
                
                # Process data
                subset = filtered_df.copy()
                subset['log_iteration'] = np.log(subset['iteration'])
                stats = subset.groupby('range')['log_iteration'].agg(['median', 'mean', 'std']).reset_index()
                stats = stats.sort_values('range')
                
                # Calculate linear fit
                x = stats['range'].values
                y_median = stats['median'].values
                coeffs = np.polyfit(x, y_median, 1)
                trend = np.poly1d(coeffs)(x)
                trend_label = f"$y={coeffs[0]:.2f}x+{coeffs[1]:.2f}$"
                
                # Plot elements
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
                # Scatter: real vs synthetic
                real_df = subset[~subset['synthetic']]
                synth_df = subset[subset['synthetic']]
                scatter_real = sns.scatterplot(
                    x='range', y='log_iteration', data=real_df,
                    color='k', alpha=0.35, edgecolor=None, s=20, 
                    ax=ax, label='log(Lital iteration) (real)'
                )
                scatter_handle_real = ax.collections[-1]
                scatter_synth = sns.scatterplot(
                    x='range', y='log_iteration', data=synth_df,
                    color='orange', marker='X', edgecolor='black', s=60, alpha=0.6,
                    ax=ax, label='log(Lital iteration) (synthetic)'
                )   
                scatter_handle_synth = ax.collections[-1]
                trend_line, = ax.plot(
                    x, trend, linewidth=2, color='red', linestyle='--',
                    label=trend_label
                )
                
                # Add trendline legend inside subplot
                ax.legend(
                    handles=[trend_line],
                    labels=[trend_label],
                    loc='upper right',
                    frameon=True,
                    fontsize=14,
                )
                
                # Collect handles for unified legend (from first subplot)
                if dist_idx == 0 and size_idx == 0:
                    custom_handles = [
                        median_line.lines[0], 
                        mean_line.lines[0], 
                        ribbon, 
                        Line2D([], [], linestyle="none", marker='o', color='k', alpha=0.35, markersize=8, label='log(Lital iteration) (real)'),
                        Line2D([], [], linestyle="none", marker='X', color='orange', markeredgecolor='black', markersize=10, label='log(Lital iteration) (synthetic)'),
                    ]
                
                # Subplot titles and labels
                inset_text = f"{tsp_type[0].capitalize()}TSP\n{size}-City\n{dist.capitalize()}"
                ax.text(
                    0.05, 0.95, inset_text,
                    ha='left', va='top',
                    transform=ax.transAxes,
                    fontsize=14, fontweight='bold',
                    bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.7),
                    zorder=10
                )

                if size_idx == 1:
                    ax.set_xlabel(r"$rand_{max}$" if dist == 'uniform' else r"$\sigma$", fontsize=14, fontweight='bold')
                else:
                    ax.set_xlabel("")
                    ax.set_xticklabels([])

                if dist_idx == 0:
                    ax.set_ylabel("Log Lital Iterations", fontsize=14, fontweight='bold')
                else:
                    ax.set_ylabel("")
        
        fig.legend(
            handles=custom_handles,
            labels=legend_labels,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.01),
            ncol=5,
            fontsize=14,
            frameon=True
        )
        plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.93])
        os.makedirs('./plot/phase_transition', exist_ok=True)
        save_path = os.path.join('./plot/phase_transition', f'phase_transition_combined_{tsp_type}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        ic("Saved:", save_path)

if __name__ == "__main__":
    plot_combined_phase_transition()
