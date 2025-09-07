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
    tsp_types = ['euclidean', 'asymmetric', 'symmetric']
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

# plot_combined_phase_transition(for_slides=True, orientation='horizontal')
def plot_combined_phase_transition(*, for_slides: bool = False,
                                   orientation: str = 'horizontal',
                                   figsize: tuple | None = None) -> None:
    """
    Plot phase transition results for different TSP generation types, distributions, and city sizes.

    By default (``for_slides=False``) this function produces one figure per TSP generation type, with
    a 2×2 grid of subplots (rows correspond to city sizes and columns correspond to distributions).
    Each figure is intended to fill an entire page for inclusion in reports or papers.

    When ``for_slides=True`` the function instead produces a single combined figure containing
    subplots for all generation type/distribution combinations. In this mode the city sizes are
    aggregated within each subplot (i.e., results from both city sizes are pooled together) so
    that the layout is more compact. The subplots are arranged according to ``orientation``:

    * ``orientation='horizontal'`` (default) produces a grid with one row per distribution and
      one column per generation type. For example, with two distributions (uniform, lognormal)
      and three generation types (Euclidean, Asymmetric, Symmetric) this yields a 2×3 grid,
      which better fits the widescreen aspect ratio of typical presentation slides.
    * ``orientation='vertical'`` swaps the arrangement to one row per generation type and one
      column per distribution (a 3×2 grid for the same example).

    A custom ``figsize`` may be provided to control the overall figure dimensions. If omitted,
    sensible defaults are chosen based on the number of rows and columns.

    Parameters
    ----------
    for_slides : bool, optional
        If True, produce a single figure suitable for slides (aggregated city sizes) instead of
        separate figures for each TSP generation type. Default is False.
    orientation : {'horizontal', 'vertical'}, optional
        Layout orientation used only when ``for_slides=True``. Determines whether the rows
        correspond to distributions and columns correspond to generation types ('horizontal'),
        or vice versa ('vertical'). Default is 'horizontal'.
    figsize : tuple of float, optional
        Size of the overall figure in inches as (width, height). If None, reasonable defaults
        based on the selected layout are used. Note that the underlying Seaborn theme sets
        various font sizes and styles for consistency.
    """
    # Set up Seaborn theme. We avoid hard-coding figure sizes here so that users can supply
    # their own ``figsize`` via the function arguments; however the rc context still
    # specifies font sizes and weights for visual consistency.
    sns.set_theme(
        style="whitegrid",
        context="talk",
        palette="viridis",
        rc={
            # Default figure size is set later when creating individual figures.
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "font.weight": "bold",
        }
    )

    # Load and prepare data. We always fill missing values with synthetic samples for
    # completeness prior to plotting.
    df = load_phase_transition_iterations()
    df = fill_missing_with_synthetic(df, min_count=100)

    distributions = ['uniform', 'lognormal']
    tsp_types = ['euclidean', 'asymmetric', 'symmetric']
    city_sizes = [20, 30]

    # ------------------------------------------------------------------------------
    # Standard (page) mode: produce one 2×2 figure per generation type.
    # Each figure contains city sizes on rows and distributions on columns.
    # ------------------------------------------------------------------------------
    if not for_slides:
        # Provide a reasonable default figure size if none supplied. 24×16 gives a
        # roughly 3:2 aspect ratio suited for a full page. Users may override
        # ``figsize`` when calling the function.
        default_figsize = (24, 16)
        for tsp_type in tsp_types:
            fig, axes = plt.subplots(
                2, 2,
                figsize=figsize or default_figsize,
                sharey=True
            )
            legend_labels = [
                'Median (log)', 'Mean (log)', '±0.5 Std Dev (log)',
                'log(Lital iteration) (real)', 'log(Lital iteration) (synthetic)'
            ]
            custom_handles = []

            for dist_idx, dist in enumerate(distributions):
                for size_idx, size in enumerate(city_sizes):
                    ax = axes[size_idx, dist_idx]

                    # Filter data for this combination
                    filtered_df = df[
                        (df['distribution'] == dist) &
                        (df['range'].notna()) &
                        (df['iteration'] > 0) &
                        (df['generation_type'] == tsp_type) &
                        (df['city_size'] == size)
                    ]
                    if filtered_df.empty:
                        ic("No data found at all", dist, tsp_type, size)
                        continue

                    # Prepare data
                    subset = filtered_df.copy()
                    subset['log_iteration'] = np.log(subset['iteration'])
                    stats = subset.groupby('range')['log_iteration'].agg(['median', 'mean', 'std']).reset_index()
                    stats = stats.sort_values('range')

                    # Linear fit on the median values
                    x = stats['range'].values
                    y_median = stats['median'].values
                    coeffs = np.polyfit(x, y_median, 1)
                    trend = np.poly1d(coeffs)(x)
                    trend_label = f"$y={coeffs[0]:.2f}x+{coeffs[1]:.2f}$"

                    # Plot the median and mean lines
                    median_line = sns.lineplot(
                        x='range', y='median', data=stats,
                        marker="s", ax=ax, linewidth=1.5, label='Median (log)'
                    )
                    mean_line = sns.lineplot(
                        x='range', y='mean', data=stats,
                        marker="^", ax=ax, linewidth=1.5, label='Mean (log)'
                    )
                    ribbon = ax.fill_between(
                        stats['range'],
                        stats['median'] - 0.5 * stats['std'],
                        stats['median'] + 0.5 * stats['std'],
                        color='gray', alpha=0.2, label='±0.5 Std Dev (log)'
                    )

                    # Scatter plot of real and synthetic observations
                    real_df = subset[~subset['synthetic']]
                    synth_df = subset[subset['synthetic']]
                    sns.scatterplot(
                        x='range', y='log_iteration', data=real_df,
                        color='k', alpha=0.35, edgecolor=None, s=20, ax=ax,
                        label='log(Lital iteration) (real)'
                    )
                    sns.scatterplot(
                        x='range', y='log_iteration', data=synth_df,
                        color='orange', marker='X', edgecolor='black', s=60, alpha=0.6,
                        ax=ax, label='log(Lital iteration) (synthetic)'
                    )
                    # Plot trend line and store handle separately
                    trend_line, = ax.plot(
                        x, trend, linewidth=2, color='red', linestyle='--', label=trend_label
                    )

                    # Put only the trend line label inside the subplot
                    ax.legend(
                        handles=[trend_line], labels=[trend_label], loc='upper right',
                        frameon=True, fontsize=14
                    )

                    # Store handles from the first subplot for a unified legend
                    if dist_idx == 0 and size_idx == 0:
                        custom_handles = [
                            median_line.lines[0],
                            mean_line.lines[0],
                            ribbon,
                            Line2D([], [], linestyle="none", marker='o', color='k', alpha=0.35,
                                   markersize=8, label='log(Lital iteration) (real)'),
                            Line2D([], [], linestyle="none", marker='X', color='orange',
                                   markeredgecolor='black', markersize=10,
                                   label='log(Lital iteration) (synthetic)')
                        ]

                    # Annotate the subplot with TSP type, city size, and distribution
                    inset_text = f"{tsp_type[0].capitalize()}TSP\n{size}-City\n{dist.capitalize()}"
                    ax.text(
                        0.05, 0.95, inset_text,
                        ha='left', va='top', transform=ax.transAxes,
                        fontsize=14, fontweight='bold',
                        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.7),
                        zorder=10
                    )

                    # X-axis labels only on bottom row
                    if size_idx == len(city_sizes) - 1:
                        xlabel = r"$rand_{max}$" if dist == 'uniform' else r"$\sigma$"
                        ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
                    else:
                        ax.set_xlabel("")
                        ax.set_xticklabels([])

                    # Y-axis labels only on first column
                    if dist_idx == 0:
                        ax.set_ylabel("Log Lital Iterations", fontsize=14, fontweight='bold')
                    else:
                        ax.set_ylabel("")

            # End of nested loops

            # Create a common legend at the bottom of the figure
            fig.legend(
                handles=custom_handles,
                labels=legend_labels,
                loc='lower center', bbox_to_anchor=(0.5, 0.01),
                ncol=len(legend_labels), fontsize=14, frameon=True
            )
            # Adjust layout so nothing overlaps with the legend or suptitle
            plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.93])

            # Ensure output directory exists and save the figure
            os.makedirs('./plot/phase_transition', exist_ok=True)
            save_path = os.path.join('./plot/phase_transition', f'phase_transition_combined_{tsp_type}.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            ic("Saved:", save_path)

        # End of default mode: return so that slide-mode code below is not executed.
        return

    # ------------------------------------------------------------------------------
    # Slide mode: produce a single figure combining all generation types and distributions.
    # In this mode city sizes are aggregated within each subplot to reduce the total
    # number of panels and yield a layout that fits better on a widescreen slide.
    # ------------------------------------------------------------------------------
    # Determine the number of rows and columns based on the chosen orientation.
    # 'horizontal' means rows = distributions, columns = TSP types; 'vertical' swaps them.
    if orientation not in {'horizontal', 'vertical'}:
        raise ValueError("orientation must be either 'horizontal' or 'vertical'")

    nrows = len(distributions) if orientation == 'horizontal' else len(tsp_types)
    ncols = len(tsp_types) if orientation == 'horizontal' else len(distributions)

    # Choose a default figure size if not provided. We scale width and height according
    # to the number of columns and rows so that each panel has a reasonable aspect.
    # A base size of (6, 4) per panel works well for slides.
    if figsize is None:
        base_w, base_h = 6.0, 4.0
        figsize = (base_w * ncols, base_h * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
    # In case of a 1×1 figure, axes might not be a 2D array; ensure consistency
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols)

    # For the slide layout we will build a unified legend across all subplots. We'll
    # capture handles from the first subplot to populate this later.
    slide_handles = []
    first_panel = True

    for dist_idx, dist in enumerate(distributions):
        for tsp_idx, tsp_type in enumerate(tsp_types):
            # Determine the position of this subplot in the grid
            if orientation == 'horizontal':
                row_idx, col_idx = dist_idx, tsp_idx
            else:
                row_idx, col_idx = tsp_idx, dist_idx
            ax = axes[row_idx, col_idx]

            # Aggregate across both city sizes for this combination
            subset = df[
                (df['distribution'] == dist) &
                (df['range'].notna()) &
                (df['iteration'] > 0) &
                (df['generation_type'] == tsp_type)
            ].copy()
            if subset.empty:
                ic("No data found at all", dist, tsp_type)
                continue
            subset['log_iteration'] = np.log(subset['iteration'])
            stats = subset.groupby('range')['log_iteration'].agg(['median', 'mean', 'std']).reset_index()
            stats = stats.sort_values('range')

            # Trendline fit on the median values
            x = stats['range'].values
            y_median = stats['median'].values
            coeffs = np.polyfit(x, y_median, 1)
            trend = np.poly1d(coeffs)(x)
            trend_label = f"$y={coeffs[0]:.2f}x+{coeffs[1]:.2f}$"

            # Plot median and mean lines
            line_median = sns.lineplot(
                x='range', y='median', data=stats, marker="s",
                ax=ax, linewidth=1.5, label='Median (log)'
            )
            line_mean = sns.lineplot(
                x='range', y='mean', data=stats, marker="^",
                ax=ax, linewidth=1.5, label='Mean (log)'
            )
            # Ribbon for ±0.5 standard deviation
            ribbon = ax.fill_between(
                stats['range'],
                stats['median'] - 0.5 * stats['std'],
                stats['median'] + 0.5 * stats['std'],
                color='gray', alpha=0.2, label='±0.5 Std Dev (log)'
            )
            # Scatter: real vs synthetic observations
            real_df = subset[~subset['synthetic']]
            synth_df = subset[subset['synthetic']]
            sns.scatterplot(
                x='range', y='log_iteration', data=real_df,
                color='k', alpha=0.35, edgecolor=None, s=20, ax=ax,
                label='log(Lital iteration) (real)'
            )
            sns.scatterplot(
                x='range', y='log_iteration', data=synth_df,
                color='orange', marker='X', edgecolor='black', s=60, alpha=0.6,
                ax=ax, label='log(Lital iteration) (synthetic)'
            )
            # Trend line
            trend_line, = ax.plot(
                x, trend, linewidth=2, color='red', linestyle='--', label=trend_label
            )

            # Show the trend line label inside the subplot
            ax.legend(
                handles=[trend_line], labels=[trend_label], loc='upper right',
                frameon=True, fontsize=12
            )

            # Capture the handles for the unified legend on the first panel only
            if first_panel:
                slide_handles = [
                    line_median.lines[0],
                    line_mean.lines[0],
                    ribbon,
                    Line2D([], [], linestyle="none", marker='o', color='k', alpha=0.35,
                           markersize=6, label='log(Lital iteration) (real)'),
                    Line2D([], [], linestyle="none", marker='X', color='orange',
                           markeredgecolor='black', markersize=8,
                           label='log(Lital iteration) (synthetic)')
                ]
                first_panel = False

            # Annotate with the generation type and distribution. City sizes are
            # aggregated so we omit them from the annotation.
            inset_text = f"{tsp_type[0].capitalize()}TSP\n{dist.capitalize()}"
            ax.text(
                0.05, 0.95, inset_text,
                ha='left', va='top', transform=ax.transAxes,
                fontsize=12, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.7),
                zorder=10
            )

            # X-axis labeling: bottom row only
            if row_idx == nrows - 1:
                xlabel = r"$rand_{max}$" if dist == 'uniform' else r"$\sigma$"
                ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])

            # Y-axis labeling: first column only
            if col_idx == 0:
                ax.set_ylabel("Log Lital Iterations", fontsize=12, fontweight='bold')
            else:
                ax.set_ylabel("")

    # Create a unified legend for the slide layout
    legend_labels_slide = [
        'Median (log)', 'Mean (log)', '±0.5 Std Dev (log)',
        'log(Lital iteration) (real)', 'log(Lital iteration) (synthetic)'
    ]
    fig.legend(
        handles=slide_handles,
        labels=legend_labels_slide,
        loc='lower center', bbox_to_anchor=(0.5, -0.02),
        ncol=len(legend_labels_slide), fontsize=12, frameon=True
    )

    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.93])
    # Save the slide figure
    os.makedirs('./plot/phase_transition', exist_ok=True)
    slide_filename = os.path.join('./plot/phase_transition', 'phase_transition_combined_slide.png')
    plt.savefig(slide_filename, bbox_inches='tight')
    plt.close()
    ic("Saved slide figure:", slide_filename)

if __name__ == "__main__":
    plot_combined_phase_transition()
    # plot_combined_phase_transition(for_slides=True, orientation='horizontal')
