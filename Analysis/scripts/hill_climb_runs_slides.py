import matplotlib.pyplot as plt
from util.load_experiment import load_full, load_all_hard_instances
import pandas as pd
from collections import defaultdict
import numpy as np
import math
import os

plt.rcParams.update({
    'font.size': 20,            
    'axes.titlesize': 24,       
    'axes.titleweight': 'bold', 
    'axes.labelsize': 22,       
    'axes.labelweight': 'bold', 
    'xtick.labelsize': 18,      
    'ytick.labelsize': 18,      
    'xtick.color': 'black',
    'ytick.color': 'black',
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'legend.fontsize': 20,      
    'legend.title_fontsize': 20,
    'legend.edgecolor': 'black',
    'legend.fancybox': True,
    'font.weight': 'bold',
})

DISPLAY_NAME_MAP = {
    'generation_type': 'tsp type',
    'range': 'control',
}

# Load data and organise it by configuration. This is done once at import time
all_data = load_full()
hard_instances_df = load_all_hard_instances()

# Group data by configuration
config_groups = defaultdict(list)
for data in all_data:
    config = data.get('configuration', {})
    config_key = tuple(sorted(config.items()))
    config_groups[config_key].append(data)

# ----- MANUAL SELECTION -----
# These are the six configurations that will be plotted. Each entry is a tuple
# of key-value pairs describing the configuration. Feel free to modify this
# list or supply your own when calling ``plot_hill_climb_runs_combined``.
selected_configs = [
    (('city_size', 20), ('distribution', 'lognormal'), ('generation_type', 'euclidean'), ('mutation_type', 'inplace'), ('range', 0.6)), 
    (('city_size', 30), ('distribution', 'uniform'), ('generation_type', 'euclidean'), ('mutation_type', 'swap'), ('range', 25)),
    (('city_size', 20), ('distribution', 'lognormal'), ('generation_type', 'euclidean'), ('mutation_type', 'scramble'), ('range', 1.0)),
    (('city_size', 20), ('distribution', 'lognormal'), ('generation_type', 'asymmetric'), ('mutation_type', 'inplace'), ('range', 1.0)), 
    (('city_size', 20), ('distribution', 'lognormal'), ('generation_type', 'asymmetric'), ('mutation_type', 'swap'), ('range', 2.0)),
    (('city_size', 20), ('distribution', 'uniform'), ('generation_type', 'asymmetric'), ('mutation_type', 'scramble'), ('range', 25)),
]

def plot_hill_climb_runs_combined(*, for_slides: bool = False,
                                  orientation: str = 'vertical',
                                  figsize: tuple | None = None,
                                  configs: list | None = None) -> None:
    """
    Plot Lital iteration progress across multiple hill-climb runs for a collection of
    configurations. The default layout (``for_slides=False``) uses a 3×2 grid (rows × columns)
    sized to fill a page. When ``for_slides=True``, the panels are re-arranged to better fit
    a presentation slide. The ``orientation`` argument controls whether the slide layout
    uses more rows than columns (``'vertical'``) or more columns than rows (``'horizontal'``).

    Parameters
    ----------
    for_slides : bool, optional
        If True, arrange the panels in a widescreen-friendly layout; otherwise use the
        default page-filling layout. Default is False.
    orientation : {'vertical', 'horizontal'}, optional
        Determines the shape of the grid when ``for_slides=True``. A vertical layout
        (the default) creates a taller figure by setting the number of rows equal to
        the default and adjusting columns accordingly. A horizontal layout swaps the
        arrangement, making the figure wider. This parameter is ignored when
        ``for_slides=False``.
    figsize : tuple of float, optional
        A custom figure size in inches (width, height). If None, the function
        computes a reasonable size based on the number of panels and the layout.
    configs : list of configuration tuples, optional
        The list of configuration keys to plot. If None, uses the module-level
        ``selected_configs`` defined above.
    """
    if configs is None:
        configs = selected_configs
    # Determine number of panels
    num_panels = len(configs)

    # Determine layout
    if not for_slides:
        # Default: 3×2 layout fills a page
        n_rows, n_cols = 3, 2
    else:
        if orientation not in {'vertical', 'horizontal'}:
            raise ValueError("orientation must be either 'vertical' or 'horizontal'")
        # For slide layouts we attempt to produce a roughly 16:9 aspect ratio. We fix
        # one dimension and compute the other based on the number of panels.
        if orientation == 'horizontal':
            n_cols = 3  # three panels across
            n_rows = math.ceil(num_panels / n_cols)
        else:  # vertical layout
            n_rows = 3
            n_cols = math.ceil(num_panels / n_rows)

    # Compute default figure size if none provided. Use a base size per panel and scale
    # by the number of rows and columns. The base size is larger in the default mode
    # (page) than in slide mode to account for smaller fonts and spacing on slides.
    if figsize is None:
        if not for_slides:
            # Default page figure size chosen to resemble the original script (26×28)
            base_w, base_h = 8.5, 9.0
        else:
            # Use a slightly smaller base for slides to fit more panels
            base_w, base_h = 6.5, 5.5
        figsize = (base_w * n_cols, base_h * n_rows)

    # Create subplots. Share x across columns and y within rows for consistency
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
    # Ensure axs is a 2D array even when n_rows or n_cols is 1
    if n_rows == 1 and n_cols == 1:
        axs = np.array([[axs]])
    elif n_rows == 1 or n_cols == 1:
        axs = axs.reshape(n_rows, n_cols)

    # Share y-axis among columns within each row for consistent scaling
    for row in range(n_rows):
        for col in range(1, n_cols):
            axs[row, col].get_shared_y_axes().join(axs[row, 0], axs[row, col])

    axs_flat = axs.flatten()

    # Plot each configuration
    for idx, config_key in enumerate(configs):
        if idx >= n_rows * n_cols:
            break  # Do not plot more configurations than subplots
        ax = axs_flat[idx]
        data_list = config_groups.get(config_key, [])
        config_dict = dict(config_key)

        # Plot all_iterations lines for each run in the configuration
        for run_idx, data in enumerate(data_list):
            all_iter = data.get('results', {}).get('all_iterations', [])
            if not all_iter:
                continue
            x = range(1, len(all_iter) + 1)
            # Only label the first line so the legend is not cluttered
            label = "Generation's Lital Iteration" if run_idx == 0 else None
            ax.plot(x, all_iter, marker='o', linestyle='-', markersize=3,
                    alpha=0.5, label=label)

        # Filter hard_instances for this configuration
        mask = pd.Series(True, index=hard_instances_df.index)
        for key, value in config_dict.items():
            if key in hard_instances_df.columns:
                mask &= (hard_instances_df[key] == value)
            else:
                mask &= False
        filtered_df = hard_instances_df[mask]

        # Plot hard_instances as red points
        if not filtered_df.empty:
            ax.scatter(
                filtered_df['generation'], 
                filtered_df['iterations'], 
                color='red', 
                zorder=5,
                s=40,
                label='Hard Instances (Local Optima)'
            )

        # Title inside the plot
        title_text = '\n'.join([
            f"{DISPLAY_NAME_MAP.get(k, k)}: {v}" for k, v in config_dict.items()
        ])
        ax.text(
            0.01, 0.98, title_text,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='top',
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.8,
                edgecolor="lightgray",
                pad=0.3
            )
        )

        ax.grid(True, alpha=0.3)
        # Y-axis label only for the first column
        if idx % n_cols == 0:
            ax.set_ylabel("Lital Iterations", fontsize=14)
        else:
            ax.set_ylabel("")
        # X-axis label only for the bottom row
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Generation", fontsize=14)
        else:
            ax.set_xlabel("")

    # Remove y-ticks for axes that are not in the first column
    for idx_ax, ax in enumerate(axs_flat):
        col_idx = idx_ax % n_cols
        if col_idx != 0:
            ax.tick_params(axis='y', left=False, labelleft=False)

    # Hide any unused axes
    for j in range(len(configs), n_rows * n_cols):
        fig.delaxes(axs_flat[j])

    # Collect legend handles and labels from the first plotted axis
    # There may be no handles if no data were plotted; guard accordingly
    legend_handles, legend_labels = [], []
    for ax in axs_flat:
        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            if l not in legend_labels:
                legend_labels.append(l)
                legend_handles.append(h)
        if legend_handles:
            break

    if legend_handles:
        # Place legend below the plots
        fig.legend(
            legend_handles, legend_labels,
            loc='lower center', bbox_to_anchor=(0.5, -0.02),
            ncol=len(legend_labels), fontsize=14, frameon=True
        )

    # Adjust layout
    fig.tight_layout(rect=[0, 0.02, 1, 1])

    # Ensure output directory exists and save the figure
    out_dir = 'plot/hill_climbed_runs'
    os.makedirs(out_dir, exist_ok=True)
    fname = 'hill_climbed_runs_combined_slide.png' if for_slides else 'hill_climbed_runs_combined.png'
    fig.savefig(os.path.join(out_dir, fname), bbox_inches='tight')
    print(f"Plot saved to: ./{out_dir}/{fname}")
    plt.close(fig)


if __name__ == "__main__":
    plot_hill_climb_runs_combined(for_slides=False)
    plot_hill_climb_runs_combined(for_slides=True, orientation="horizontal", figsize=(27, 12))