import matplotlib.pyplot as plt
from util.load_experiment import load_full, load_all_hard_instances
import pandas as pd
from collections import defaultdict

# Load data
all_data = load_full()
hard_instances_df = load_all_hard_instances()

# Group data by configuration
config_groups = defaultdict(list)
for data in all_data:
    config = data.get('configuration', {})
    config_key = tuple(sorted(config.items()))
    config_groups[config_key].append(data)

# ----- MANUAL SELECTION -----
selected_configs = [
    (('city_size', 20), ('distribution', 'lognormal'), ('generation_type', 'euclidean'), ('mutation_type', 'inplace'), ('range', 0.4)), 
    (('city_size', 20), ('distribution', 'lognormal'), ('generation_type', 'euclidean'), ('mutation_type', 'inplace'), ('range', 0.6)), 
    (('city_size', 20), ('distribution', 'uniform'), ('generation_type', 'asymmetric'), ('mutation_type', 'swap'), ('range', 40)),
    (('city_size', 20), ('distribution', 'uniform'), ('generation_type', 'asymmetric'), ('mutation_type', 'swap'), ('range', 70)),
    (('city_size', 20), ('distribution', 'lognormal'), ('generation_type', 'asymmetric'), ('mutation_type', 'scramble'), ('range', 0.2)),
    (('city_size', 20), ('distribution', 'lognormal'), ('generation_type', 'asymmetric'), ('mutation_type', 'scramble'), ('range', 2.2)),
]

# If you want to see all possible config_keys:
# for i, config_key in enumerate(config_groups.keys()):
#     print(f"{i}: {config_key}")

n_rows, n_cols = 3, 2
fig, axs = plt.subplots(n_rows, n_cols, figsize=(24, 18), sharex=True)
# Share y only within rows
for row in range(n_rows):
    axs[row, 1].get_shared_y_axes().join(axs[row, 0], axs[row, 1])
axs_flat = axs.flatten()


for idx, config_key in enumerate(selected_configs):
    if idx >= n_rows * n_cols:
        break
    ax = axs_flat[idx]
    data_list = config_groups.get(config_key, [])
    config_dict = dict(config_key)
    
    # Plot all_iterations lines for each run in the configuration
    for run_idx, data in enumerate(data_list):
        all_iter = data.get('results', {}).get('all_iterations', [])
        if not all_iter:
            continue
        x = range(1, len(all_iter) + 1)
        label = 'Generation\'s Lital Iteration' if run_idx == 0 else None
        ax.plot(x, all_iter, marker='o', linestyle='-', markersize=4, alpha=0.5, label=label)
    
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
            label='Hard Instances(Local Optima)'
        )
    
    # Title insdie the plot
    title_text = '\n'.join([f"{k}: {v}" for k, v in config_dict.items()])
    ax.text(
        0.01,
        0.98,
        title_text,
        transform=ax.transAxes,  # Use axes coordinates
        fontsize=9,
        verticalalignment='top',
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.8,
            edgecolor="lightgray",
            pad=0.4
        )
    )

    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.set_ylabel("Lital Iterations")
    if idx >= (n_rows-1)*n_cols:
        ax.set_xlabel("Generation")

# Remove y-ticks for right column subplots
for row in range(n_rows):
    axs[row, 1].tick_params(axis='y', left=False, labelleft=False)

# Hide unused axes
for j in range(idx + 1, n_rows * n_cols):
    fig.delaxes(axs[j])

# fig.suptitle("Hill Climbed Runs\nLital Iteration against Hill Climbing Generation", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.legend(['Lital Iteration', 'Hard Instances'])
plt.savefig("plot/hill_climbed_runs/hill_climbed_runs_combined.png")
plt.show()
