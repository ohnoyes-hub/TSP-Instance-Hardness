import matplotlib.pyplot as plt
from analysis_util.load_json import load_full, load_all_hard_instances
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

# Plot for each configuration
for config_key, data_list in config_groups.items():
    config_dict = dict(config_key)
    
    plt.figure(figsize=(12, 7))
    
    # Plot all_iterations lines for each run in the configuration
    for idx, data in enumerate(data_list):
        all_iter = data.get('results', {}).get('all_iterations', [])
        if not all_iter:
            continue
        x = range(len(all_iter))
        label = 'All Iterations' if idx == 0 else None  # Avoid duplicate labels
        plt.plot(x, all_iter, marker='o', linestyle='-', markersize=4, alpha=0.5, label=label)
    
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
        plt.scatter(
            filtered_df['generation'], 
            filtered_df['iterations'], 
            color='red', 
            zorder=5, 
            label='Hard Instances'
        )
    
    # Formatting
    config_title = ', '.join([f"{k}: {v}" for k, v in config_dict.items()])
    plt.title(f"Hill Climbed Run Lital Iteration and Local Optima")
    plt.suptitle(f"Configuration: {config_title}")
    plt.xlabel("Generation")
    plt.ylabel("Lital Iterations")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plot/hill_climbed_runs/{config_title}.png")
    plt.close()