import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
from util.load_experiment import load_all_hard_instances

sns.set_theme(
        style="whitegrid",
        context="talk",
        palette="viridis",
        rc={
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 14,
            "font.size": 16,
            'axes.titleweight': 'bold',
            'axes.titlesize': 24,       # Title size for axes
            'axes.titleweight': 'bold', # Bold axis titles
            'axes.labelsize': 22,       # Axis label size
            'axes.labelweight': 'bold', # Bold axis labels
            'xtick.color': 'black',
            'ytick.color': 'black',
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'font.weight': 'bold',
        }
    )

df = load_all_hard_instances()

# Set log scale explicitly to avoid issues with zero values
df = df[(df['iterations'] > 0) & (df['optimal_cost'] > 0)]
df['log_iterations'] = np.log(df['iterations'])
df['log_optimal_cost'] = np.log(df['optimal_cost'])

# Plot jointplot with marginal distributions
g = sns.jointplot(
    data=df,
    x='log_iterations',
    y='log_optimal_cost',
    hue='city_size',
    kind='scatter',
    palette='pastel',
    marginal_kws=dict(fill=True),
    height=10,
    ratio=5,
    marginal_ticks=True
)

# plot labels
g.set_axis_labels('Log Total Iteration', 'Log Optimal Cost', fontsize=12)

# Set marginal axis labels
g.ax_marg_x.set_ylabel('Density', fontsize=12)  # Top marginal (x-axis distribution)
g.ax_marg_y.set_xlabel('Density', fontsize=12)  # Right marginal (y-axis distribution)

# Adjust title and layout
# plt.suptitle('Hardest TSP Instance\'s Optimal Cost against Their Lital Iteration\nMarginals by City Size', fontsize=14)
plt.tight_layout()
# plt.subplots_adjust(top=0.95)

# save plot
output_dir = "./plot/optimal_cost_v_iteration"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "optimal_cost_v_iteration.png")
plt.savefig(output_path)
print(f"Plot saved to {output_path}")
plt.show()

