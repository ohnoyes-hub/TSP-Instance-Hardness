import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from analysis_util.load_json import load_full

# Load all experiment data
all_data = load_full()

# Define valid ranges for each distribution
lognormal_ranges = [round(0.2 * i, 1) for i in range(1, 26)]  # 0.2 to 5.0
uniform_ranges = list(range(5, 101, 5))                      # 5 to 100

# Collect data points
lognormal_points = {'euclidean': [], 'asymmetric': []}
uniform_points = {'euclidean': [], 'asymmetric': []}

for entry in all_data:
    config = entry.get('configuration', {})
    distribution = config.get('distribution')
    range_val = config.get('range')
    gen_type = config.get('generation_type')
    results = entry.get('results', {})
    all_iterations = results.get('all_iterations', [])

    if not all_iterations or gen_type not in ['euclidean', 'asymmetric']:
        continue

    first_50 = all_iterations[:30]

    if distribution == 'lognormal' and range_val in lognormal_ranges:
        for iteration in first_50:
            lognormal_points[gen_type].append({'range': range_val, 'iteration': iteration})
    elif distribution == 'uniform' and range_val in uniform_ranges:
        for iteration in first_50:
            uniform_points[gen_type].append({'range': range_val, 'iteration': iteration})

# Create and plot for each combination
for dist_type, points_dict in [('lognormal', lognormal_points), ('uniform', uniform_points)]:
    for gen_type, points in points_dict.items():
        if not points:
            continue
        df = pd.DataFrame(points)
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x='range', y='iteration', alpha=0.6)
        plt.xlabel('Range (σ)' if dist_type == 'lognormal' else 'Range (Max Distance)')
        plt.ylabel('Iterations to Solve Initial Matrix')
        plt.title(f'{dist_type.capitalize()} Distribution - {gen_type.capitalize()}')
        plt.grid(True)
        plt.tight_layout()

        # Save plot
        folder_path = './plot/phase_transition'
        os.makedirs(folder_path, exist_ok=True)
        filename = f'{dist_type}_{gen_type}_distribution.png'
        plot_path = os.path.join(folder_path, filename)
        plt.savefig(plot_path, bbox_inches='tight')
        plt.show()

