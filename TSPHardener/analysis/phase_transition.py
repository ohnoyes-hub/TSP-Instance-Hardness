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
lognormal_points = []
uniform_points = []

for entry in all_data:
    config = entry.get('configuration', {})
    distribution = config.get('distribution')
    range_val = config.get('range')
    results = entry.get('results', {})
    all_iterations = results.get('all_iterations', [])
    
    if not all_iterations:
        continue

    initial_iteration = all_iterations[0]

    if distribution == 'lognormal' and range_val in lognormal_ranges:
        lognormal_points.append({'range': range_val, 'iteration': initial_iteration})
    elif distribution == 'uniform' and range_val in uniform_ranges:
        uniform_points.append({'range': range_val, 'iteration': initial_iteration})

# Create DataFrames
df_lognormal = pd.DataFrame(lognormal_points)
df_uniform = pd.DataFrame(uniform_points)

# Plot lognormal distribution
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_lognormal, x='range', y='iteration')
plt.xlabel('Range (σ)')
plt.ylabel('Iterations to Solve Initial Matrix')
plt.title('Lognormal Distribution')
plt.grid(True)
plt.tight_layout()

# Save plot
folder_path = './plot/phase_transition'
os.makedirs(folder_path, exist_ok=True)
plot_path = os.path.join(folder_path, 'lognormal_distribution.png')
plt.show()

# Plot uniform distribution
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_uniform, x='range', y='iteration', color='orange')
plt.xlabel('Range (Max Distance)')
plt.ylabel('Iterations to Solve Initial Matrix')
plt.title('Uniform Distribution')
plt.grid(True)
plt.tight_layout()

# Save plot
plot_path = os.path.join(folder_path, 'uniform_distribution.png')
plt.savefig(plot_path, bbox_inches='tight')

plt.show()
