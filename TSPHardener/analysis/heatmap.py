import matplotlib.pyplot as plt  
import numpy as np
from .load_json import load_full

# Load all experiment data
all_data = load_full()

# Replicating the heatmap of Wouter
plt.rcParams.update({
    'font.size': 24,
    'figure.facecolor': 'black',
    'axes.facecolor': 'black',
    'axes.edgecolor': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'text.color': 'white',
    'legend.facecolor': 'black',
    'legend.edgecolor': 'white'
})

# Initialize lists to collect all data points
all_x_vals = []
all_y_vals = []

for data in all_data:
    results = data.get('results', {})
    all_iterations = results.get('all_iterations', [])
    for generation, iterations in enumerate(all_iterations):
        all_x_vals.append(generation)
        all_y_vals.append(iterations / 1000)  # Convert to thousands

all_x_vals = np.array(all_x_vals, dtype=float)
all_y_vals = np.array(all_y_vals, dtype=float)

# Determine data ranges for bins
x_min, x_max = 0, max(all_x_vals) if len(all_x_vals) > 0 else 0
y_min, y_max = 0, max(all_y_vals) if len(all_y_vals) > 0 else 0

# Create a 2D histogram (heatmap)
bins = [20, 13]  # Adjust bins as needed
heatmap, xedges, yedges = np.histogram2d(
    all_x_vals, all_y_vals,
    bins=bins,
    range=[[x_min, x_max], [y_min, y_max]]
)

# Convert to logarithmic scale
heatmap_log = np.log1p(heatmap)

# Plotting
fig, ax = plt.subplots(figsize=(20, 10))
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
im = ax.imshow(heatmap_log.T, extent=extent, origin='lower', aspect='auto', cmap='hot')

# Customize plot
ax.set_title('Experiment Heatmap')
ax.set_xlabel('Generations')
ax.set_ylabel('Little Iterations (thousands)')
ax.set_xticks(np.arange(x_min, x_max + 1, (x_max // 5)))  # Adjust ticks
ax.set_yticks(np.arange(y_min, y_max + 1, (y_max // 5)))

# Colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Log(Frequency)')

plt.tight_layout()
plt.savefig("./plot/heatmap.png")
plt.show()