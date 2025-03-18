import matplotlib.pyplot as plt  
import numpy as np
import os
from ..analysis_util.load_json import load_full

distributions = {
    "uniform": np.arange(5, 105, 5),  # 5, 10, ..., 100
    "lognormal": np.arange(0.2, 5.2, 0.2)  # 0.2, 0.4, ..., 5.0
}

mutations = ["swap", "scramble", "inplace"]

TARGET_CITY_SIZE = 30

all_data = load_full()

# Load all experiment data
output_dir = F"./plot/heatmap/size{TARGET_CITY_SIZE}"
os.makedirs(output_dir, exist_ok=True)

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


for mutation in mutations:
    for distribution, ranges in distributions.items():
        for target_range in ranges:
            # Initialize lists to collect all data points
            all_x_vals = []
            all_y_vals = []

            for data in all_data:
                config = data.get('configuration', {})
                if (config.get('distribution') == distribution and
                    config.get('city_size') == TARGET_CITY_SIZE and
                    config.get('range') == target_range and
                    config.get('mutation_type') == mutation):

                    results = data.get('results', {})
                    all_iterations = results.get('all_iterations', [])
                    for generation, iterations in enumerate(all_iterations):
                        all_x_vals.append(generation / 200)  # Keep your original scaling
                        all_y_vals.append(iterations)

            all_x_vals = np.array(all_x_vals, dtype=float)
            all_y_vals = np.array(all_y_vals, dtype=float)

            # Determine data ranges for bins
            x_min, x_max = 0, max(all_x_vals) if len(all_x_vals) > 0 else 1
            y_min, y_max = 0, max(all_y_vals) if len(all_y_vals) > 0 else 1

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
            ax.set_title(f'Heatmap ({mutation}, {distribution}, Size: {TARGET_CITY_SIZE} Range: {target_range})')
            ax.set_xlabel('Generations')
            ax.set_ylabel('Lital Iterations (thousands)')

            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Log(Frequency)')

            plt.tight_layout()

            # Save frame
            filename = f"{output_dir}/heatmap_{mutation}_{distribution}_{target_range:.1f}.png"
            plt.savefig(filename)
            plt.close(fig)

print("All heatmap frames have been saved.")
