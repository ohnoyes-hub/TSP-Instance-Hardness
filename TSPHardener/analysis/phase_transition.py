import matplotlib.pyplot as plt
from analysis_util.load_json import load_phase_transition_iterations
from icecream import ic
import os

def plot_phase_transition(dist: str = 'uniform'):
    df = load_phase_transition_iterations()

    # Filter data
    filtered_df = df[
        (df['distribution'] == dist) &
        (df['range'].notna()) &
        (df['iteration'] > 0)
    ]

    if filtered_df.empty:
        ic("No data found", dist)
        return

    # Configuration parameters
    city_sizes = [20, 30]
    tsp_types = ['euclidean', 'asymmetric']

    for tsp_type in tsp_types:
        for size in city_sizes:
            # Subset data
            subset = filtered_df[
                (filtered_df['generation_type'] == tsp_type) &
                (filtered_df['city_size'] == size)
            ]

            if subset.empty:
                ic("No data:", tsp_type, size)
                continue

            # Calculate statistics
            stats = subset.groupby('range')['iteration'].agg(['max', 'median', 'mean', 'std']).reset_index().sort_values('range')

            # Plot setup
            plt.figure(figsize=(12, 6))
            
            # Plot lines
            plt.plot(stats['range'], stats['max'], label='Max')
            plt.plot(stats['range'], stats['median'], label='Median')
            plt.plot(stats['range'], stats['mean'], label='Mean')
            plt.plot(stats['range'], stats['std'], linestyle='--', label='Std Dev')

            # Scatter raw data
            plt.scatter(subset['range'], subset['iteration'], color='grey', alpha=0.3, label='Data Points')

            # Labels and title
            plt.xlabel(r"$rand_{max}$" if dist == 'uniform' else r"$\sigma$")
            plt.ylabel("Iterations")
            plt.title(f"City Size {size}, {tsp_type.title()} TSP Phase Transition ({dist.capitalize()})")
            plt.legend()
            plt.grid(True)
            plt.xticks(sorted(stats['range'].unique()))

            # Save plot
            os.makedirs('./plot/phase_transition', exist_ok=True)
            save_path = os.path.join('./plot/phase_transition',f'phase_transition_{dist}_{tsp_type}_{size}.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            ic("Saved:", save_path)

if __name__ == "__main__":
    plot_phase_transition('uniform')
    plot_phase_transition('lognormal')