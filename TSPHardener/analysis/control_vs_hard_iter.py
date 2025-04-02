import matplotlib.pyplot as plt
import seaborn as sns
from analysis_util.load_json import load_all_hard_instances

def plot_iteration_range(dist: str = 'uniform'):
    # Load all hardest instances with configurations
    df = load_all_hard_instances()

    # Filter for uniform distribution and valid range values
    uniform_df = df[
        (df['distribution'] == dist) &
        #(df['mutation_type'] == "inplace") & # TODO: create plots for each mutation type. Easier to see phase transition
        (df['range'].notna()) &
        (df['iterations'] > 0)
    ]

    if uniform_df.empty:
        print("No data to plot. Check your filters.")
        return
    
    # configurations
    city_sizes = [20, 30]
    mutation_types = ['swap', 'scramble', 'inplace']
    tsp_types = ['euclidean', 'asymmetric']

    
    for mutation in mutation_types:
        for tsp in tsp_types:
            for size in city_sizes:
                subset_df = uniform_df[
                    (uniform_df['mutation_type'] == mutation) &
                    (uniform_df['generation_type'] == tsp) &
                    (uniform_df['city_size'] == size)
                ]
                
                if subset_df.empty:
                    print(f"No data for mutation={mutation}, tsp_type={tsp}, city_size={size}")
                    continue

                # Compute statistics
                stats_df = (
                    subset_df
                    .groupby('range')['iterations']
                    .agg(['min', 'max', 'median', 'mean', 'std'])
                    .reset_index()
                ).sort_values('range')

                # Plot each statistic
                plt.figure(figsize=(12, 6))
                
                plt.plot(stats_df['range'], stats_df['min'], label='min', color='blue')
                plt.plot(stats_df['range'], stats_df['max'], label='max', color='red')
                plt.plot(stats_df['range'], stats_df['median'], label='median', color='green')
                plt.plot(stats_df['range'], stats_df['mean'], label='mean', color='orange')
                plt.plot(stats_df['range'], stats_df['std'], label='std', color='purple', linestyle='dashed')

                # Scatter plot of actual data points
                plt.scatter(subset_df['range'], subset_df['iterations'], alpha=0.7)

                # Title and labels
                if dist == 'uniform':
                    title = f"City Size {size}, Mutation: {mutation}, TSP: {tsp}\n$rand_{{max}}$ vs Lital Iterations ({dist} Distribution)"
                    plt.xlabel(r"$rand_{max}$")
                elif dist == 'lognormal':
                    title = f"City Size {size}, Mutation: {mutation}, TSP: {tsp}\n$\sigma$ vs Lital Iterations ({dist} Distribution)"
                    plt.xlabel(r"$\sigma$")

                plt.title(title)
                plt.ylabel("Lital Iterations")
                plt.grid(True)
                plt.legend()

                # X-axis ticks
                x_vals = sorted(subset_df['range'].unique())
                plt.xticks(x_vals)
                plt.tight_layout()

                # Save the figure
                filename = f'./plot/replication/range_vs_iteration_{dist}_{mutation}_{tsp}_{size}.png'
                plt.savefig(filename, bbox_inches='tight')
                print(f"Saved plot: {filename}")
                plt.close()

if __name__ == "__main__":
    plot_iteration_range()
    plot_iteration_range('lognormal')