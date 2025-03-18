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
    else:
        city_sizes = [20, 30]
        mutation_types = ['swap', 'scramble', 'inplace']
        tsp_type = ['euclidean', 'asymmetric']

        for tsp in tsp_type:
            for size in city_sizes:
                city_df = uniform_df[uniform_df['city_size'] == size]
                
                if city_df.empty:
                    print(f"No data for city_size={size}")
                    continue
                
                # statistics
                stats_df = (
                    city_df
                    .groupby('range')['iterations']
                    .agg(['min', 'max', 'median', 'mean', 'std'])
                    .reset_index()
                )                   

                stats_df = stats_df.sort_values('range')

                # Plot each statistic
                plt.figure(figsize=(12, 6))
                
                plt.plot(stats_df['range'], stats_df['min'], label='min', color='blue')
                plt.plot(stats_df['range'], stats_df['max'], label='max', color='red')
                plt.plot(stats_df['range'], stats_df['median'], label='median', color='green')
                plt.plot(stats_df['range'], stats_df['mean'], label='mean', color='orange')
                plt.plot(stats_df['range'], stats_df['std'], label='std', color='purple', linestyle='dashed')

                plt.scatter(
                    city_df['range'],
                    city_df['iterations'],
                    alpha=0.7
                )
                if dist == 'uniform':
                    plt.title(f"City Size {size}: $rand_{{max}}$ against Lital iterations ({dist} Distribution)")
                if dist == 'lognormal':
                    plt.title(f"City Size {size}: $\sigma$ against Lital iterations ({dist} Distribution)")
                plt.xlabel(r"$rand_{max}$")
                plt.ylabel("Lital Iterations")
                plt.grid(True)
                plt.legend()

                x_vals = sorted(city_df['range'].unique())
                plt.xticks(x_vals)
                plt.tight_layout()
                
                # Save the figure, or just call plt.show()
                plt.savefig(f'./plot/replication/range_vs_iteration_{dist}_{size}.png', bbox_inches='tight')
                plt.show()

if __name__ == "__main__":
    plot_iteration_range()