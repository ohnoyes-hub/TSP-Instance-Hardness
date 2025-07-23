"""
In the following script I compute the Euclidean properties
effect on Lital iterations for Euclidean and Asymmetric TSPs.
"""
import numpy as np
from matplotlib.lines import Line2D
import os
from icecream import ic
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from util.load_experiment import load_all_hard_instances, load_initial_and_hard_instances

sns.set_theme(
        style="whitegrid",
        context="talk",
        palette="viridis",
        rc={
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "font.size": 14,
            'axes.titleweight': 'bold',
            'axes.titlesize': 24,       # Title size for axes
            'axes.titleweight': 'bold', # Bold axis titles
            'axes.labelsize': 14,       # Axis label size
            'axes.labelweight': 'bold', # Bold axis labels
            'xtick.color': 'black',
            'ytick.color': 'black',
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'font.weight': 'bold'
        }
    )

def triangle_inequality_violation(matrix):
    """
    Calculate the triangle inequality violation for a distance matrix.
    The triangle inequality states that for any three points A, B, and C,
    the distance from A to B should be less than or equal to the distance from A to C plus the distance from C to B.
    """
    matrix = np.array(matrix)
    n = matrix.shape[0]

    # Mask diagonals as they're infinite in TSP
    mask = ~np.eye(n, dtype=bool)

    violation_count = 0
    violation_magnitude = 0.0
    checks = 0

    for i in range(n):
        for j in range(n):
            if i == j:
                continue  # skip diagonal
            for k in range(n):
                if k == i or k == j:
                    continue  # avoid trivial checks

                checks += 1
                if matrix[i, j] > matrix[i, k] + matrix[k, j]:
                    violation_count += 1
                    violation_magnitude += matrix[i, j] - (matrix[i, k] + matrix[k, j])

    avg_violation = violation_magnitude / violation_count if violation_count else 0
    violation_ratio = violation_count / checks if checks else 0

    return {
        'violation_count': violation_count,
        'total_violation_magnitude': violation_magnitude,
        'average_violation_magnitude': avg_violation,
        'violation_ratio': violation_ratio
    }

def compute_tiq_for_all_instances():
    df_instances = load_initial_and_hard_instances()
    results = []
    for row in df_instances.itertuples():
        tiq = triangle_inequality_violation(row.matrix)
        out = {
            **tiq,
            "instance_type": row.instance_type,
            "mutation_type": getattr(row, "mutation_type", None),
            "city_size": getattr(row, "city_size", None),
            "distribution": getattr(row, "distribution", None),
            "generation_type": getattr(row, "generation_type", None),
            "range": getattr(row, "range", None),
            "iteration": getattr(row, "iteration", None),
            "generation": getattr(row, "generation", None),
            "optimal_cost": getattr(row, "optimal_cost", None),
        }
        results.append(out)
    df_tiq = pd.DataFrame(results)
    df_tiq.to_csv("triangle_inequality_violations_w_initial.csv", index=False)

def compute_and_save():
    df_hard_instances = load_all_hard_instances()
    results = []

    for row in df_hard_instances.itertuples():
        # row.matrix is the distance matrix
        tiq_vals = triangle_inequality_violation(row.matrix)

        # Build a dictionary combining tiq_vals
        # i.e. iteration= row.iterations, generation=row.generation, etc.

        config_data = {
            "distribution": row.distribution,
            "generation_type": row.generation_type,
            "city_size": row.city_size,
            "range": row.range,
            "mutation_type": row.mutation_type
        }
        combined_dict = {
            **tiq_vals,
            **config_data,
            "iteration": row.iterations,
            "generation": row.generation,
            "optimal_cost": row.optimal_cost
        }

        results.append(combined_dict)

    df_tiq = pd.DataFrame(results)

    # Save to CSV
    df_tiq.to_csv("triangle_inequality_violations.csv", index=False)
    ic("Saved triangle_inequality_violations.csv")


def plot_tiq_vs_iterations():
    tiq_values = ['violation_count', 'total_violation_magnitude', 'average_violation_magnitude', 'violation_ratio']
    df = pd.read_csv("triangle_inequality_violations.csv")
    mutation_types = df['mutation_type'].unique()
    distributions = df['distribution'].unique()
    city_sizes = df['city_size'].unique()
    df['log_iterations'] = np.log(df['iteration'])

    for tiq_value in tiq_values:
        pretty_name = tiq_value.replace('_', ' ').title()
        
        for mut_type in mutation_types:
            # Filter data for this mutation type
            df_filtered = df[df['mutation_type'] == mut_type]

            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                data=df_filtered,
                x='log_iterations',
                y=tiq_value,
                alpha=0.6
            )

            # Log scale for iterations
            plt.xscale('log')

            # Update labels and title using the pretty name and mutation type
            plt.xlabel("Log Lital Iterations")
            plt.ylabel(pretty_name)
            plt.title(f"{pretty_name} vs Iterations — Mutation: {mut_type}")

            # Grid and layout
            plt.grid(True)
            plt.tight_layout()

            # Save and display
            os.makedirs('./plot/tiq', exist_ok=True)
            safe_mut = mut_type.replace(' ', '_').lower()
            filename = f'./plot/tiq/scatter_{tiq_value}_{safe_mut}_vs_iterations.png'
            plt.savefig(filename, bbox_inches='tight')
            ic("Saved plot:", filename)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df,
            x='log_iterations',
            y=tiq_value,
            hue='mutation_type',
            alpha=0.6
        )
        # plt.xscale('log') 
        plt.xlabel("Log Lital Iterations")
        plt.ylabel(pretty_name)
        plt.title(f"Triangle Inequality {pretty_name} vs Lital Iterations (by Mutation Type)")
        plt.legend(title="Mutation Type")
        plt.grid(True)
        plt.tight_layout()

        os.makedirs('./plot/tiq', exist_ok=True)
        plt.savefig(f'./plot/tiq/scatter_{tiq_value}_vs_iterations.png', bbox_inches='tight')
        plt.show()

def plot_tiq_vs_iterations_with_cost_colorbar():
    tiq_values = ['violation_count', 'total_violation_magnitude', 'average_violation_magnitude', 'violation_ratio']
    df = pd.read_csv("triangle_inequality_violations.csv")
    df['log_iterations'] = np.log(df['iteration'])
    df['log_optimal_cost'] = np.log(df['optimal_cost'])

    for tiq_value in tiq_values:
        pretty_name = tiq_value.replace('_', ' ').title()

        plt.figure(figsize=(10, 6))
        # Use matplotlib scatter to allow colorbar for continuous variable
        scatter = plt.scatter(
            df['log_iterations'],
            df[tiq_value],
            c=df['log_optimal_cost'],
            cmap='viridis',
            alpha=0.7,
            label=None,
            edgecolors='none',
        )
        
        plt.xlabel("Log Lital Iterations")
        plt.ylabel(pretty_name)
        plt.title(f"{pretty_name} vs Iterations (color = log optimal cost)")
        plt.colorbar(scatter, label='Log Optimal Cost')
        plt.grid(True)
        plt.tight_layout()
        os.makedirs('./plot/tiq', exist_ok=True)
        plt.savefig(f'./plot/tiq/scatter_{tiq_value}_vs_iterations_by_distribution_cost.png', bbox_inches='tight')
        plt.close()
        # # Use hue for distribution, by grouping/distributing marker edgecolor or marker style
        # for dist in df['distribution'].unique():
        #     mask = df['distribution'] == dist
        #     plt.scatter(
        #         df.loc[mask, 'log_iterations'],
        #         df.loc[mask, tiq_value],
        #         edgecolor='black',
        #         label=dist,
        #         facecolors='none',  # Only edge colors shown for distinction
        #         alpha=0.7,
        #     )

        # plt.xlabel("Log Lital Iterations")
        # plt.ylabel(pretty_name)
        # plt.title(f"{pretty_name} vs Iterations by Distribution (color=log optimal cost)")
        # plt.legend(title="Distribution")
        # plt.colorbar(scatter, label='Log Optimal Cost')

        # plt.grid(True)
        # plt.tight_layout()

        # os.makedirs('./plot/tiq', exist_ok=True)
        # plt.savefig(f'./plot/tiq/scatter_{tiq_value}_vs_iterations_by_distribution_cost.png', bbox_inches='tight')
        # plt.close()

def plot_tiq_vs_iterations_by_instance_type():
    df = pd.read_csv("triangle_inequality_violations_w_initial.csv")
    df['log_iterations'] = np.log(df['iteration'].replace(0, np.nan))
    # df.loc[df['instance_type'] == 'initial', 'log_iterations'] = 0
    tiq_value = "violation_count"
    
    print(df['instance_type'].value_counts())
    print(df[df['instance_type'] == 'initial']['iteration'].describe())
    print(df[['instance_type', 'iteration', 'violation_count']].head(10))

    df['instance_type'] = pd.Categorical(df['instance_type'], categories=['hardest', 'initial'], ordered=True)
    df = df.sort_values('instance_type') 
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="log_iterations",
        y=tiq_value,
        hue="instance_type",
        style="instance_type",
        alpha=0.7,
        palette="binary_r"
    )
    plt.xlabel("Log Lital Iterations")
    plt.ylabel(f"Triangle Inequality {tiq_value.replace('_', ' ').title()}")
    # plt.title(f"Triangle Inequality {tiq_value.replace('_', ' ').title()} vs Log Lital Iterations\n(initial and hardest instances)")
    plt.legend(title="Instance Type")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'./plot/tiq/scatter_{tiq_value}_by_instance_type.png', bbox_inches='tight')
    plt.show()

# TODO: The hardest instances get overridden as initial instances 
def plot_tiq_vs_iteration_by_instance_type_with_gradient():
    df = pd.read_csv("triangle_inequality_violations_w_initial.csv")
    df['log_iterations'] = np.log(df['iteration'].replace(0, np.nan))
    tiq_value = "violation_count"

    # You need a grouping key. Let's use all columns that uniquely identify an experiment except 'instance_type', 'iteration', 'violation_count'
    group_cols = ['city_size', 'distribution', 'generation_type', 'range', 'mutation_type']  # add/remove columns as needed

    # Pivot the dataframe so initial and hardest are on same row
    pivot = df.pivot_table(
        index=group_cols,
        columns='instance_type',
        values=['log_iterations', tiq_value]
    ).dropna()

    # Plot scatter as before
    palette = {'initial': 'tab:orange', 'hardest': 'tab:blue'}

    plt.figure(figsize=(10, 6))

    # Plot initial first, then hardest
    sns.scatterplot(
        data=df[df['instance_type'] == 'initial'],
        x="log_iterations",
        y=tiq_value,
        color=palette['initial'],
        label='initial',
        alpha=0.7,
        marker='o'
    )
    sns.scatterplot(
        data=df[df['instance_type'] == 'hardest'],
        x="log_iterations",
        y=tiq_value,
        color=palette['hardest'],
        label='hardest',
        alpha=0.7,
        marker='X'
    )

    # Draw gradients as gray arrows
    for idx, row in pivot.iterrows():
        x0, y0 = row[('log_iterations', 'initial')], row[(tiq_value, 'initial')]
        x1, y1 = row[('log_iterations', 'hardest')], row[(tiq_value, 'hardest')]
        if not (np.isnan(x0) or np.isnan(y0) or np.isnan(x1) or np.isnan(y1)):
            plt.arrow(
                x0, y0, x1-x0, y1-y0,
                head_width=0.3, head_length=0.4, length_includes_head=True,
                color='gray', alpha=0.3, zorder=1
            )

    plt.xlabel("Log Lital Iterations")
    plt.ylabel(tiq_value.replace('_', ' ').title())
    plt.title(f"TIQ {tiq_value.replace('_', ' ').title()} vs Log Lital Iterations\n(initial to hardest gradient)")
    plt.legend(title="Instance Type")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'./plot/tiq/scatter_{tiq_value}_gradient_by_instance_type.png', bbox_inches='tight')
    plt.show()

def print_violaiton_counts_instance_type():
    df = pd.read_csv("triangle_inequality_violations_w_initial.csv")
    df['violation_count'] = df['violation_count'].fillna(0)
    zero_violation = df[df['violation_count'] == 0]

    grouped_counts = zero_violation.groupby(['instance_type']).size().reset_index(name='zero_violation_count')

    print(grouped_counts)

    mutation_types = ['swap', 'inplace', 'scramble']
    for mut in mutation_types:
        mut_df = zero_violation[zero_violation['mutation_type'] == mut]
        counts = mut_df.groupby(['instance_type']).size().reset_index(name='zero_violation_count')
        print(f"\nZero violations for mutation type '{mut}':")
        print(counts)
    generation_types = ['asymmetric', 'euclidean']
    for gen in generation_types:
        gen_df = zero_violation[zero_violation['generation_type'] == gen]
        counts = gen_df.groupby(['instance_type']).size().reset_index(name='zero_violation_count')
        print(f"\nZero violations for generation type '{gen}':")
        print(counts)

def plot_tiq_vs_optimal_cost_by_instance_type():
    df = pd.read_csv("triangle_inequality_violations_w_initial.csv")
    df = df[df['optimal_cost'] > 0]  # Avoid log(0)
    df['log_optimal_cost'] = np.log(df['optimal_cost'])
    tiq_value = "violation_count"  # Or any other TIQ metric if you prefer

    df['tsp_type'] = pd.Categorical(df['generation_type'], categories=['euclidean', 'asymmetric'], ordered=True)
    df = df.sort_values('tsp_type')

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="log_optimal_cost",
        y=tiq_value,
        hue="tsp_type",
        style="tsp_type",
        alpha=0.7,
        markers="o",
        palette="PRGn"
    )
    plt.xlabel("Log Optimal Cost")
    plt.ylabel(f"Triangle Inequality {tiq_value.replace('_', ' ').title()}")
    # plt.title(f"TIQ {tiq_value.replace('_', ' ').title()} vs Log Optimal Cost\n(initial and hardest instances)")
    plt.legend(title="TSP Type")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./plot/tiq/scatter_tiq_vs_optimal_cost_by_instance_type.png', bbox_inches='tight')
    plt.show()

plot_tiq_vs_optimal_cost_by_instance_type()
print_violaiton_counts_instance_type()
# compute_and_save()
# plot_tiq_vs_iteration_by_instance_type_with_gradient()
plot_tiq_vs_iterations_by_instance_type()
# plot_tiq_vs_iterations()
# plot_tiq_vs_iterations_with_cost_colorbar()
