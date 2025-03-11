import os
from icecream import ic
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

from .load_json import load_full, load_all_hard_instances

def plot_hard_instances():
    all_data = load_full()
    os.makedirs('plot/hardest_generation/individuaul', exist_ok=True)
    os.makedirs('plot/hardest_generation', exist_ok=True)

    # Individual Plots
    for data in all_data:
        config = data.get('configuration', {})
        hard_instances = data.get('results', {}).get('hard_instances', {})
        iterations_data = []

        for key, instance in hard_instances.items():
            if not key.startswith('iteration_'):
                continue
            try:
                iteration_num = int(key.split('_')[-1])
                iterations_val = instance.get('iterations')
                if isinstance(iterations_val, (int, float)):
                    iterations_data.append((iteration_num, iterations_val))
            except (ValueError, KeyError):
                continue

        if not iterations_data:
            continue

        iteration_nums, iterations = zip(*sorted(iterations_data, key=lambda x: x[0]))
        plt.figure()
        plt.plot(iteration_nums, iterations, 'o-')
        plt.xlabel('Generation')
        plt.ylabel('Iterations')
        config_str = ', '.join(f'{k}: {v}' for k, v in config.items())
        plt.title(f'Configuration: {config_str}')
        filename = '_'.join(f'{k}_{v}' for k, v in config.items()).replace('/', '_')
        plt.savefig(f'plot/hardest_generation/individuaul/{filename}.png', bbox_inches='tight')
        plt.close()

def plot_hard_instances_super():
    df = load_all_hard_instances()
    top_10 = df.nlargest(10, 'iterations')

    # Create configuration groups for legend
    df['config_group'] = df.apply(
        lambda row: (
            f"Mut: {row['mutation_type']}\n"
            f"Gen: {row['generation_type']}\n"
            f"Size: {row['city_size']}\n"
            f"Dist: {row['distribution']}"
        ), axis=1
    )

    # Sort by generation for proper line connections
    df_sorted = df.sort_values('generation')

    # Set up plot
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    # Create line plot with markers
    lineplot = sns.lineplot(
        data=df_sorted,
        x='generation',
        y='iterations',
        hue='config_group',
        marker='o',
        markersize=8,
        linewidth=1.5,
        palette='tab20'  # Use categorical color palette
    )

    # Add annotations for top 10 points
    for idx, row in top_10.iterrows():
        plt.annotate(
            f"G: {row['generation']}\nI: {row['iterations']}",
            (row['generation'], row['iterations']),
            xytext=(10, 10),
            textcoords='offset points',
            bbox=dict(boxstyle="round", alpha=0.9, facecolor='white'),
            fontsize=9
        )

    # Style plot
    plt.title("Iterations per Generation with Configuration Trends", pad=20)
    plt.xlabel("Generation Number")
    plt.ylabel("Iterations Required")

    # Adjust legend
    handles, labels = lineplot.get_legend_handles_labels()
    plt.legend(
        handles=handles,
        title='Configuration',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=9
    )
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join('plot/hardest_generation', 'all_configurations_vs_generation.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_hard_instances_super()
