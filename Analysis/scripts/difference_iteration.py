import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from icecream import ic
from scipy.stats import zscore
import numpy as np
import joypy

from util.load_experiment import load_json

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

def compute_differences(arr):
    """
    Computes the generation differences between iterations.
    """
    return [arr[i] - arr[i - 1] for i in range(1, len(arr))] if len(arr) > 1 else []

def difference_between_iterations():
    base_dirs = [
        "../data/Continuation",
        "../data/Results"
    ]
    analysis_data = []

    for base_dir in base_dirs:
        json_files = glob.glob(os.path.join(base_dir, '**', '*.json'), recursive=True)
        
        for file_path in json_files:
            data, errors, _ = load_json(file_path)
            if errors:
                ic("Skipped", file_path, errors)
                continue

            config = data.get('configuration', {})
            iterations = data.get('results', {}).get('all_iterations', [])

            diffs = compute_differences(iterations)
            if not diffs:
                continue

            analysis_data.extend([{
                'difference': diff,
                'mutation_type': config.get('mutation_type'),
                'generation_type': config.get('generation_type'),
                'distribution': config.get('distribution'),
                'source_file': os.path.basename(file_path)
            } for diff in diffs])

    df = pd.DataFrame(analysis_data)
    if df.empty:
        ic("No valid data found!")
        return

    # Z-score normalization of differences
    df['zscore_difference'] = zscore(df['difference'])

    # -----------------------
    # Violin plot by mutation type excluding 'scramble' with dotplot overlay
    # -----------------------
    # df_filtered = df[df['mutation_type'] != 'scramble']
    bar_data = (
        df.groupby('mutation_type')['difference']
        .agg(['mean', 'count', 'std'])
        .reset_index()
    )
    bar_data['sem'] = bar_data['std'] / np.sqrt(bar_data['count'])

    plt.figure(figsize=(8, 5))
    sns.barplot(
        x='mutation_type',
        y='difference',
        data=df,
        estimator=np.mean,
        errorbar=('se', 1),  # New Seaborn >=0.12; use ci=68 if old seaborn
        palette="pastel",
        capsize=0.2,
        edgecolor='k'
    )

    plt.xlabel("Mutation Type", fontsize=12, fontweight='bold')
    plt.ylabel("Mean Generation Difference(Lital Iterations)", fontsize=12, fontweight='bold')
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    # plt.title("Mean Generation Difference by Mutation Type", fontsize=13, fontweight='bold')
    plt.tight_layout()

    os.makedirs('./plot/iteration_diff', exist_ok=True)
    plot_path = os.path.join('./plot/iteration_diff', 'bar_mean_diff_iteration.png')
    plt.savefig(plot_path, bbox_inches='tight')
    ic("Saved filtered mutation violin plot with dotplot (vertical):", plot_path)
    plt.close()

    # print the average differences by mutation type
    avg_diffs = df.groupby('mutation_type')['difference'].mean().reset_index()
    print("\nAverage differences by mutation type:")
    for _, row in avg_diffs.iterrows():
        print(f"{row['mutation_type']}: {row['difference']:.2f}")

if __name__ == "__main__":
    difference_between_iterations()
