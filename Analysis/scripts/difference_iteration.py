import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from icecream import ic
from scipy.stats import zscore

from util.load_experiment import load_json

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

    plt.figure(figsize=(10, 2.5))
    sns.violinplot(
        y='mutation_type',     # <-- changed from x=
        x='zscore_difference', # <-- changed from y=
        data=df,
        cut=0,
        scale='width',
        palette="pastel",
        linewidth=0.6,
    )
    # Overlay dotplot (stripplot) for individual data points
    sns.stripplot(
        y='mutation_type',     # <-- changed from x=
        x='zscore_difference', # <-- changed from y=
        data=df,
        color='k',
        size=2,
        jitter=0.4,
        alpha=0.7,
        palette="pastel"
    )
    # plt.title("Z-score Scaled Generation Differences by Mutation Type")
    plt.ylabel("Mutation Type")
    plt.xlabel("Generation Difference of Lital Iterations (Z-score Scaled)")

    os.makedirs('./plot/iteration_diff', exist_ok=True)
    plot_path = os.path.join('./plot/iteration_diff', 'violin_zscore_diff_iteration_vertical.png')
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
