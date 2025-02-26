import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from icecream import ic

from .load_json import load_json

def compute_differences(arr):
    """
    Computes the generation differences between iterations.
    """
    return [arr[i] - arr[i - 1] for i in range(1, len(arr))] if len(arr) > 1 else []

def difference_between_iterations():
    base_dirs = [
        "./Continuation", 
        "./Results"
    ]
    analysis_data = []

    for base_dir in base_dirs:
        json_files = glob.glob(os.path.join(base_dir, '**', '*.json'), recursive=True)
        
        for file_path in json_files:
            data, errors, _ = load_json(file_path)
            if errors:
                ic("Skipped", file_path, errors)
                continue

            # Extract key parameters
            config = data.get('configuration', {})
            iterations = data.get('results', {}).get('all_iterations', []) 

            # Compute differences PER FILE
            diffs = compute_differences(iterations)
            if not diffs:
                continue

            # Store with configuration context
            analysis_data.extend([{
                'difference': diff,
                'mutation_type': config.get('mutation_type'),
                'generation_type': config.get('generation_type'),
                'distribution': config.get('distribution'),
                'source_file': os.path.basename(file_path)
            } for diff in diffs])

    # Create DataFrame and plot
    df = pd.DataFrame(analysis_data)
    if df.empty:
        ic("No valid data found!")
        return

    # Statistics
    stats_df = df.groupby(
        ['mutation_type', 'generation_type', 'distribution']
    )['difference'].describe()
    print("\n=== Descriptive Statistics by Configuration ===")
    ic(stats_df)

    aggregated = df.groupby(
        ['mutation_type', 'generation_type', 'distribution']
    )['difference'].agg(['mean', 'std', 'median', 'min', 'max'])
    print("\n=== Custom Aggregated Stats ===")
    ic(aggregated)

    # Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='mutation_type', y='difference', data=df)
    plt.title("Generation Differences by Mutation Type")
    plt.xlabel("Mutation Type")
    plt.ylabel("Generation Difference")

    # Save plot
    os.makedirs('./plot/iteration_diff', exist_ok=True)
    plot_path = os.path.join('./plot/iteration_diff', 'boxplot_diff.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    # Violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='generation_type', y='difference', data=df, cut=0, scale='width')
    plt.title("Distribution of Generation Differences by Generation Type")
    
    # Save plot
    plot_path = os.path.join('./plot/iteration_diff', 'violin_diff.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    # Histogram
    plt.figure(figsize=(10, 6))
    plt.figure(figsize=(10, 6))
    sns.histplot(df['difference'], kde=True, bins=30)
    plt.title("Histogram of All Generation Differences")
    plt.xlabel("Generation Difference")
    plt.ylabel("Frequency")
    
    # Save plot
    plot_path = os.path.join('./plot/iteration_diff', 'hist_diff.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    difference_between_iterations()
