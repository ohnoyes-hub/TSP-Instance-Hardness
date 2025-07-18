import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from util.load_experiment import load_all_hard_instances
import seaborn as sns
def is_symmetric(matrix):
    """
    Quick check for matrix symmetry.
    """
    arr = np.array(matrix)
    return np.allclose(arr, arr.T, atol=1e-8)
    
def count_distinct_distances(matrix, symmetric=True):
    """
    Count distinct off-diagonal distances in a symmetric distance matrix.
    """
    arr = np.array(matrix)
    if symmetric:
        vals = arr[np.triu_indices_from(arr, k=1)]
    else:
        vals = arr[~np.eye(arr.shape[0], dtype=bool)]
    vals = vals[np.isfinite(vals)]
    return np.unique(vals).size
    # return len(np.unique(np.round(vals, decimals=6)))

def normed_unique(matrix, symmetric=True):
    n = len(matrix)
    total_possible = (n * (n - 1)) // 2 if symmetric else n * (n - 1)
    count = count_distinct_distances(matrix, symmetric)
    return count / total_possible if total_possible else np.nan

def normalized_nearest_neighbor_distance(matrix):
    arr = np.array(matrix).astype(float)
    np.fill_diagonal(arr, np.inf)
    nearest_distances = np.min(arr, axis=1)
    return np.mean(nearest_distances / np.mean(nearest_distances))

def analyze_by_config(df, config_col, output_prefix):
    summary = []
    for cfg in df[config_col].unique():
        sub = df[df[config_col] == cfg].copy()

        # Log scaling Lital iterations
        sub['log_iterations'] = np.log(sub['iterations'] + 1e-9) # +1e-9 to avoid log(0)

        # Detect symmetry for this config (assumes all same)
        test_matrix = sub.iloc[0]['matrix']
        symmetric = is_symmetric(test_matrix)

        # Precompute metrics
        sub['normed_unique_distances'] = sub['matrix'].apply(lambda m: normed_unique(m, symmetric))
        sub['num_distinct_distances'] = sub['matrix'].apply(lambda m: count_distinct_distances(m, symmetric))
        sub['nearest_neighbor_norm'] = sub['matrix'].apply(normalized_nearest_neighbor_distance)
        
        # Compute correlations - both raw and log scaled
        results = {
            'pearson_raw': pearsonr(sub['normed_unique_distances'], sub['iterations'])[0],
            'spearman_raw': spearmanr(sub['normed_unique_distances'], sub['iterations'])[0],
            'pearson_log': pearsonr(sub['normed_unique_distances'], sub['log_iterations'])[0],
            'spearman_log': spearmanr(sub['normed_unique_distances'], sub['log_iterations'])[0]
        }
        
        print(
            f"[{config_col}={cfg}]\n"
            f"Raw: Pearson={results['pearson_raw']:.3f}, Spearman={results['spearman_raw']:.3f}\n"
            f"Log: Pearson={results['pearson_log']:.3f}, Spearman={results['spearman_log']:.3f}"
        )

        # Scatter plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Raw iterations plot
        ax1.scatter(sub['iterations'], sub['normed_unique_distances'], alpha=0.7)
        ax1.set_xlabel('Iterations (raw)')
        ax1.set_ylabel('Normalized Distinct Distances')
        ax1.set_title(f"Raw Iterations - {cfg}")
        ax1.grid(True)
        
        # Log-scaled iterations plot
        ax2.scatter(sub['log_iterations'], sub['normed_unique_distances'], alpha=0.7)
        ax2.set_xlabel('Log Lital Iterations')
        ax2.set_ylabel('Normalized Distinct Distances')
        ax2.set_title(f"Log-scaled Iterations - {cfg}")
        ax2.grid(True)
        
        output_dir = "./plot/intercity_similarity"
        os.makedirs(output_dir, exist_ok=True)
        plt.tight_layout()
        plt_path = os.path.join(output_dir, f"{output_prefix}_{config_col}_{cfg}.png")
        plt.savefig(plt_path)
        plt.close()
        
        # Save data
        # Drop the matrix column before saving to CSV
        if 'matrix' in sub.columns:
            sub = sub.drop(columns=['matrix'])
        csv_path = os.path.join(output_dir, f"{output_prefix}_{config_col}_{cfg}.csv")  
        sub.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path} and plot\n")

        
        summary.append({
            config_col: cfg,
            "symmetric": symmetric,
            **results,
            "N": len(sub)
        })

    return pd.DataFrame(summary)

def plot_joint_normed_distinct_vs_log_iter(df, outpath="./plot/intercity_similarity/jointplot_normed_unique_vs_log_iter.png"):
    # Compute metrics if not present
    if 'normed_unique_distances' not in df.columns or 'log_iterations' not in df.columns:
        # Symmetry detection is per row!
        if 'normed_unique_distances' not in df.columns:
            df['normed_unique_distances'] = df.apply(
                lambda row: normed_unique(row['matrix'], is_symmetric(row['matrix'])), axis=1)
        if 'log_iterations' not in df.columns:
            df['log_iterations'] = np.log(df['iterations'] + 1e-9)
    
    # Seaborn jointplot with hue support (seaborn >= 0.11)
    plt.figure(figsize=(10, 10))
    jp = sns.jointplot(
        data=df,
        x="log_iterations",
        y="normed_unique_distances",
        hue="distribution",      # Color by generation_type
        kind="scatter",
        marginal_kws=dict(common_norm=False, fill=True),  # Separate marginals
        alpha=0.4,
        height=8,
        palette='CMRmap',
        marginal_ticks=True
    )
    jp.set_axis_labels("Log Lital Iteration", "Normalized Distinct Distances", fontsize=14)
    # jp.figure.suptitle("Normalized Distinct Distances vs. Log Lital Iteration\nMarginals by Distribution Type", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # room for suptitle

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath)
    plt.close()
    print(f"Saved jointplot to {outpath}")

def plot_joint_normed_nearest_neighbor_vs_log_iter(df, outpath="./plot/intercity_similarity/jointplot_nearest_neighbor_vs_log_iter.png"):
    # Compute metrics if not present
    if 'nearest_neighbor_norm' not in df.columns or 'log_iterations' not in df.columns:
        # Symmetry detection is per row!
        if 'nearest_neighbor_norm' not in df.columns:
            df['nearest_neighbor_norm'] = df.apply(
                lambda row: normalized_nearest_neighbor_distance(row['matrix']), axis=1)
        if 'log_iterations' not in df.columns:
            df['log_iterations'] = np.log(df['iterations'] + 1e-9)

    # Seaborn jointplot with hue support (seaborn >= 0.11)
    plt.figure(figsize=(10, 10))
    jp = sns.jointplot(
        data=df,
        x="log_iterations",
        y="nearest_neighbor_norm",
        hue="distribution",      # Color by generation_type
        kind="scatter",
        marginal_kws=dict(common_norm=False, fill=True),  # Separate marginals
        alpha=0.4,
        height=8,
        palette='CMRmap',
        marginal_ticks=True
    )
    jp.set_axis_labels("Log Lital Iteration", "Normalized Nearest Neighbor Distance", fontsize=14)
    # jp.figure.suptitle("Normalized Nearest Neighbor Distance vs. Log Lital Iteration\nMarginals by Distribution Type", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # room for suptitle

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath)
    plt.close()
    print(f"Saved jointplot to {outpath}")


def print_normed_unique_summary(df, threshold=0.6):
    # Ensure the relevant columns exist
    if 'normed_unique_distances' not in df.columns:
        df['normed_unique_distances'] = df.apply(
            lambda row: normed_unique(row['matrix'], is_symmetric(row['matrix'])), axis=1)
    if 'log_iterations' not in df.columns:
        df['log_iterations'] = np.log(df['iterations'] + 1e-9)

    summary_rows = []
    grouped = df.groupby(['generation_type', 'distribution'])
    for (gen_type, distr), group in grouped:
        count_leq = (group['normed_unique_distances'] <= threshold).sum()
        count_gt = (group['normed_unique_distances'] > threshold).sum()
        avg_log_leq = group.loc[group['normed_unique_distances'] <= threshold, 'log_iterations'].mean()
        avg_log_gt = group.loc[group['normed_unique_distances'] > threshold, 'log_iterations'].mean()
        summary_rows.append({
            'generation_type': gen_type,
            'distribution': distr,
            f'count_≤{threshold}': count_leq,
            f'avg_log_iter_≤{threshold}': avg_log_leq,
            f'count_>{threshold}': count_gt,
            f'avg_log_iter_>{threshold}': avg_log_gt,
        })
    summary_df = pd.DataFrame(summary_rows)
    pd.set_option('display.float_format', lambda x: f"{x:.2f}")
    print(summary_df)
    # Optionally save as CSV:
    summary_df.to_csv("./plot/intercity_similarity/normed_unique_summary.csv", index=False)
    print("Saved summary to ./plot/intercity_similarity/normed_unique_summary.csv")


def main():
    # Load experiments
    df = load_all_hard_instances()
    print_normed_unique_summary(df)
    if 'distribution' in df.columns:
        plot_joint_normed_distinct_vs_log_iter(df)
        plot_joint_normed_nearest_neighbor_vs_log_iter(df)

    # Option to analyze by different config columns
    for config_col in ['distribution', 'generation_type']:
        if config_col in df.columns:
            print(f"Analyzing by {config_col}...")
            summary = analyze_by_config(df, config_col=config_col, output_prefix='distinct_distances')
            print(summary)

if __name__ == '__main__':
    main()