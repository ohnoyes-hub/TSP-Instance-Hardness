import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from icecream import ic
from TSPHardener.analysis_util.load_experiment import load_all_hard_instances

import numpy as np
import scipy.stats as stats
from scipy.stats import mannwhitneyu, kruskal, ranksums
from statsmodels.stats.multitest import multipletests

def basic():
    df = load_all_hard_instances()

    if df['iterations'].isnull().any():
        print("Warning: Missing 'iterations' values. Dropping invalid rows.")
        df = df.dropna(subset=['iterations'])

    # Descriptive statistics grouped by configuration
    config_cols = ['mutation_type', 'generation_type', 'distribution', 'city_size', 'range']
    stats = df.groupby(config_cols)['iterations'].agg(
        ['mean', 'median', 'std', 'count', 'max', 'min']
    ).sort_values(by='median', ascending=False)

    print("=== Summary Statistics by Configuration ===")
    ic(stats)

    # Identify the best configuration (highest mean iterations)
    best_config = stats.index[0]
    ic(best_config, (stats.loc[best_config, 'median'], stats.loc[best_config, 'max']))

    # Save results to CSV
    stats.to_csv('./stats/configuration_stats.csv')



def pairwise_mannwhitney(df, group_col, target_col='iterations'):
    """
    Perform pairwise Mann-Whitney U tests between all groups in `group_col`.
    Returns a DataFrame with p-values and effect sizes (rank-biserial correlation).
    """
    groups = df[group_col].unique()
    comparisons = []
    p_values = []
    effect_sizes = []

    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            group1 = df[df[group_col] == groups[i]][target_col]
            group2 = df[df[group_col] == groups[j]][target_col]
            
            # Mann-Whitney U test
            stat, p = mannwhitneyu(group1, group2, alternative='two-sided')
            comparisons.append(f"{groups[i]} vs {groups[j]}")
            p_values.append(p)
            
            # Rank-biserial effect size
            n1, n2 = len(group1), len(group2)
            effect = 1 - (2 * stat) / (n1 * n2)  # Rank-biserial correlation
            effect_sizes.append(effect)

    # Adjust p-values for multiple comparisons (Bonferroni)
    reject, adj_pvals, _, _ = multipletests(p_values, method='bonferroni')
    
    results = pd.DataFrame({
        'Comparison': comparisons,
        'p-value': p_values,
        'Adj. p-value': adj_pvals,
        'Effect Size (rank-biserial)': effect_sizes
    })
    return results

def main():
    df = load_all_hard_instances()
    df = df.dropna(subset=['iterations'])
    
    # Create a unique 'configuration' column for all parameter combinations
    df['configuration'] = df.apply(
        lambda row: f"{row['mutation_type']}-{row['generation_type']}-{row['distribution']}",
        axis=1
    )
    
    # Example: Compare generation_type (euclidean vs asymmetric)
    print("=== Comparison: Euclidean vs Asymmetric Generation ===")
    euclidean = df[df['generation_type'] == 'euclidean']['iterations']
    asymmetric = df[df['generation_type'] == 'asymmetric']['iterations']
    stat, p = mannwhitneyu(euclidean, asymmetric, alternative='two-sided')
    effect = 1 - (2 * stat) / (len(euclidean) * len(asymmetric))  # Effect size
    ic(f"Mann-Whitney U p-value: {p:.4f}, Effect Size: {effect:.3f}\n")
    
    # Compare all configurations pairwise
    print("=== Pairwise Comparisons (All Configurations) ===")
    config_results = pairwise_mannwhitney(df, group_col='configuration')
    print(config_results.sort_values('Adj. p-value'))
    
    # Save results
    config_results.to_csv("./stats/pairwise_config_comparisons.csv", index=False)
    print("Results saved to 'pairwise_config_comparisons.csv'")

if __name__ == "__main__":
    main()