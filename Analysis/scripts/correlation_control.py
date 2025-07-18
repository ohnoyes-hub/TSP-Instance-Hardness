import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from util.load_experiment import load_phase_transition_iterations
from icecream import ic

def compute_correlations(subset: pd.DataFrame, dist: str, tsp_type: str, size: int):
    x = subset['range'].values
    y = subset['iteration'].values

    # Require at least 2 unique points
    if len(x) < 2 or np.all(x == x[0]):
        ic(f"Not enough variation for correlation: {dist}, {tsp_type}, size={size}")
        return None

    pearson_corr, pearson_p  = pearsonr(x, y)
    spearman_corr, spearman_p = spearmanr(x, y)

    return {
        'distribution': dist,
        'generation_type': tsp_type,
        'city_size': size,
        'pearson_r': pearson_corr,
        'pearson_p': pearson_p,
        'spearman_rho': spearman_corr,
        'spearman_p': spearman_p
    }

    # Pearson
    pearson_corr, pearson_p  = pearsonr(x, y)
    # Spearman
    spearman_corr, spearman_p = spearmanr(x, y)

    print(f"\n[{dist} | {tsp_type} | size={size}]")
    print(f"  Pearson r = {pearson_corr:.3f}, p = {pearson_p:.3e}")
    print(f"  Spearman ρ = {spearman_corr:.3f}, p = {spearman_p:.3e}")

def correlations_test_iteration_control():
    """
    Makes correlation test between range and iteration in the random sampling runs.
    """
    df = load_phase_transition_iterations()
    results = []

    city_sizes = [20, 30]
    tsp_types  = ['euclidean', 'asymmetric']

    for dist in ['uniform', 'lognormal']:
        df_dist = df[(df['distribution']==dist) & df['range'].notna() & (df['iteration']>0)]
        if df_dist.empty:
            ic("No data found for distribution", dist)
            continue

        for tsp_type in tsp_types:
            for size in city_sizes:
                subset = df_dist[(df_dist['generation_type']==tsp_type) & (df_dist['city_size']==size)]
                if subset.empty:
                    ic("No data for", tsp_type, size)
                    continue

                # Compute and collect correlation stats
                corr = compute_correlations(subset, dist, tsp_type, size)
                if corr:
                    results.append(corr)
    
    # --- Save the correlation table ---
    if results:
        results_df = pd.DataFrame(results)
        outdir = '../results'
        os.makedirs(outdir, exist_ok=True)
        csv_path  = os.path.join(outdir, 'correlations.csv')

        results_df.to_csv(csv_path, index=False)

        print(f"Saved correlation table to {csv_path}")
    else:
        print("No correlation results to save.")

if __name__ == "__main__":
    correlations_test_iteration_control()
