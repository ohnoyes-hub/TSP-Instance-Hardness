import os
import sys

from util.load_experiment import load_phase_transition_iterations
import scipy.stats as stats
import matplotlib.pyplot as plt

# 1) Load and filter
df = load_phase_transition_iterations()
print(df.head())
df = df[df['iteration'] > 0]    # drop any zeros or invalid entries

# 2) Define the keys we'll loop over
distributions = df['distribution'].unique()      # e.g. ['uniform','lognormal']
tsp_types     = ['euclidean','asymmetric']
city_sizes    = [20, 30]

for dist in distributions:
    sub_dist = df[df['distribution'] == dist]
    
    # a) Compare city sizes (20 vs 30) within each TSP type
    for tsp in tsp_types:
        group = sub_dist[sub_dist['generation_type'] == tsp]
        it20  = group[group['city_size']==20]['iteration']
        it30  = group[group['city_size']==30]['iteration']
        if it20.empty or it30.empty:
            continue
        
        # Mann–Whitney U test
        u_stat, p_val = stats.mannwhitneyu(it20, it30, alternative='two-sided')
        print(f"{dist.capitalize()} / {tsp.title()} | City 20 vs 30 → U={u_stat:.2f}, p={p_val:.3e}")
        
        # Overlaid histograms
        plt.figure()
        plt.hist(it20, bins=30, density=True, alpha=0.6, label='City 20')
        plt.hist(it30, bins=30, density=True, alpha=0.6, label='City 30')
        plt.xlabel('Lital Iterations')
        plt.ylabel('Density')
        plt.title(f"{dist.capitalize()} – {tsp.title()} – City Size Comparison")
        plt.legend()
        plt.show()
    
    # b) Compare TSP types (Euclidean vs Asymmetric) within each city size
    for size in city_sizes:
        group = sub_dist[sub_dist['city_size']==size]
        it_eu = group[group['generation_type']=='euclidean']['iteration']
        it_as = group[group['generation_type']=='asymmetric']['iteration']
        if it_eu.empty or it_as.empty:
            continue
        
        u_stat, p_val = stats.mannwhitneyu(it_eu, it_as, alternative='two-sided')
        print(f"{dist.capitalize()} / City {size} | Euclidean vs Asymmetric → U={u_stat:.2f}, p={p_val:.3e}")
