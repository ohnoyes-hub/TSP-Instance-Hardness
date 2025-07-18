""""
This script compares the Lital Iteration of different configurations against each other.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from util.load_experiment import load_phase_transition_iterations

# Load data
df = load_phase_transition_iterations()

# Define configuration pairs
config_pairs = [('distribution'), ('city_size', 'range')]

# Statistical test function
def statistical_comparison(df, group_by):
    groups = df.groupby(group_by)['iterations']
    results = {}

    # Assuming two groups comparison for simplicity
    unique_groups = df[group_by].drop_duplicates().values
    if len(unique_groups) != 2:
        raise ValueError("Exactly two groups needed for direct pair comparison")

    group1, group2 = unique_groups
    data1 = groups.get_group(group1)
    data2 = groups.get_group(group2)

    # Mann-Whitney U Test
    stat, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
    results['statistic'] = stat
    results['p_value'] = p

    print(f"Comparison {group_by}: {group1} vs {group2} | p-value: {p}")
    return results

# Run statistical comparisons
for pair in config_pairs:
    print(f"\nComparing configuration pair: {pair}")
    try:
        statistical_comparison(df, pair)
    except ValueError as e:
        print(e)

# Histogram Visualization Example
def plot_histogram(df, group_by, bins=30):
    plt.figure(figsize=(10,6))
    sns.histplot(data=df, x='iteration', hue=group_by, bins=bins, kde=True, alpha=0.6)
    plt.title(f'Iteration Distribution grouped by {group_by}')
    plt.xlabel('Iterations')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# Example call for histogram visualization
plot_histogram(df, 'city_size')
