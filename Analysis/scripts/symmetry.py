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

def compute_symmetry_metrics(matrix):
    """
    How similar is the upper triangle to the lower triangle?
    """
    matrix = np.array(matrix)
    n = matrix.shape[0]
    asymmetry = np.abs(matrix - matrix.T)  # Compare M[i,j] vs. M[j,i]
    np.fill_diagonal(asymmetry, 0)  # Ignore diagonal
    
    symmetric_pairs = np.sum(asymmetry == 0) - n  # Subtract diagonal
    total_pairs = n * (n - 1)
    
    return {
        "symmetric_ratio": symmetric_pairs / total_pairs,
        "mean_asymmetry": np.mean(asymmetry[asymmetry > 0]) if np.any(asymmetry > 0) else 0,
        "max_asymmetry": np.max(asymmetry)
    }

def compute_symmetric_ratios():
    df_instances = load_initial_and_hard_instances()
    results = []

    for row in df_instances.itertuples():
        sym_metrics = compute_symmetry_metrics(row.matrix)
        out = {
            **sym_metrics,
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

    df_symmetry = pd.DataFrame(results)
    df_symmetry.to_csv("symmetric_ratios.csv", index=False)

def print_symmetric_ratio_counts(threshold=1.0, tol=1e-6):
    import pprint
    df = pd.read_csv("symmetric_ratios.csv")

    # Count exactly 1.0 values (with tolerance)
    is_one = (df['symmetric_ratio'] >= threshold - tol) & (df['symmetric_ratio'] <= threshold + tol)
    total_counts = df.groupby('mutation_type').size().to_dict()
    ones_counts = df[is_one].groupby('mutation_type').size().to_dict()

    # Compute scaled (normalized) counts
    scaled_counts = {}
    for mut, count in ones_counts.items():
        total = total_counts.get(mut, 1)  # Avoid divide by zero
        scaled_counts[mut] = count / total
    print("=== Symmetric Ratio = 1.0 Counts per Mutation Type ===")
    for mut in sorted(total_counts.keys()):
        raw = ones_counts.get(mut, 0)
        scaled = scaled_counts.get(mut, 0)
        print(f"{mut:20s} : count={raw:5d} / total={total_counts[mut]:5d} (fraction: {scaled:.3f})")

    # Optionally: print out all fractions for comparison
    print("\nComparison (fraction of ratio=1 per mutation type):")
    pprint.pprint(scaled_counts)

def plot_horizontal_jitter_symmetric_ratio():
    df = pd.read_csv("symmetric_ratios.csv")

    # Add small random jitter to symmetric_ratio = 1 to spread overlapping points
    mask_ones = df['symmetric_ratio'] == 1
    df.loc[mask_ones, 'symmetric_ratio'] = df.loc[mask_ones, 'symmetric_ratio'] + np.random.normal(0, 0.0075, mask_ones.sum())

    plt.figure(figsize=(10, 3))

    # Violin plot in the background
    sns.violinplot(
        data=df,
        x="symmetric_ratio",
        y="mutation_type",
        inner=None,          # No box plot/stat summary inside the violin
        linewidth=0.6,
        scale='width',
        palette="pastel",
        cut=0,
    )

    # Strip plot overlaid (jitter)
    sns.stripplot(
        data=df,
        x="symmetric_ratio",
        y="mutation_type",
        jitter=0.4,
        alpha=0.7,
        size=2,
        linewidth=0.5,
        palette="pastel",
    )

    plt.xlim(-0.05, 1.05)
    plt.xlabel("Symmetric Ratio")
    plt.ylabel("Mutation Type")
    # plt.title("Symmetric Ratio of Hardest Instances by Mutation Type")

    plt.grid(True, axis='x')
    plt.tight_layout()
    os.makedirs('./plot/symmetry', exist_ok=True)
    plt.savefig('./plot/symmetry/jitter_violin_plot_symmetric_ratio.png', bbox_inches='tight')
    plt.show()


# Run these to generate plot
# compute_symmetric_ratios()
print_symmetric_ratio_counts()
# === Symmetric Ratio = 1.0 Counts per Mutation Type ===
# inplace              : count= 2209 / total=10150 (fraction: 0.218)
# scramble             : count=  500 / total= 1320 (fraction: 0.379)
# swap                 : count= 2146 / total= 9973 (fraction: 0.215)

plot_horizontal_jitter_symmetric_ratio()