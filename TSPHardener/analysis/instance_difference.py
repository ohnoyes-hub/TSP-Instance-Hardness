import numpy as np
import pandas as pd
from analysis_util.load_json import load_full
import seaborn as sns
import matplotlib.pyplot as plt
import os
from icecream import ic

def compute_sad_and_frobenius(initial, evolved):
    initial = np.array(initial)
    evolved = np.array(evolved)

    # Create a boolean mask to ignore diagonal (i.e., where i == j)
    n = initial.shape[0]
    mask = ~np.eye(n, dtype=bool)  # True for non-diagonal, False for diagonal

    # Apply the mask
    diff = evolved - initial
    diff_masked = diff[mask]

    sad = np.sum(np.abs(diff_masked))
    frob = np.sqrt(np.sum(diff_masked ** 2))
    return sad, frob

def compute_symmetry_metrics(matrix):
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

def collect_matrix_differences():
    data = load_full()
    records = []

    for entry in data:
        config = entry.get("configuration", {})
        results = entry.get("results", {})
        init_matrix = results.get("initial_matrix")

        if not init_matrix:
            continue

        # Track hard_instances differences
        hard_instances = results.get("hard_instances", {})
        for key, instance in hard_instances.items():
            evolved_matrix = instance.get("matrix")
            iterations = instance.get("iterations", None)

            if evolved_matrix and iterations is not None:
                sad, frob = compute_sad_and_frobenius(init_matrix, evolved_matrix)
                symmetry = compute_symmetry_metrics(evolved_matrix)

                record = {
                    "generation": key,
                    "iterations": iterations,
                    "sad": sad,
                    "frobenius": frob,
                    "symmetric_ratio": symmetry["symmetric_ratio"],
                    "mean_asymmetry": symmetry["mean_asymmetry"],
                    **config  # Includes generation_type, distribution, etc.
                }
                records.append(record)

        # Track last_matrix differences
        last_matrix = results.get("last_matrix", None)
        if last_matrix:
            print("analysing configuration", config)
            sad, frob = compute_sad_and_frobenius(init_matrix, last_matrix)
            record = {
                "generation": "last",
                "iterations": None,
                "sad": sad,
                "frobenius": frob,
                **config
            }
            records.append(record)

    df = pd.DataFrame(records)
    return df

def visualize_differences():
    # Load matrix difference if it exists
    df = pd.read_csv("matrix_differences.csv")
    df['sad'] = np.log(df['sad'])
    df['frobenius'] = np.log(df['frobenius'])

    # Individual Plots
    mutation_types = df['mutation_type'].unique()
    for mutation in mutation_types:
        sub_df = df[df['mutation_type'] == mutation]

        # SAD vs. Iterations
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=sub_df, x='iterations', y='sad', alpha=0.6)
        sns.regplot(data=sub_df, x='iterations', y='sad', scatter=False, color='black')
        plt.xscale('log')
        plt.title(f"SAD vs Iterations - {mutation}")
        plt.xlabel("Lital Iterations")
        plt.ylabel("Log SAD")
        plt.grid(True)
        plt.tight_layout()
        filename = f'scatter_sad_iterations_{mutation}.png'
        plt.savefig(os.path.join('./plot/instance_diff', filename), bbox_inches='tight')
        ic("Saved plot:", filename)
        plt.close()

        # Frobenius vs. Iterations
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=sub_df, x='iterations', y='frobenius', alpha=0.6)
        sns.regplot(data=sub_df, x='iterations', y='frobenius', scatter=False, color='black')
        plt.xscale('log')
        plt.title(f"Frobenius vs Iterations - {mutation}")
        plt.xlabel("Lital Iterations")
        plt.ylabel("Log Frobenius Norm")
        plt.grid(True)
        plt.tight_layout()
        filename = f'scatter_frobenius_iterations_{mutation}.png'
        plt.savefig(os.path.join('./plot/instance_diff', filename), bbox_inches='tight')
        ic("Saved plot:", filename)
        plt.close()

    # Scatter plot for SAD vs. iterations
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='iterations', y='sad', hue='mutation_type', alpha=0.6)
    sns.regplot(data=df, x='iterations', y='sad', scatter=False, color='black')
    plt.xscale('log')  # If needed
    plt.title("Sum of Absolute Difference of Initial Matrix and Hardest Matrix against Lital Iteration (by Mutation Type)")
    plt.xlabel("Lital Iterations")
    plt.ylabel("SAD")
    plt.legend(title='Mutation Type')
    plt.grid(True)
    plt.tight_layout()
    os.makedirs('./plot/instance_diff', exist_ok=True)
    plot_path = os.path.join('./plot/instance_diff', 'scatter_sad_iterations.png')
    plt.savefig(plot_path, bbox_inches='tight')
    ic("Saved plot:", plot_path)
    plt.show()

    # Repeat for Frobenius
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='iterations', y='frobenius', hue='mutation_type', alpha=0.6)
    sns.regplot(data=df, x='iterations', y='frobenius', scatter=False, color='black')
    plt.xscale('log')
    plt.xlabel("Lital Iterations")
    plt.ylabel("Frobenius Norm")
    plt.legend(title='City Size')
    plt.grid(True)
    plt.tight_layout()
    plt.title("Frobenius Norm of Initial and Hardest Matrices vs. Lital Iterations")
    plt.suptitle("By Mutation Type")
    plt.legend(title='City Size')
    plt.title("Frobenius vs. Iterations (by City Size)")
    # save plot
    plot_path = os.path.join('./plot/instance_diff', 'scatter_frobenius_iterations.png')
    plt.savefig(plot_path, bbox_inches='tight')
    ic("Saved plot:", plot_path)
    plt.show()

    # Violin plot for symmetric ratio
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x='mutation_type', y='symmetric_ratio', cut=0, scale='width')
    plt.title("Symmetric Ratio by Mutation Type")
    plt.xlabel("Mutation Type")
    plt.ylabel("Symmetric Ratio")
    plt.tight_layout()
    plt.savefig(os.path.join('./plot/instance_diff', 'violin_symmetric_ratio.png'), bbox_inches='tight')
    ic("Saved plot:", plot_path)
    plt.show()

    # TODO: change visualization of scatterplot, Generation Type: Euclidean would always on 0 so we are really interested in the asymmetric
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df, 
        x="mean_asymmetry", 
        y="iterations", 
        hue="generation_type",
        alpha=0.7
    )
    plt.title("Asymmetry vs. Iterations")
    plt.xlabel("Mean Asymmetry")
    plt.ylabel("Iterations")
    plt.legend(title='Generation Type')
    plt.tight_layout()
    plt.savefig(os.path.join('./plot/instance_diff', 'scatter_mean_asymmetry_iterations.png'), bbox_inches='tight')
    ic("Saved plot:", plot_path)
    plt.show()


if __name__ == "__main__":
    # df = collect_matrix_differences()
    # print(df.head())
    # df.to_csv("matrix_differences.csv", index=False)
    visualize_differences()

