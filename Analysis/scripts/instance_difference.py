import numpy as np
import pandas as pd
from util.load_experiment import load_full
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
    df = pd.read_csv("../results/instance_differences.csv")
    df['sad'] = np.log(df['sad'])
    df['frobenius'] = np.log(df['frobenius'])
    df['log_iterations'] = np.log(df['iterations'])  # Avoid log(0)

    # Individual Plots
    mutation_types = df['mutation_type'].unique()
    for mutation in mutation_types:
        sub_df = df[df['mutation_type'] == mutation]

        # SAD vs. Iterations
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=sub_df, x='iterations', y='sad', alpha=0.5)
        sns.regplot(data=sub_df, x='log_iterations', y='sad', scatter=False, color='black')
        plt.xscale('log')
        plt.title(f"SAD against Log Lital Iterations - {mutation}")
        plt.xlabel("Log Lital Iterations")
        plt.ylabel("Log SAD")
        plt.legend(title='Mutation Type')
        plt.grid(True)
        plt.tight_layout()
        filename = f'scatter_sad_iterations_{mutation}.png'
        plt.savefig(os.path.join('./plot/instance_diff', filename), bbox_inches='tight')
        ic("Saved plot:", filename)
        plt.close()

        # Frobenius vs. Iterations
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=sub_df, x='log_iterations', y='frobenius', alpha=0.5)
        sns.regplot(data=sub_df, x='iterations', y='frobenius', scatter=False, color='black')
        plt.xscale('log')
        plt.title(f"Frobenius vs Log Lital Iterations - {mutation}")
        plt.xlabel("Log Lital Iterations")
        plt.ylabel("Log Frobenius Norm")
        plt.grid(True)
        plt.tight_layout()
        filename = f'scatter_frobenius_iterations_{mutation}.png'
        plt.savefig(os.path.join('./plot/instance_diff', filename), bbox_inches='tight')
        ic("Saved plot:", filename)
        plt.close()

    # Scatter plot for SAD vs. iterations
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='log_iterations', y='sad', hue='mutation_type', alpha=0.5)
    sns.regplot(data=df, x='log_iterations', y='sad', scatter=False, color='black')
    plt.xscale('log')  # If needed
    plt.title("Sum of Absolute Difference of Initial Matrix and Hardest Matrix against Log Lital Iteration (by Mutation Type)")
    plt.xlabel("Log Lital Iterations")
    plt.ylabel("SAD")
    plt.legend(title='Mutation Type')
    plt.grid(True)
    plt.tight_layout()
    os.makedirs('./plot/instance_diff', exist_ok=True)
    plot_path = os.path.join('./plot/instance_diff', 'scatter_sad_iterations.png')
    plt.savefig(plot_path, bbox_inches='tight')
    ic("Saved plot:", plot_path)
    # plt.show()

    # Repeat for Frobenius
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='iterations', y='frobenius', hue='city_size', alpha=0.5)
    sns.regplot(data=df, x='iterations', y='frobenius', scatter=False, color='black')
    plt.xscale('log')
    plt.xlabel("Lital Iterations")
    plt.ylabel("Frobenius Norm")
    plt.legend(title='City Size')
    plt.grid(True)
    plt.tight_layout()
    plt.title("Frobenius Difference of Initial and Hardest Instances vs. Lital Iterations(by City Size")
    plt.legend(title='City Size')
    # save plot
    plot_path = os.path.join('./plot/instance_diff', 'scatter_frobenius_iterations.png')
    plt.savefig(plot_path, bbox_inches='tight')
    ic("Saved plot:", plot_path)
    # plt.show()

    # Violin plot for symmetric ratio
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x='mutation_type', y='symmetric_ratio', cut=0, scale='width')
    plt.title("Symmetric Ratio by Mutation Type")
    plt.xlabel("Mutation Type")
    plt.ylabel("Symmetric Ratio")
    plt.tight_layout()
    plt.savefig(os.path.join('./plot/instance_diff', 'violin_symmetric_ratio.png'), bbox_inches='tight')
    ic("Saved plot:", plot_path)
    # plt.show()

    # Scatter of frobenius norm difference against sad
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='frobenius', y='sad', hue='distribution', alpha=0.5)
    sns.regplot(data=df, x='iterations', y='frobenius', scatter=False, color='black')
    plt.xscale('log')
    plt.xlabel("Frobenius Norm")
    plt.ylabel("SAD")
    plt.legend(title='Distribution')
    plt.grid(True)
    plt.tight_layout()
    plt.title("SAD against Frobenius Norm of Initial and Hardest Instances(by Distribution)")
    # save plot
    plot_path = os.path.join('./plot/instance_diff', 'scatter_frobenius_sad.png')
    plt.savefig(plot_path, bbox_inches='tight')
    ic("Saved plot:", plot_path)
    # plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df, 
        x="mean_asymmetry", 
        y="iterations", 
        hue="generation_type",
        alpha=0.5
    )
    plt.title("Asymmetry vs. Iterations")
    plt.xlabel("Mean Asymmetry")
    plt.ylabel("Iterations")
    plt.legend(title='Generation Type')
    plt.tight_layout()
    plt.savefig(os.path.join('./plot/instance_diff', 'scatter_mean_asymmetry_iterations.png'), bbox_inches='tight')
    ic("Saved plot:", plot_path)
    # plt.show()

    # Scatter of frobenius norm difference against iterations (by tsp_type)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='iterations', y='frobenius', hue='generation_type', alpha=0.5)
    sns.regplot(data=df, x='iterations', y='frobenius', scatter=False, color='black')
    plt.xscale('log')
    plt.xlabel("Lital Iterations")
    plt.ylabel("Frobenius Norm")
    plt.legend(title='TSP Type')
    plt.grid(True)
    plt.tight_layout()
    plt.title("Frobenius Difference of Initial and Hardest Instances against Lital Iterations (by TSP Type)")
    # save plot
    plot_path = os.path.join('./plot/instance_diff', 'scatter_frob_iteration_tsp_type.png')
    plt.savefig(plot_path, bbox_inches='tight')
    ic("Saved plot:", plot_path)
    # plt.show()
    plt.close()

def visualize_differences_improved():
    # Load matrix difference if it exists
    df = pd.read_csv("../results/instance_differences.csv")
    df['sad_log'] = np.log(df['sad'] + 1e-9)
    df['frobenius_log'] = np.log(df['frobenius'] + 1e-9)
    df['log_iterations'] = np.log(df['iterations'])  # Avoid log(0)
    df.loc[df['sad'] == 0, 'sad_log'] = 0
    os.makedirs('./plot/instance_diff', exist_ok=True)

    # Define a function for enhanced scatter plots with KDE contours and marginals
    def enhanced_scatter(df, x, y, hue, title, xlabel, ylabel, filename, 
                         regression=True, log_x=False, log_y=False):
        plt.figure(figsize=(10, 8))
        
        # Create main scatter plot with KDE contours
        g = sns.jointplot(
            data=df, x=x, y=y, hue=hue,
            kind='scatter',  # Base layer is scatter plot
            alpha=0.3,       # Semi-transparent points
            # palette='viridis',
            height=8,
            marginal_kws=dict(fill=True, alpha=0.7, common_norm=False),
            marginal_ticks=True
        )
        
        # Add KDE contours to the main plot
        # for hue_value in df[hue].unique():
        #     subset = df[df[hue] == hue_value]
        #     sns.kdeplot(
        #         data=subset, x=x, y=y, 
        #         levels=3, 
        #         alpha=0.6,
        #         linewidths=2,
        #         fill=False,
        #         bw_adjust=1.5,  # Adjust bandwidth for smoother contours
        #         ax=g.ax_joint,
        #         label=str(hue_value)  # Label for legend
        #     )
        
        # Add regression line: not working
        # if regression:
        #     sns.regplot(
        #         data=df, x=x, y=y, 
        #         scatter=False, 
        #         color='black',
        #         ax=g.ax_joint
        #     )
        
        # Set log scales
        # if log_x:
        #     g.ax_joint.set_xscale('log')
        #     valid = df[x][df[x] > 0]
        #     if not valid.empty:
        #         min_x = valid.min()
        #         g.ax_joint.set_xlim(left=min_x * 0.9)
        # if log_y:
        #     g.ax_joint.set_yscale('log')
        
        # Set labels and title
        g.ax_joint.set_xlabel(xlabel)
        g.ax_joint.set_ylabel(ylabel)
        # g.ax_marg_x.set_title(title, y=1.2)
        
        # Adjust layout and save
        plt.legend(title="Mutation Type")
        plt.tight_layout()
        plot_path = os.path.join('./plot/instance_diff', filename)
        g.savefig(plot_path, bbox_inches='tight')
        ic("Saved plot:", plot_path)
        plt.close()

    # 1. SAD vs Iterations (by mutation_type)
    enhanced_scatter(
        df=df,
        x='log_iterations',
        y='sad_log',
        hue='mutation_type',
        title="SAD vs Log Iterations\n(Mutation Type Marginals)",
        xlabel="Log Iterations",
        ylabel="Log SAD",
        filename='enhanced_sad_iterations_mutation.png',
        regression=True,
        log_x=True
    )

    # # 2. Frobenius vs Iterations (by city_size)
    enhanced_scatter(
        df=df,
        x='iterations',
        y='frobenius_log',
        hue='mutation_type',
        title="Frobenius vs Log Iterations\n(Mutation Type Marginals)",
        xlabel="Iterations",
        ylabel="Log Frobenius Norm",
        filename='enhanced_frobenius_iterations_city_size.png',
        regression=True,
        log_x=True
    )

    # # 3. SAD vs Frobenius (by distribution)
    enhanced_scatter(
        df=df,
        x='frobenius_log',
        y='sad_log',
        hue='distribution',
        title="SAD vs Frobenius Norm\n(Distribution Marginals)",
        xlabel="Frobenius Norm",
        ylabel="Log SAD",
        filename='enhanced_sad_frobenius_distribution.png',
        regression=False
    )

    # # 4. Asymmetry vs Iterations (by generation_type)
    # enhanced_scatter(
    #     df=df,
    #     x='mean_asymmetry',
    #     y='iterations',
    #     hue='generation_type',
    #     title="Asymmetry vs Iterations (by Generation Type) with Density Contours",
    #     xlabel="Mean Asymmetry",
    #     ylabel="Iterations",
    #     filename='enhanced_asymmetry_iterations_generation.png',
    #     regression=False,
    #     log_y=True
    # )

    # 5. Frobenius vs Iterations (by tsp_type)
    enhanced_scatter(
        df=df,
        x='iterations',
        y='frobenius_log',
        hue='mutation_type',
        title="Frobenius vs Iterations (by TSP Type) with Density Contours",
        xlabel="Iterations",
        ylabel="Log Frobenius Norm",
        filename='enhanced_frobenius_iterations_tsp_type.png',
        regression=True,
        log_x=True
    )

    # 6. SAD vs Optimal cost (by distribution)
    enhanced_scatter(
        df=df,
        x='optimal_cost',
        y='sad_log',
        hue='distribution',
        title="SAD vs Optimal Cost\nTSP Distribution Marginals",
        xlabel="Optimal Cost",
        ylabel="Log SAD",
        filename='enhanced_sad_optimal_cost_distribution.png',
        regression=False,
        log_x=True
    )

    # Keep your existing violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x='mutation_type', y='symmetric_ratio', cut=0, scale='width')
    plt.title("Symmetric Ratio by Mutation Type")
    plt.xlabel("Mutation Type")
    plt.ylabel("Symmetric Ratio")
    plt.tight_layout()
    plt.savefig(os.path.join('./plot/instance_diff', 'violin_symmetric_ratio.png'), bbox_inches='tight')
    ic("Saved plot:", 'violin_symmetric_ratio.png')

if __name__ == "__main__":
    # df = collect_matrix_differences()
    # print(df.head())
    # df.to_csv("../results/instance_differences.csv", index=False)
    # visualize_differences()
    visualize_differences_improved()

