import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from util.load_experiment import load_json
from icecream import ic

def all_iteration():
    base_dirs = [
        #"./Continuation", 
        "./Results"
    ]

    # Collect all iteration data
    all_iterations = []

    for base_dir in base_dirs:
        pattern = os.path.join(base_dir, '**', '*.json')
        json_files = glob.glob(pattern, recursive=True)
        
        for file_path in json_files:
            data, errors, warnings = load_json(file_path)
            if errors:
                ic("Skipped,", file_path, errors) 
                continue
            
            # Extract 'all_iterations' from the JSON data
            results = data.get('results', {})
            iterations = results.get('all_iterations', [])
            
            if isinstance(iterations, list):
                for iter_num in iterations:
                    all_iterations.append({
                        'iteration': iter_num,
                        'source': os.path.basename(file_path)  # Track source file
                    })

    # Create DataFrame
    df = pd.DataFrame(all_iterations)

    if df.empty:
        ic("No valid iteration data found.")
    else:
        # Initialize plots
        sns.set_theme(style="whitegrid")
        
        # Histogram of all experiments
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='iteration', bins=30, kde=True)
        plt.title("Distribution of Iteration Numbers (All Experiments)")
        plt.xlabel("Lital's Iteration")
        plt.ylabel("Frequency")

        # Save histogram plot
        os.makedirs('./plot/all_iterations', exist_ok=True)
        plot_path = os.path.join('./plot/all_iterations', 'hist_all_iterations.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

        # Strip plot (individual points)
        sns.stripplot(data=df, x='iteration', jitter=0.3, alpha=0.5)
        plt.title("Individual Lital's Iteration Points")
        plt.xlabel("Lital's Iteration")
        
        # Save strip plot
        plot_path = os.path.join('./plot/all_iterations', 'strip_all_iterations.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
        # Box plot (distribution summary)
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='iteration')
        plt.title("Box Plot of All Lital Iterations")
        plt.xlabel("Lital's Iteration")
        
        # Save box plot
        plot_path = os.path.join('./plot/all_iterations', 'box_all_iterations.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

def all_iteration_config():
    base_dirs = [
        #"./Continuation", 
        "./Results"
    ]

    # configuration context
    data_records = []

    for base_dir in base_dirs:
        pattern = os.path.join(base_dir, '**', '*.json')
        json_files = glob.glob(pattern, recursive=True)
        
        for file_path in json_files:
            data, errors, warnings = load_json(file_path)
            if errors:
                ic("Error", file_path, errors)
                continue
            
            config = data.get('configuration', {})
            iterations = data.get('results', {}).get('all_iterations', [])
            
            if isinstance(iterations, list):
                for iter_num in iterations:
                    data_records.append({
                        'iteration': iter_num,
                        'mutation_type': config.get('mutation_type', 'unknown'),
                        'generation_type': config.get('generation_type', 'unknown'),
                        'distribution': config.get('distribution', 'unknown'),
                        'source_file': os.path.basename(file_path)
                    })

    df = pd.DataFrame(data_records)

    if df.empty:
        ic("No valid iteration data found.")
    else:
        sns.set_theme(style="whitegrid", palette="pastel")
        
        # Grouped Violin Plot
        plt.figure(figsize=(12, 7))
        sns.violinplot(data=df, x='mutation_type', y='iteration', 
                    hue='generation_type', split=True)
        plt.title("Lital's Iteration Distribution by Mutation and Generation Types")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save violin plot
        os.makedirs('./plot/all_iterations', exist_ok=True)
        plot_path = os.path.join('./plot/all_iterations', 'violin_all_iterations.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

        # 2. Faceted Histograms
        g = sns.FacetGrid(df, col='distribution', row='mutation_type', 
                        margin_titles=True, height=4, aspect=1.2)
        g.map(sns.histplot, 'iteration', kde=True, bins=15)
        g.set_axis_labels("Lital's Iteration", "Count")
        g.fig.suptitle("Lital's Iteration Distributions by Configuration", y=1.03)
        
        # Save facet grid plot
        plot_path = os.path.join('./plot/all_iterations', 'facet_all_iterations.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

        # 3. Swarm Plot with Multiple Categories
        plt.figure(figsize=(16, 10))
        # sns.stripplot(data=df, x='distribution', y='iteration', 
        #      hue='mutation_type', dodge=True, size=3, 
        #      jitter=0.3, alpha=0.5)
        # sns.violinplot(data=df, x='distribution', y='iteration', 
        #       hue='mutation_type', inner=None)
        # sns.swarmplot(data=df, x='distribution', y='iteration', 
        #             hue='mutation_type', size=2, color='black')
        # sns.swarmplot(data=df, x='distribution', y='iteration', 
        #             hue='mutation_type', dodge=True, size=1,
        #             linewidth=0.5, alpha=0.7) 
        # plt.title("Detailed Lital's Iteration Distribution Across Configurations")
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        # plt.tight_layout()

        # # Save swarm plot
        # plot_path = os.path.join('./plot/all_iterations', 'swarm_all_iterations.png')
        # plt.savefig(plot_path, bbox_inches='tight')
        # plt.close()


if __name__ == "__main__":
    all_iteration_config()