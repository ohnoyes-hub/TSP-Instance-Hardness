import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .load_json import load_json

def main():
    base_dirs = [
        #"./Continuation",
        "./Results"
    ]

    config_data = []

    for base_dir in base_dirs:
        pattern = os.path.join(base_dir, '**', '*.json')
        json_files = glob.glob(pattern, recursive=True)

        for file_path in json_files:
            data, errors, warnings = load_json(file_path)
            # Skip files with errors or no data
            if data is None or errors:
                continue
            
            config = data.get('configuration', {})
            mutation = config.get('mutation_type', 'unknown')
            generation = config.get('generation_type', 'unknown')
            distribution = config.get('distribution', 'unknown')

            hard_instances = data['results'].get('hard_instances', {})
            for instance in hard_instances.values():
                hardest_val = instance.get('hardest')
                if isinstance(hardest_val, (int, float)):
                    config_data.append({
                        'mutation_type': mutation,
                        'generation_type': generation,
                        'distribution': distribution,
                        'hardest': hardest_val
                    })

    if not config_data:
        print("No valid data found for plotting.")
        return

    df = pd.DataFrame(config_data)

    plt.figure(figsize=(18, 6))
    
    # Mutation Type
    plt.subplot(1, 3, 1)
    sns.violinplot(x='mutation_type', y='hardest', data=df)
    # change to swarmplot if (> 10000) points
    sns.stripplot(x='mutation_type', y='hardest', data=df, color='black', alpha=0.5) # comment to just see violin plot
    plt.title('By Mutation Type')
    plt.xlabel('Mutation Type')
    plt.ylabel("Lital's Iterations (hardest)")
    plt.xticks(rotation=45)

    # Generation Type
    plt.subplot(1, 3, 2)
    sns.violinplot(x='generation_type', y='hardest', data=df)
    sns.stripplot(x='generation_type', y='hardest', data=df, color='black', alpha=0.5) # comment to just see violin plot
    plt.title('By Generation Type')
    plt.xlabel('Generation Type')
    plt.ylabel('')
    plt.xticks(rotation=45)

    # Distribution
    plt.subplot(1, 3, 3)
    sns.violinplot(x='distribution', y='hardest', data=df)
    sns.stripplot(x='distribution', y='hardest', data=df, color='black', alpha=0.5) # comment to just see violin plot
    plt.title('By Distribution')
    plt.xlabel('Distribution')
    plt.ylabel('')
    plt.xticks(rotation=45)

    plt.suptitle("Distribution of Lital's Hardest Iterations by Configuration")
    plt.tight_layout()

    # Save plot
    os.makedirs('./plot/violin_config', exist_ok=True)
    plot_path = os.path.join('./plot/violin_config', 'stripplot_violin_by_config.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()