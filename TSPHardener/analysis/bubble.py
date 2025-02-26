import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .load_json import load_json

def main():
    base_dirs = [
        "./Continuation",
        "./Results"
    ]
    config_data = []

    # Load and aggregate data
    for base_dir in base_dirs:
        pattern = os.path.join(base_dir, '**', '*.json')
        json_files = glob.glob(pattern, recursive=True)

        for file_path in json_files:
            data, errors, warnings = load_json(file_path)
            if data is None or errors:
                continue
            
            config = data.get('configuration', {})
            mutation = config.get('mutation_type', 'unknown')
            generation = config.get('generation_type', 'unknown')
            distribution = config.get('distribution', 'unknown')

            hard_instances = data['results'].get('hard_instances', {})
            hardest_values = [
                instance.get('hardest') 
                for instance in hard_instances.values() 
                if isinstance(instance.get('hardest'), (int, float))
            ]

            if hardest_values:
                config_data.append({
                    'mutation': mutation,
                    'generation': generation,
                    'distribution': distribution,
                    'mean_hardest': sum(hardest_values) / len(hardest_values),
                    'count': len(hardest_values)
                })

    if not config_data:
        print("No valid data found for plotting.")
        return

    df = pd.DataFrame(config_data)

    # Create bubble plot
    plt.figure(figsize=(12, 8))
    bubble_plot = sns.scatterplot(
        data=df,
        x='mutation',
        y='generation',
        size='mean_hardest',
        hue='distribution',
        sizes=(50, 100),  # Adjust min/max bubble sizes
        alpha=0.7,
        palette='viridis'
    )
    plt.title("Bubble Plot: Mean 'Hardest' Iterations by Configuration")
    plt.xlabel("Mutation Type")
    plt.ylabel("Generation Type")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save plot
    os.makedirs('./plot/bubble_config', exist_ok=True)
    plot_path = os.path.join('./plot/bubble_config', 'bubble_config.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()