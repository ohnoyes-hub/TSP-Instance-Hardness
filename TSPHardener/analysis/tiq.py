import numpy as np
from matplotlib.lines import Line2D
import os
import json
from utils.json_utils import custom_decoder
from icecream import ic
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .load_json import load_json

def compute_triangle_inequality_metrics(matrix):
    matrix = np.array(matrix)
    n = len(matrix)
    violations = []
    
    # Sample 1000 triplets to reduce computation (adjust as needed)
    for _ in range(1000):
        i, j, k = np.random.choice(n, 3, replace=False)
        direct = matrix[i, k]
        indirect = matrix[i, j] + matrix[j, k]
        if indirect == 0:
            continue  # Avoid division by zero
        ratio = direct / indirect
        if ratio > 1:
            violations.append(ratio)
    
    if not violations:
        return {
            "avg_violation": 0,
            "max_violation": 0,
            "violation_freq": 0
        }
    
    return {
        "avg_violation": np.mean(violations),
        "max_violation": np.max(violations),
        "violation_freq": len(violations) / 1000 
    }

def main():
    base_dirs = [
        #"./Continuation",
        "./Results"
    ]

    total_files = 0
    error_files = 0
    warning_files = 0
    detailed_issues = []

    for base_dir in base_dirs:
        pattern = os.path.join(base_dir, '**', '*.json')
        json_files = glob.glob(pattern, recursive=True)
        
        for file_path in json_files:
            total_files += 1
            data, errors, warnings = load_json(file_path)

            # TODO: move checker to move to a error collection function
            if errors or warnings:
                entry = {
                    "file": file_path,
                    "errors": errors,
                    "warnings": warnings
                }
                detailed_issues.append(entry)
                if errors:
                    error_files += 1
                if warnings:
                    warning_files += 1

            # Generate plot only if data is valid
            if data is not None and not errors:
                # Compute triangle inequality metrics
                matrix = data['results']['last_matrix']
                metrics = compute_triangle_inequality_metrics(matrix)

                hard_instances = data['results']['hard_instances']
                plot_data = []
                
                # Determine matrix type (Euclidean or other)
                matrix_type = data['configuration'].get('distribution', 'unknown')
                is_euclidean = 'euclidean' in matrix_type.lower()
                
                # Collect hard instance data + metrics
                for key, instance in hard_instances.items():
                    if not key.startswith('iteration_'):
                        continue
                    try:
                        iteration_num = int(key.split('_')[-1])
                    except ValueError:
                        continue

                    iterations_val = instance.get('iterations')
                    hardest_val = instance.get('hardest')
                    if not isinstance(iterations_val, (int, float)) or not isinstance(hardest_val, (int, float)):
                        continue

                    plot_data.append({
                        'iteration': iteration_num,
                        'iterations': iterations_val,
                        'hardest': hardest_val,
                        'avg_violation': metrics['avg_violation'],
                        'max_violation': metrics['max_violation'],
                        'violation_freq': metrics['violation_freq'],
                        'is_euclidean': is_euclidean 
                    })
    
    df = pd.DataFrame(plot_data)

    # Print summary
    ic("=== Validation Summary ===")
    ic(total_files)
    ic(error_files)
    ic(warning_files)
    if detailed_issues:
        ic("=== Detailed Issues ===")
        for issue in detailed_issues:
            ic(issue['file'])
            if issue['errors']:
                ic("Errors:")
                for err in issue['errors']:
                    ic(err)
            if issue['warnings']:
                ic("Warnings:")
                for warn in issue['warnings']:
                    ic(warn)

    # Violation frequncy vs. iterations (scatter plot)
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df,
        x='violation_freq',
        y='iterations',
        hue='is_euclidean',
        palette={True: 'blue', False: 'red'},
        alpha=0.7
    )
    plt.title("Triangle Inequality Violations vs. Lital Iterations")
    plt.xlabel("Violation Frequency")
    plt.ylabel("Iterations")
    plt.legend(title='Matrix Type', labels=['Euclidean', 'Non-Euclidean'])
    plt.savefig('./plot/violations_vs_iterations.png')
    plt.close()

    # Max violation vs. matrix type (boxplot)
    plt.figure(figsize=(12, 8))
    sns.boxplot(
        data=df,
        x='is_euclidean',
        y='max_violation',
        palette={True: 'blue', False: 'red'}
    )
    plt.title("Max Violation by Matrix Type")
    plt.xlabel("Is Euclidean?")
    plt.ylabel("Max Violation Ratio")
    plt.xticks([0, 1], ['Non-Euclidean', 'Euclidean'])
    plt.savefig('./plot/max_violation_boxplot.png')
    plt.close()

    # pair plot for all metric
    g = sns.pairplot(
        df,
        vars=['avg_violation', 'max_violation', 'iterations', 'hardest'],
        hue='is_euclidean',
        palette={True: 'blue', False: 'red'},
        plot_kws={'alpha': 0.5}
    )
    g.fig.suptitle("Pairwise Relationships Between TIQ Metrics", y=1.02)
    plt.savefig('./plot/tiq_pairplot.png')
    plt.close()

if __name__ == '__main__':
    main()