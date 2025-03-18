import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
from ..analysis_util.load_json import load_json, load_all_hard_instances
                
def bubble_plot():
    df = load_all_hard_instances()
    df = df.dropna(subset=['matrix', 'optimal_cost'])

    # Convert 'inf' in diagonal to 0
    def clean_matrix(matrix):
        matrix = np.array(matrix)
        np.fill_diagonal(matrix, 0)
        return matrix

    # Calculate Nuclear Norm for each matrix
    df['norm'] = df['matrix'].apply(
        lambda x: np.linalg.norm(clean_matrix(x), ord='nuc')
    )

    # Scale optimal cost for better visualization
    df['scaled_optimal_cost'] = df['optimal_cost'] / 100000

    ic(df[['norm', 'scaled_optimal_cost']].head())

    # Create configuration identifier
    df['configuration'] = df['mutation_type'] + '_' + df['generation_type'] + '_' + df['distribution']

    # Generate bubble plot
    plt.figure(figsize=(14, 8))
    sns.scatterplot(
        data=df,
        x='iterations',
        y='scaled_optimal_cost',
        hue='configuration',
        size='norm',
        alpha=0.7,
        palette='tab20',
        sizes=(30, 300),  # Adjust bubble size range
        edgecolor='black'
    )

    # Customize plot
    plt.title('Lital Iterations vs. Scaled Optimal Cost (Size = Nuclear Norm)', fontsize=14)
    plt.xlabel('Lital Iterations', fontsize=12)
    plt.ylabel('Scaled Optimal Cost (÷ 100,000)', fontsize=12)
    plt.legend(
        title='Configuration',
        bbox_to_anchor=(1.25, 1),
        loc='upper left',
        frameon=True
    )
    plt.tight_layout()
    plt.savefig("./plot/bubble_plot.png")
    plt.show()

if __name__ == "__main__":
    bubble_plot()