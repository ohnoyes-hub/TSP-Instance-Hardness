import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from .load_json import load_full

# Load data using the existing function
all_data = load_full()

# Extract relevant data points (including distribution, generation_type, mutation_type)
all_data = load_full()

# Extract relevant data points from hardest_instances
data_points = []
for entry in all_data:
    config = entry.get('configuration', {})
    range_val = config.get('range')
    distribution = config.get('distribution')
    generation_type = config.get('generation_type')
    mutation_type = config.get('mutation_type')
    results = entry.get('results', {})
    
    # Changed: Access hardest_instances instead of all_iterations
    hardest_instances = results.get('hardest_instances', [])
    for instance in hardest_instances:
        iter_val = instance.get('iteration')
        if iter_val is not None:  # Skip instances without iteration data
            data_points.append({
                'range': range_val,
                'iteration': iter_val,  # Changed key name
                'distribution': distribution,
                'generation_type': generation_type,
                'mutation_type': mutation_type
            })

# Create DataFrame and clean data
df = pd.DataFrame(data_points)
df['range'] = pd.to_numeric(df['range'], errors='coerce')
df['iteration'] = pd.to_numeric(df['iteration'], errors='coerce')  # New conversion
df = df.dropna(subset=['range', 'iteration']) 

# Calculate mean and median per group (include mutation_type)
mean_vals = df.groupby(['distribution', 'generation_type', 'range'])['iteration'].mean().reset_index()
median_vals = df.groupby(['distribution', 'generation_type', 'range'])['iteration'].median().reset_index()

# Create facet grid with distribution as columns and combined generation/mutation as rows
g = sns.relplot(
    data=df,
    x='range',
    y='iteration',
    col='distribution',
    row='generation_mutation',
    kind='scatter',
    alpha=0.6,
    height=4,
    aspect=1.2,
    facet_kws={'sharex': 'col', 'sharey': 'row'}
)

# Add mean and median lines to each subplot
for (generation_mutation_val, distribution_val), ax in g.axes_dict.items():
    # Split the combined row value into generation_type and mutation_type
    gen_type, mut_type = generation_mutation_val.split(' / ')
    
    # Filter mean data for current subplot
    mean_subset = mean_vals[
        (mean_vals['distribution'] == distribution_val) &
        (mean_vals['generation_type'] == gen_type) &
        (mean_vals['mutation_type'] == mut_type)
    ]
    sns.lineplot(
        data=mean_subset,
        x='range',
        y='iteration',
        color='red',
        linestyle='--',
        label='Mean',
        ax=ax
    )
    
    # Filter median data for current subplot
    median_subset = median_vals[
        (median_vals['distribution'] == distribution_val) &
        (median_vals['generation_type'] == gen_type) &
        (median_vals['mutation_type'] == mut_type)
    ]
    sns.lineplot(
        data=median_subset,
        x='range',
        y='iteration',
        color='green',
        linestyle=':',
        label='Median',
        ax=ax
    )

# Adjust legend and titles
g.set_titles("Distribution: {col_name} | Type/Mutation: {row_name}")
g.fig.suptitle("All Iterations Analysis by Configuration", y=1.02)
plt.savefig("./plot/all-iterations-analysis.png")
plt.show()