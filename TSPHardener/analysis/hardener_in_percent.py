import pandas as pd
from icecream import ic
from analysis_util.load_json import load_all_hard_instances

# Shows by how much the final instance is harder than the baseline per configuration

# Load data with configurations and file paths
df = load_all_hard_instances() #include_file_path=True)

# Extract baseline (generation=0) for each run
baselines = df[df['generation'] == 0][['file_path', 'iterations']]
baselines = baselines.rename(columns={'iterations': 'baseline_iterations'})

# Merge baseline into main DataFrame
merged = df.merge(baselines, on='file_path')

# Skip generation=0 (baseline)
merged = merged[merged['generation'] > 0]

# Calculate percentage increase in 'iterations' (hardness proxy)
merged['percentage_increase'] = (
    (merged['iterations'] - merged['baseline_iterations']) / merged['baseline_iterations']
) * 100

# Add a binary flag for whether the final instance was harder
merged['is_harder'] = merged['iterations'] > merged['baseline_iterations']

# --- Summary Statistics ---

# Overall percentage of runs that got harder
harder_percent = merged['is_harder'].mean() * 100
print(f"Overall % of harder instances: {harder_percent:.2f}%")

# Grouped statistics by mutation type, city size, and TSP type
grouped = merged.groupby(['mutation_type', 'city_size', 'generation_type']).agg({
    'percentage_increase': 'mean',
    'is_harder': 'mean'
}).reset_index()

# Convert 'is_harder' to percentage
grouped['is_harder'] = (grouped['is_harder'] * 100).round(2)
grouped['percentage_increase'] = grouped['percentage_increase'].round(2)

# Rename for clarity
grouped = grouped.rename(columns={
    'percentage_increase': 'Avg % Increase in Iterations',
    'is_harder': '% of Harder Instances'
})

print(grouped)
