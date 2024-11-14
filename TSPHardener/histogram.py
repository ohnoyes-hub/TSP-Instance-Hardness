import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the JSON file
with open('../CleanData/aggregated_summary.json') as f:
    data = json.load(f)

# Prepare data for the histogram
mutation_types = []
distributions = []
generation_types = []
iterations_counts = []

# Iterate through runs in the data
for run in data['runs']:
    config = run['configuration']
    mutation_types.append(config['mutation_type'])
    distributions.append(config['distribution'])
    generation_types.append(config['generation_type'])
    iterations_counts.append(len(run['iterations']))  # Number of iterations

# Create a DataFrame for plotting
df = pd.DataFrame({
    'Mutation Type': mutation_types,
    'Distribution': distributions,
    'Generation Type': generation_types,
    'Iterations': iterations_counts
})

df['Mutation + Generation Type'] = df['Mutation Type'] + " - " + df['Generation Type']

# Plotting using Seaborn
# plt.figure(figsize=(14, 8))
# sns.barplot(data=df, x='Mutation + Generation Type', y='Iterations', hue='Distribution', ci=None, dodge=True)
# plt.xticks(rotation=45, ha='right')
# plt.xlabel('Mutation Type')
# plt.ylabel('Number of Iterations')
# plt.title('Number of Iterations by Mutation and Generation Type, Differentiated by Distribution')
# plt.tight_layout()
# plt.show()
# plt.savefig('../Plots/iterations_histogram.png')

# Plotting using Seaborn
plt.figure(figsize=(14, 8))
sns.barplot(data=df, y='Mutation + Generation Type', x='Iterations', hue='Distribution', ci=None, dodge=True)
plt.ylabel('Mutation + Generation Type')
plt.xlabel('Number of Iterations')
plt.title('Number of Iterations by Mutation and Generation Type, Differentiated by Distribution')
plt.tight_layout()
plt.show()
plt.savefig('../Plots/iterations_histogram.png')