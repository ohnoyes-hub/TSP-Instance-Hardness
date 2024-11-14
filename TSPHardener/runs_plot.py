import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# # Load JSON data
# with open('../CleanData/run_summary.json') as f:
#     data = json.load(f)

# # # Prepare data for plotting
# plot_data = []

# for run in data['runs']:
#     config = run['configuration']
#     if (config['city_size'] == 20 and config['range'] == 0.2 and 
#         config['mutation_type'] == 'scramble' and config['generation_type'] == 'asymmetric' and 
#         config['distribution'] == 'lognormal'):
#         run_number = run['run_number']
#         hardest = run['hardest']
        
#         for generation, hardest in enumerate(hardest):
#             plot_data.append({
#                 'generation': generation,
#                 'hardest': hardest,
#                 'run_number': run_number
#             })

# # # Create a DataFrame
# df = pd.DataFrame(plot_data)

# Save the DataFrame to a CSV file
#df.to_csv('test_hardest_run_plot_file.csv', index=False)

# Load saved DataFrame from CSV file
df = pd.read_csv('../CleanData/test_hardest_run_plot_file.csv')

#print(df.keys())

# scale the iteration values to 0-4
#df['iteration'] = df['iteration'] / 4000
# normalize the generation values to 0-10 by dividing by 10000
#df['generation'] = df['generation'] / 10000


# Plot iterations vs generation for the specific configuration
sns.set(style='whitegrid')
plt.figure(figsize=(10, 6))
#ax = sns.lineplot(data=df, x='generation', y='iteration', hue='run_number', marker='o', markersize=3, linewidth=0.5)
ax = sns.lineplot(data=df, x='generation', y='hardest', hue='run_number', palette='dark', linewidth=0.5)

ax.set_title('Asymmetric Hardest Iterations vs Generation for Configuration (City Size 20, Range 0.2, Mutation Type Scramble)')
ax.set_xlabel('Generation')
ax.set_ylabel('Iteration')
plt.legend(title='Run Number')
plt.show()
plt.savefig('../Plot/test_hardest_individual_iterations_vs_generation.png')

# # Prepare data for plotting
# plot_data = []

# for run in data['runs']:
#     config = run['configuration']
#     city_size = config['city_size']
#     run_number = run['run_number']
#     #iterations = run['iterations']
#     hardest = run['hardest']
    
#     for generation, iteration in enumerate(iterations):
#         plot_data.append({
#             'city_size': city_size,
#             'generation': generation,
#             'iteration': iteration,
#             'run_number': run_number
#         })

# # Create a DataFrame
# df = pd.DataFrame(plot_data)

# # Set up the multiplot
# sns.set(style='whitegrid')
# g = sns.FacetGrid(df, col='city_size', hue='run_number', col_wrap=4, height=4, aspect=1.5)
# g.map(sns.lineplot, 'generation', 'iteration').add_legend()

# g.set_axis_labels('Generation', 'Iteration')
# g.set_titles('City Size: {col_name}')
# plt.subplots_adjust(top=0.9)
# g.figure.suptitle('Iterations vs Generation for Different Configurations')
# print("Plotting completed.")
# plt.show()
# plt.savefig('../Plot/test_iterations_vs_generation.png')