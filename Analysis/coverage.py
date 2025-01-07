# import os
# import json
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np

# # # Define control parameters and corresponding file names
# control_parameters = list(range(5, 51, 5))
# file_names = [f'result20_{cp}_wouter.json' for cp in control_parameters]

# # Path to your JSON files (update this as needed)
# base_dir = "../CombinedData/uniform_asymmetric_wouter_size20/run1"

# # Initialize storage for iterations data
# iterations_data = {cp: [] for cp in control_parameters}

# # Load data from JSON files
# for cp, file_name in zip(control_parameters, file_names):
#     file_path = os.path.join(base_dir, file_name)
#     try:
#         with open(file_path, 'r') as f:
#             data = json.load(f)
#             # Calculate the number of iterations
#             count_iteration = len(data['results'].keys()) - 2  # Subtract 2 for 'initial_matrix' and 'iteration_0'
#             iterations_data[cp].append(count_iteration)
#     except Exception as e:
#         print(f"Error processing file {file_name}: {e}")

# # Prepare data for plotting
# control_param_values = []
# min_values = []
# max_values = []
# median_values = []
# std_values = []

# for cp, iterations in iterations_data.items():
#     if iterations:
#         control_param_values.append(cp)
#         min_values.append(min(iterations))
#         max_values.append(max(iterations))
#         median_values.append(np.median(iterations))
#         std_values.append(np.std(iterations))

# # Create the plot
# plt.figure(figsize=(12, 8))

# # Scatter plot for individual iteration data
# for cp, iterations in iterations_data.items():
#     plt.scatter([cp] * len(iterations), iterations, alpha=0.6, label=None)

# # Plot min, max, median, and standard deviation lines
# plt.plot(control_param_values, min_values, label='Min', color='blue')
# plt.plot(control_param_values, max_values, label='Max', color='orange')
# plt.plot(control_param_values, median_values, label='Median', color='red')
# plt.plot(control_param_values, std_values, label='Std', linestyle='dashed', color='cyan')

# # Customize the plot
# plt.yscale('log')  # Use a logarithmic scale for the y-axis
# plt.xlabel('Control Parameters (RandMax values)')
# plt.ylabel('Iterations (Little iterations)')
# plt.title('Iterations vs. Control Parameters')
# plt.legend()
# plt.tight_layout()
# plt.show()

import json
import glob
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_pattern = "result20_*_swap.json"

data_points = []

for filename in glob.glob(file_pattern):
    # Extract the control parameter from the filename
    match = re.search(r'result20_(\d+)_swap.json', filename)
    if match:
        control_param = int(match.group(1))
    else:
        continue

    # Load JSON data
    with open(filename, 'r') as f:
        data = json.load(f)

    # Iterate over each iteration key in the results
    for key, value in data.get('results', {}).items():
        if key.startswith("iteration_") and 'iterations' in value:
            iteration_value = value['iterations']
            data_points.append({
                'control_param': control_param,
                'iterations': iteration_value
            })

# Convert to a pandas DataFrame
df = pd.DataFrame(data_points)

# Create a scatter plot with seaborn
sns.scatterplot(data=df, x='control_param', y='iterations')

# Label the axes and show the plot
plt.xlabel("Control Parameter")
plt.ylabel("Iterations")
plt.title("Scatter Plot of Control Parameter vs. Iterations")
plt.show()