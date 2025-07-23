import matplotlib.pyplot as plt
import seaborn as sns
from util.load_experiment import load_all_hard_instances
import os

df = load_all_hard_instances()  # loads all hard instances as DataFrame
# Count unique hard instances per mutation type (you may want to drop duplicates if needed)
mutation_counts = df['mutation_type'].value_counts().reset_index()
mutation_counts.columns = ['mutation_type', 'count']

plt.figure(figsize=(6,4))
sns.barplot(x='mutation_type', y='count', data=mutation_counts, order=['swap','inplace','scramble'], palette='pastel', edgecolor='k')
plt.xlabel('Mutation Strategy')
plt.ylabel('Number of Hard Instances')
# plt.title('Number of Hard Instances by Mutation Strategy')
plt.tight_layout()

folder = './plot/'
if not os.path.exists(folder):
    os.makedirs(folder)
# save the plot
plt.savefig(os.path.join(folder, 'hard_instances_by_mutation_strategy.png'))
print(f"Saved plot to {os.path.join(folder, 'hard_instances_by_mutation_strategy.png')}"    )
plt.show()