import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from util.load_experiment import load_all_hard_instances

# Load data
df = load_all_hard_instances()

# Count zero entries in each matrix
def count_zeros(matrix):
    arr = np.array(matrix)
    return np.sum(arr == 0)

df['zero_count'] = df['matrix'].apply(count_zeros)

# Remove problematic zero or negative iterations if any
df = df[df['iterations'] > 0]

df['log_iterations'] = np.log(df['iterations'])

# Plot jointplot with marginal distributions
g = sns.jointplot(
    data=df,
    x='log_iterations',
    y='zero_count',
    hue='distribution',
    kind='scatter',
    palette='CMRmap',
    marginal_kws=dict(fill=True, common_norm=False, alpha=0.4),
    height=10,
    ratio=5,
    marginal_ticks=True
)

g.ax_marg_y.set_visible(False)

# Customize plot labels and title
g.set_axis_labels('Log Lital Iteration', 'Zero Count in Matrix', fontsize=12)
# plt.suptitle('Zero Occurrences in Hardest Matrices vs Log Lital Iterations\nMarginals by Distribution', fontsize=14)
plt.tight_layout()
plt.subplots_adjust(top=0.95)

# Save the plot
output_dir = "./plot/zero_entries"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "zero_entries_vs_log_iterations_joint.png")
plt.savefig(output_path)
print(f"Plot saved to {output_path}")
plt.show()

print("=========Summary Statistics by Distribution=========")
count_max = 200
for dist in df['distribution'].unique():
    sub = df[df['distribution'] == dist]
    count_eq0 = (sub['zero_count'] == 0).sum()
    count_gt100 = (sub['zero_count'] > count_max).sum()
    avg_logit_eq0 = sub.loc[sub['zero_count'] == 0, 'log_iterations'].mean()
    avg_logit_gt100 = sub.loc[sub['zero_count'] > count_max, 'log_iterations'].mean()
    print(f"Distribution: {dist}")
    print(f"  Zero count == 0: {count_eq0} (avg log(iter)={avg_logit_eq0:.2f})")
    print(f"  Zero count > {count_max}: {count_gt100} (avg log(iter)={avg_logit_gt100:.2f})\n")