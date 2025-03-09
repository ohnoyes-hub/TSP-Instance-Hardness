import matplotlib.pyplot as plt
import seaborn as sns
from .load_json import load_all_hard_instances

# Load all hardest instances with configurations
df = load_all_hard_instances()

# Filter for uniform distribution and valid range values
uniform_df = df[
    (df['distribution'] == 'uniform') &
    (df['range'].notna())
]

if uniform_df.empty:
    print("No data to plot. Check your filters.")
else:
    plt.figure(figsize=(12, 7))
    sns.scatterplot(
        data=uniform_df,
        x='range',
        y='iterations',
        alpha=0.7,
        palette='viridis'
    )
    plt.title(r"$rand_{max}$ vs. Iteration (Uniform Distribution)")
    plt.xlabel(r"Control Parameter($rand_{max}$)")
    plt.ylabel("Lital's Iteration")
    plt.grid(True)
    plt.tight_layout()
    
    # Save or display the plot
    plt.savefig('./plot/range_vs_iteration_uniform.png', bbox_inches='tight')
    plt.show()