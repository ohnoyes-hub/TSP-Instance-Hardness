import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from .load_json import load_all_iteration

# Load data
df = load_all_iteration()

euclidean = df.get("euclidean", [])
asymmetric = df.get("asymmetric", [])

data = {
    'generation_type': ['euclidean'] * len(euclidean) + ['asymmetric'] * len(asymmetric),
    'iterations': euclidean + asymmetric
}

plt.figure(figsize=(10, 6))
sns.violinplot(x='generation_type', y='iterations', data=data, split=True)
plt.title('Comparison of Iterations: Euclidean vs. Asymmetric')
plt.savefig("./plot/comparing_euclidean_asymmetric/violinplot_euc_asy.png")

df = pd.DataFrame({
    "iterations": euclidean + asymmetric,
    "generation_type": ["euclidean"] * len(euclidean) + ["asymmetric"] * len(asymmetric)
})

# Plot
plt.figure(figsize=(10, 6))
sns.histplot(
    data=df,
    x="iterations",
    hue="generation_type",
    bins=30,  # Adjust bin count
    multiple="dodge",  # Bars side-by-side
    palette=["blue", "orange"],
    edgecolor="black"
)
plt.title("Comparison of All Lital Iterations: Euclidean vs. Asymmetric")
plt.xlabel("Lital Iterations")
plt.ylabel("Frequency")
plt.savefig("./plot/comparing_euclidean_asymmetric/histogram_euc_asy.png")
plt.show()