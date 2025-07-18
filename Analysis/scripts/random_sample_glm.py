"""
This script performs a Generalized Linear Model (GLM) analysis on a random sampled runs.
The purpose is to compare the difference between city sizes 20 and 30 in uniform and lognormal configurations.
The test would give us an effect size for city sizes on instace hardness.
"""
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from util.load_experiment import load_phase_transition_iterations
from icecream import ic
import numpy as np
import os

from scipy.stats import mannwhitneyu

# Configuration parameters
city_sizes = [20, 30]
tsp_types = ['euclidean', 'asymmetric']
output_dir = "./plot/histograms_random_sampling/"
os.makedirs(output_dir, exist_ok=True)
max_freq = 350  # maximum frequency cap for all histograms

df = load_phase_transition_iterations()

# Filter data
filtered_df = df[
    (df['range'].notna()) &
    (df['iteration'] > 0)
]

model = smf.glm(
    formula="iteration ~ city_size * generation_typ",
    data=filtered_df,
    family=sm.families.NegativeBinomial()
).fit
ic(model.summary())

for tsp in tsp_types:
    a = filtered_df.query("generation_type == @tsp and city_size == 20")['iteration']
    b = filtered_df.query("generation_type == @tsp and city_size == 30")['iteration']
    u, p = mannwhitneyu(a, b, alternative='two-sided')
    ic(f"{tsp}: U={u:.1f}, p={p:.3g}")