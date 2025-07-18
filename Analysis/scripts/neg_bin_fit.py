"""
Analysis script to fit Negative Binomial GLMs examining the effect of city_size, range,
and categorical variables on iteration counts for both Phase Transition and Hill Climb experiments.
"""

import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
from icecream import ic

# Import custom loading functions
from util.load_experiment import (
    load_phase_transition_iterations,
    load_hill_climb_iterations
)

def print_dispersion(results):
    """Calculate and print the dispersion statistic."""
    disp = results.pearson_chi2 / results.df_resid
    ic(f"Dispersion: {disp:.3f}")
    return disp

def fit_nb_glm_formula(df, formula, family=sm.families.NegativeBinomial()):
    """
    Fits a GLM with the specified formula and family.
    Prints LaTeX summary and returns the fitted model.
    """
    model = smf.glm(formula=formula, data=df, family=family)
    results = model.fit()
    print(results.summary().as_latex())
    print_dispersion(results)
    return results

def analyze_phase_transition():
    df_pt = load_phase_transition_iterations()
    if df_pt.empty:
        print("No data for phase transition experiments.")
        return

    print("\n>>> Phase Transition (Random Sampling) Experiments")
    for dist in df_pt['distribution'].unique():
        df_sub = df_pt[df_pt['distribution'] == dist]
        print(f"\n>>> Fitting Negative Binomial for {dist}:")
        fit_nb_glm_formula(
            df_sub,
            formula="iteration ~ C(generation_type) + city_size + range"
        )
    print(f"Sample size: {len(df_pt)}")

def analyze_hill_climb():
    df_hc = load_hill_climb_iterations()
    if df_hc.empty:
        print("No data for hill climb experiments.")
        return

    print("\n>>> Hill Climb Experiments")
    for dist in df_hc['distribution'].unique():
        df_sub = df_hc[df_hc['distribution'] == dist]
        print(f"\n>>> Fitting NB GLM for {dist} distribution:")
        fit_nb_glm_formula(
            df_sub,
            formula="iteration ~ C(mutation_type) + C(generation_type) + city_size + range"
        )
    print(f"Sample size: {len(df_hc)}")

def main():
    analyze_phase_transition()
    analyze_hill_climb()

if __name__ == "__main__":
    main()
