"""
Analysis script to fit Poisson GLMs examining the effect of city_size on iteration counts
for both Phase Transition and Hill Climb experiments.
"""
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
from icecream import ic
import numpy as np

# Import loading functions from your existing module
from util.load_experiment import (
    load_phase_transition_iterations,
    load_hill_climb_iterations
)

def fit_poisson_glm(df: pd.DataFrame, response: str, predictors: list):
    """
    Fit a Poisson GLM of the form response ~ predictor.
    Categorical variables are one-hot encoded (excluding the first level to avoid collinearity).
    """
    # Drop missing values
    cols = [response] + predictors
    df_clean = df[cols].dropna()

    # design matrices
    y = df_clean[response]
    X = sm.add_constant(df_clean[predictors])

    # Fit Poisson GLM
    poisson_model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
    results = poisson_model.fit()

    # Print summary and return results
    print(results.summary())#.as_latex())
    # calculate and print dispersion to check pseudo
    disp = results.pearson_chi2 / results.df_resid
    ic(f"Dispersion:", disp)
    return results

def fit_poisson_glm(df: pd.DataFrame, response: str, predictors: list, categorical: list = None):
    """
    Fit a Poisson GLM of the form response ~ predictors, with optional categorical predictors.
    Categorical variables are one-hot encoded (excluding the first level to avoid collinearity).
    """
    # Drop missing values
    cols = [response] + predictors
    df_clean = df[cols].dropna()

    # One-hot encode categorical predictors
    if categorical:
        df_clean = pd.get_dummies(df_clean, columns=categorical, drop_first=True)

    y = df_clean[response]
    # Remove the response from X columns
    X_cols = [col for col in df_clean.columns if col != response]
    X = sm.add_constant(df_clean[X_cols])

    # Fit Poisson GLM
    poisson_model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
    results = poisson_model.fit()

    ic(results.summary())#.as_latex())
    # calculate and print dispersion to check pseudo
    disp = results.pearson_chi2 / results.df_resid
    ic(f"Dispersion:", disp)
    return results

def main():
    covariates = ['city_size', 'range']
    categorical = ['tsp_type']
    # --- Phase Transition Experiments ---
    df_pt = load_phase_transition_iterations()
    #df_pt['log_iteration'] = np.log(df_pt['iteration'])
    # for tsp in df_pt['generation_type'].unique():
    #     df_pt = df_pt[df_pt['generation_type'] == tsp]
    print(f"\n>>> random-sampling")
    for dist in df_pt['distribution'].unique():
        df_sub = df_pt[df_pt['distribution'] == dist]
        print(f"\n>>> Fitting Negative‐Binomial for {dist}:")
        # fit_poisson_glm(
        #     df_sub,
        #     response='iteration',
        #     predictors=['city_size', 'range'],
        #     categorical=None
        # )
            # TODO make a separate test for city + range + generation_type
        print(f"\n>>> Fitting Negative‐Binomial for {dist}:")
        model = smf.glm(
            formula="iteration ~ C(generation_type) + city_size + range",
            data=df_sub,
            family=sm.families.NegativeBinomial()
        ).fit()        
        print(model.summary())#.as_latex())

        disp = model.pearson_chi2 / model.df_resid
        ic(f"Dispersion:", disp)
    
    print(f"Sample size: {len(df_pt)}")
    print(f"Fitting Poisson GLM: iteration ~ {covariates}")

    # --- Hill Climb Experiments ---  
    categorical = ['tsp_type'] #generation_type
    df_hc = load_hill_climb_iterations()
    for dist in df_hc['distribution'].unique():
        df_sub = df_hc[df_hc['distribution'] == dist]

        print(f"\n>>> Hill-climb experiment: Fitting NB GLM for {dist} distribution:")
        model = smf.glm(
            formula="iteration ~ C(mutation_type) + C(generation_type) + city_size + range",
            data=df_sub,
            family=sm.families.NegativeBinomial()
        ).fit()
        disp = model.pearson_chi2 / model.df_resid
        ic(f"Dispersion:", disp)

        print(model.summary())#.as_latex())

    print(f"Sample size: {len(df_hc)}")
    print(f"Fitting Poisson GLM: iteration ~ {covariates}")

if __name__ == "__main__":
    main()