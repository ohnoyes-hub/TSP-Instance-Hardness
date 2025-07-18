from util.load_experiment import load_hill_climb_iterations
import statsmodels.formula.api as smf
import statsmodels.api as sm

df = load_hill_climb_iterations().rename(columns={'range': 'range_val'})

for dist in ['lognormal', 'uniform']:
    sub = df[df.distribution == dist]
    model = smf.ols(
        'iteration ~ C(city_size) * C(mutation_type) * C(generation_type) + range_val',
        data=sub
    ).fit()
    print(f"\n=== OLS Results for {dist} ===")
    print(model.summary())
    print("\nANOVA Table (Type III)")
    print(sm.stats.anova_lm(model, typ=3))



"""
ic| "Loaded": 'Loaded'
    len(all_entries): 1148987
    "hill climbed Lital iterations": 'hill climbed Lital iterations'

=== OLS Results for lognormal ===
                            OLS Regression Results                            
==============================================================================
Dep. Variable:              iteration   R-squared:                       0.060
Model:                            OLS   Adj. R-squared:                  0.060
Method:                 Least Squares   F-statistic:                     3385.
Date:                Mon, 19 May 2025   Prob (F-statistic):               0.00
Time:                        15:26:28   Log-Likelihood:            -7.9749e+06
No. Observations:              634921   AIC:                         1.595e+07
Df Residuals:                  634908   BIC:                         1.595e+07
Df Model:                          12                                         
Covariance Type:            nonrobust                                         
===================================================================================================================================================
                                                                                      coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------------------------------------------------------------------------
Intercept                                                                        1.373e+04    483.999     28.362      0.000    1.28e+04    1.47e+04
C(city_size)[T.30]                                                               1.912e+04    477.228     40.059      0.000    1.82e+04    2.01e+04
C(mutation_type)[T.scramble]                                                    -1966.3539    625.782     -3.142      0.002   -3192.866    -739.841
C(mutation_type)[T.swap]                                                           68.1785    636.503      0.107      0.915   -1179.346    1315.703
C(generation_type)[T.euclidean]                                                   5.91e+04    614.440     96.179      0.000    5.79e+04    6.03e+04
C(city_size)[T.30]:C(mutation_type)[T.scramble]                                 -1.862e+04    779.960    -23.874      0.000   -2.01e+04   -1.71e+04
C(city_size)[T.30]:C(mutation_type)[T.swap]                                     -2563.4762    671.539     -3.817      0.000   -3879.671   -1247.281
C(city_size)[T.30]:C(generation_type)[T.euclidean]                               -207.7681   1105.838     -0.188      0.851   -2375.176    1959.639
C(mutation_type)[T.scramble]:C(generation_type)[T.euclidean]                     -5.78e+04    914.708    -63.186      0.000   -5.96e+04    -5.6e+04
C(mutation_type)[T.swap]:C(generation_type)[T.euclidean]                         -1.62e+04    911.739    -17.766      0.000    -1.8e+04   -1.44e+04
C(city_size)[T.30]:C(mutation_type)[T.scramble]:C(generation_type)[T.euclidean]  2063.1341   1698.954      1.214      0.225   -1266.762    5393.030
C(city_size)[T.30]:C(mutation_type)[T.swap]:C(generation_type)[T.euclidean]      5.703e+04   1790.484     31.854      0.000    5.35e+04    6.05e+04
range_val                                                                       -4490.4309     66.809    -67.213      0.000   -4621.374   -4359.488
==============================================================================
Omnibus:                  2001760.708   Durbin-Watson:                   0.662
Prob(Omnibus):                  0.000   Jarque-Bera (JB):     831266102479.994
Skew:                          49.880   Prob(JB):                         0.00
Kurtosis:                    5607.631   Cond. No.                         77.4
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

ANOVA Table (Type III)
                                                        sum_sq  ...         PR(>F)
Intercept                                         3.826913e+12  ...  7.644088e-177
C(city_size)                                      7.634455e+12  ...   0.000000e+00
C(mutation_type)                                  6.532110e+10  ...   1.043534e-03
C(generation_type)                                4.400808e+13  ...   0.000000e+00
C(city_size):C(mutation_type)                     2.956153e+12  ...  1.364291e-135
C(city_size):C(generation_type)                   1.679359e+08  ...   8.509686e-01
C(mutation_type):C(generation_type)               1.967963e+13  ...   0.000000e+00
C(city_size):C(mutation_type):C(generation_type)  5.599896e+12  ...  4.307148e-256
range_val                                         2.149222e+13  ...   0.000000e+00
Residual                                          3.020509e+15  ...            NaN

[10 rows x 4 columns]

=== OLS Results for uniform ===
                            OLS Regression Results                            
==============================================================================
Dep. Variable:              iteration   R-squared:                       0.095
Model:                            OLS   Adj. R-squared:                  0.095
Method:                 Least Squares   F-statistic:                     4481.
Date:                Mon, 19 May 2025   Prob (F-statistic):               0.00
Time:                        15:26:32   Log-Likelihood:            -6.6646e+06
No. Observations:              514066   AIC:                         1.333e+07
Df Residuals:                  514053   BIC:                         1.333e+07
Df Model:                          12                                         
Covariance Type:            nonrobust                                         
===================================================================================================================================================
                                                                                      coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------------------------------------------------------------------------
Intercept                                                                          69.5938    812.146      0.086      0.932   -1522.187    1661.375
C(city_size)[T.30]                                                               7780.5346   1048.130      7.423      0.000    5726.233    9834.836
C(mutation_type)[T.scramble]                                                     -969.2745   1048.130     -0.925      0.355   -3023.576    1085.027
C(mutation_type)[T.swap]                                                          299.3691   1070.942      0.280      0.780   -1799.643    2398.382
C(generation_type)[T.euclidean]                                                  4.586e+04    903.586     50.755      0.000    4.41e+04    4.76e+04
C(city_size)[T.30]:C(mutation_type)[T.scramble]                                 -7212.6292   1474.354     -4.892      0.000   -1.01e+04   -4322.941
C(city_size)[T.30]:C(mutation_type)[T.swap]                                       1.81e+04   1364.471     13.267      0.000    1.54e+04    2.08e+04
C(city_size)[T.30]:C(generation_type)[T.euclidean]                               1.127e+04   1206.971      9.339      0.000    8905.982    1.36e+04
C(mutation_type)[T.scramble]:C(generation_type)[T.euclidean]                    -4.514e+04   1428.695    -31.597      0.000   -4.79e+04   -4.23e+04
C(mutation_type)[T.swap]:C(generation_type)[T.euclidean]                        -2.201e+04   1218.872    -18.057      0.000   -2.44e+04   -1.96e+04
C(city_size)[T.30]:C(mutation_type)[T.scramble]:C(generation_type)[T.euclidean]  9335.8591   1948.217      4.792      0.000    5517.416    1.32e+04
C(city_size)[T.30]:C(mutation_type)[T.swap]:C(generation_type)[T.euclidean]      4.676e+04   1574.695     29.692      0.000    4.37e+04    4.98e+04
range_val                                                                          19.2055      6.094      3.152      0.002       7.262      31.149
==============================================================================
Omnibus:                  2001630.680   Durbin-Watson:                   1.065
Prob(Omnibus):                  0.000   Jarque-Bera (JB):   13309548310501.916
Skew:                          93.135   Prob(JB):                         0.00
Kurtosis:                   24929.753   Cond. No.                     1.31e+03
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.31e+03. This might indicate that there are
strong multicollinearity or other numerical problems.

ANOVA Table (Type III)
                                                        sum_sq  ...         PR(>F)
Intercept                                         7.837698e+07  ...   9.317119e-01
C(city_size)                                      5.881723e+11  ...   1.144510e-13
C(mutation_type)                                  1.705744e+10  ...   4.497613e-01
C(generation_type)                                2.749613e+13  ...   0.000000e+00
C(city_size):C(mutation_type)                     4.123267e+12  ...   1.403852e-84
C(city_size):C(generation_type)                   9.308806e+11  ...   9.783987e-21
C(mutation_type):C(generation_type)               1.082310e+13  ...  1.072953e-220
C(city_size):C(mutation_type):C(generation_type)  1.050386e+13  ...  3.256376e-214
range_val                                         1.060301e+11  ...   1.622846e-03
Residual                                          5.486857e+15  ...            NaN

[10 rows x 4 columns]
"""