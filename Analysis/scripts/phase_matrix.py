from util.load_experiment import load_all_matrices

df_all = load_all_matrices()

# print
print(df_all.shape)
print(df_all.head())
print(df_all.columns)
print(df_all.dtypes)