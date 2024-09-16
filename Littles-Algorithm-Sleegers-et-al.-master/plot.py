import numpy as np
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def read_file(csv_file):
    """
    Read the file and return the content as a dictionary.

    Parameters
    ----------
    csv_file: String
        path to the results file to read

    Returns
    -------
    DataFrame:
        content of the file as a pandas DataFrame
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError('File does not exist')
    return pd.read_csv(csv_file)

def column_rename(data):
    """
    Rename the column 'file' to 'std' in the DataFrame.

    Parameters
    ----------
    data: DataFrame
        DataFrame to create the new columns 'std' and 'id' from
    """
    # create standard deviation column
    data['std'] = data['file'].str.extract(r'std(\d{2})').astype(int)

    # create id number column
    data['id'] = data['file'].str.extract(r'_(\d+)$').astype(int)

# Following CRISP-DM standard
def describe_data(data):
    """
    Describe the dataframe 
    """

# def dot_plot(data, show_mean):
#     """
#     Create a dot plot where sd is on the x-axis and the number of iterations on the y-axis.

#     Parameters
#     ----------
#     data: DataFrame
#         DataFrame to plot
#     show_mean: Boolean
#         Display a line graph of mean 
#     """
#     sns.set_theme(style="whitegrid")
#     sns.stripplot(x='std', y='iterations', data=data, jitter=True)
#     plt.show()

if __name__ == '__main__':
    # Read the file
    data = read_file('results/partialresults48.csv')
    column_rename(data)

    print(data.head())