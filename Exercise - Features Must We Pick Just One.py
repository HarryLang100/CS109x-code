"""
Created on Sat Nov 18 16:10:19 2023

"""

#%% Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper import fit_and_plot_linear, fit_and_plot_multi

#%% Read the file "Advertising.csv"
df = pd.read_csv("Data/Advertising.csv")

#%% Define an empty Pandas dataframe to store the R-squared value associated 
# with each # predictor for both the train and test split
df_results = pd.DataFrame(columns=['Predictor', 'R2 Train', 'R2 Test'])

#%%
# For each predictor in the dataframe, call the function "fit_and_plot_linear()"
# from the helper file with the predictor as a parameter to the function

# This function will split the data into train and test split, fit a linear model
# on the train data and compute the R-squared value on both the train and test data

predictors = df.iloc[:, :3].columns
df_results = pd.DataFrame()
for current_predictor in predictors:
    r2_train, r2_test = fit_and_plot_linear(df[[current_predictor]])    
    new_row = {'Predictor': [current_predictor], 
               'R2 Train': [r2_train],
               'R2 Test': [r2_test]}  
    df_results = pd.concat([df_results, pd.DataFrame(new_row)], ignore_index=True)


#%%
# Call the function "fit_and_plot_multi()" from the helper to fit a multilinear model
# on the train data and compute the R-squared value on both the train and test data
r2_train, r2_test = fit_and_plot_multi()