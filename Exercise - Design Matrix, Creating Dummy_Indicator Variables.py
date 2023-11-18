"""
Created on Sat Nov 18 18:55:37 2023

"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the credit data.
df = pd.read_csv('Data/credit.csv')
df.head()

# The response variable will be 'Balance.'
x = df.drop('Balance', axis=1)
y = df['Balance']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Trying to fit on all features in their current representation throws an error.
try:
    test_model = LinearRegression().fit(x_train, y_train)
except Exception as e:
    print('Error!:', e)
    
### edTest(test_model1) ###
# Fit a linear model using only the numeric features in the dataframe.
numeric_features = ["Income", "Limit", "Rating", "Cards", "Age", "Education"]
model1 = LinearRegression().fit(x_train[numeric_features], y_train)

# Report train and test R2 scores.
train_score = model1.score(x_train[numeric_features], y_train)
test_score = model1.score(x_test[numeric_features], y_test)
print('Train R2:', train_score)
print('Test R2:', test_score)

# Look at unique values of Ethnicity feature.
print('In the train data, Ethnicity takes on the values:', list(x_train['Ethnicity'].unique()))

### edTest(test_chow2) ###
# Submit an answer choice as a string below.
answer2 = '2' # I think this is right? As you can have 00 for Asian, 01 for Caucasian and 10 for AA?

### edTest(test_design) ###
# Create x train and test design matrices creating dummy variables for the categorical.
# hint: use pd.get_dummies() with the drop_first hyperparameter for this

# x_train_design = pd.concat([x_train[numeric_features],
#                             pd.get_dummies(x_train["Gender"], drop_first=True)
#                             pd.get_dummies(x_train["Student"], drop_first=True)
#                             pd.get_dummies(x_train["Married"], drop_first=True)
#                             pd.get_dummies(x_train["Ethnicity"], drop_first=True)],
#                             axis=1)

# x_test_design = pd.concat([x_test[numeric_features],
#                             pd.get_dummies(x_test["Gender"], drop_first=True)
#                             pd.get_dummies(x_test["Student"], drop_first=True)
#                             pd.get_dummies(x_test["Married"], drop_first=True)
#                             pd.get_dummies(x_test["Ethnicity"], drop_first=True)],
#                             axis=1)

x_train_design = pd.concat([x_train[numeric_features],
                            pd.get_dummies(x_train[["Sex", "Student", "Married", "Ethnicity"]], drop_first=True)],
                            axis=1)
                            

x_train_design.head()