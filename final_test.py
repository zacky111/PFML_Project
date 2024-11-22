import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import PredictionErrorDisplay

import math
import warnings
from fancyimpute import KNN
from sklearn.metrics import mean_squared_error
import os

import receptury

## ------------------------------- Loading data files
folder_name = "datasets"
X_train = pd.read_csv(os.path.join(folder_name, "X_train.csv"))
X_val = pd.read_csv(os.path.join(folder_name, "X_val.csv"))
X_test = pd.read_csv(os.path.join(folder_name, "X_test.csv"))

y_train = pd.read_csv(os.path.join(folder_name, "y_train.csv")).squeeze()  # Convert to Series
y_val = pd.read_csv(os.path.join(folder_name, "y_val.csv")).squeeze()
y_test = pd.read_csv(os.path.join(folder_name, "y_test.csv")).squeeze()


## import best models
top_4= pd.read_csv(os.path.join("best_results.csv"))

model_dict = {
    'MLP': "MLPRegressor",
    'Linear Regression': "LinearRegression",
    'Polynomial Regression': "PolynomialFeatures",
    'SVM': "SVR"
}

"""
for row in top_4:
    print(f"MODEL: {row} ---------------------------------------------")
    type=row[1]
    print(row)
"""
for index, row in top_4.iterrows():
    print(f"Index: {index}")
    print(f"RMSE_Validation: {row['Model']}")
    print(f"RMSE_Validation: {row['RMSE_Validation']}")

    params=row["Params"]
    model_name=row["Model"]
    recipe=row["Recipe"]

    recipe_list=[]    
    recipe_list=recipe.split("+")

    model = None
    cmd=f"model = {model_dict[model_name]}({params})"
    print(cmd)
    exec(cmd)

    model.fit(X_train[recipe_list],y_train)
    








