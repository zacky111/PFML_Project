import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
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

results=[]

#-----------------------------------------------------------------------------------
#################################### Main loop ####################################
recepturyList=[receptury.recipe_spotify_col,
               receptury.recipe_tiktok_col,
               receptury.recipe_youtube_col,
               receptury.recipe_all]

for recipe in recepturyList:

    recipe_name = "+".join(recipe)  # Combine column names into a recipe name

    ##-------------------- Linear Regression

    print ("---------Linear Regression")
    model = LinearRegression()

    numerical_columns = X_train.select_dtypes(include=['number']).columns
    model.fit(X_train[recipe], y_train)

    y_train_pred = model.predict(X_train[recipe])
    y_val_pred = model.predict(X_val[recipe])

    rmse_train = round(np.sqrt(mean_squared_error(y_train, y_train_pred)),3)
    rmse_val = round(np.sqrt(mean_squared_error(y_val, y_val_pred)),3)

    print("Linear Regression Training RMSE:", rmse_train)
    print("Linear Regression Validation RMSE:", rmse_val)
    params=None
    results.append(["Linear Regression", recipe_name,  params, rmse_train, rmse_val])

    ##-------------------- Polynomial Regression

    print ("---------Polynomial Regression")

    for degree in range(1,5):
        
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train[recipe])

        X_val_poly = poly.transform(X_val[recipe])

        poly_model = LinearRegression()
        poly_model.fit(X_train_poly, y_train)

        y_train_pred_poly = poly_model.predict(X_train_poly)
        y_val_pred_poly = poly_model.predict(X_val_poly)

        rmse_train_poly = round(np.sqrt(mean_squared_error(y_train, y_train_pred_poly)),3)
        rmse_val_poly = round(np.sqrt(mean_squared_error(y_val, y_val_pred_poly)),3)

        params="degree: " + str(degree)
        print(f"Polynomial Regression Training RMSE:", rmse_train_poly)
        print(f"Polynomial Regression Validation RMSE:", rmse_val_poly)
        results.append([f"Polynomial Regression", recipe_name, params, rmse_train_poly, rmse_val_poly])

    ##-------------------- SVM

    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error
    import numpy as np

    print ("---------SVM")
    # Creating and training the SVM model

    # Conditional parameter filtering for RandomizedSearchCV
    def filter_params(kernel):
        if kernel == 'poly':
            return {'C': [0.1, 1, 10, 100],
                    'epsilon': [0.01, 0.1, 0.2, 0.5],
                    'degree': [2, 3, 4],
                    'gamma': ['scale', 'auto']}
        elif kernel in ['rbf', 'sigmoid']:
            return {'C': [0.1, 1, 10, 100],
                    'epsilon': [0.01, 0.1, 0.2, 0.5],
                    'gamma': ['scale', 'auto']}
        elif kernel == 'linear':
            return {'C': [0.1, 1, 10, 100],
                    'epsilon': [0.01, 0.1, 0.2, 0.5]}

    # Loop through different kernels
    for kernel in ['linear']:#, 'poly', 'rbf', 'sigmoid']:
        print(f"Optimizing SVM with kernel: {kernel}")
        
        # Creating the model and parameter space
        svm = SVR(kernel=kernel)
        param_grid = filter_params(kernel)

        # Using GridSearchCV
        grid_search_svm = GridSearchCV(
            estimator=svm,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            cv=3,  # Cross-validation
            verbose=2,
            n_jobs=-1  # Use multiple processors
        )
        
        # Fitting the model
        grid_search_svm.fit(X_train[recipe], y_train)
        
        # Best parameters
        best_params_svm = grid_search_svm.best_params_
        print(f"Best params for {kernel}: {best_params_svm}")
        
        # Prediction and RMSE calculation
        y_pred_train_svm = grid_search_svm.predict(X_train[recipe])
        y_pred_val_svm = grid_search_svm.predict(X_val[recipe])
        
        rmse_train_svm = round(np.sqrt(mean_squared_error(y_train, y_pred_train_svm)), 3)
        rmse_val_svm = round(np.sqrt(mean_squared_error(y_val, y_pred_val_svm)), 3)
        
        print(f"SVM Training RMSE (kernel={kernel}): {rmse_train_svm:.4f}")
        print(f"SVM Validation RMSE (kernel={kernel}): {rmse_val_svm:.4f}")
        
        # Adding results to the list
        results.append([f"SVM (kernel={kernel})", recipe_name, best_params_svm, rmse_train_svm, rmse_val_svm])

    ##-------------------- MLP

    print ("--------- MLP")

    param_distributions = {
    'hidden_layer_sizes': [(10,), (50,), (100,), (50, 50), (100, 50, 25)],
    'activation': ['relu', 'logistic', 'tanh'],
    'solver': ['lbfgs', 'adam', 'sgd'],
    'alpha': [1e-5, 1e-4, 1e-3, 1e-2],
    'learning_rate': ['constant', 'adaptive'],
    'learning_rate_init': [0.0001, 0.001, 0.01],
    'max_iter': [1000, 2000, 3000],
    'early_stopping': [True],
    'n_iter_no_change': [5, 10, 20]
    }

    mlp = MLPRegressor(random_state=1)

    random_search = RandomizedSearchCV(
    mlp,
    param_distributions=param_distributions,
    n_iter=5,  # Number of random trials
    scoring='neg_mean_squared_error',
    cv=3,
    random_state=42,
    verbose=2
    )

    random_search.fit(X_train[recipe], y_train)
    params=random_search.best_params_
    y_pred_train = random_search.predict(X_train[recipe])
    y_pred_val = random_search.predict(X_val[recipe])

    rmse_train = round(np.sqrt(mean_squared_error(y_train, y_pred_train)),3)
    rmse_val = round(np.sqrt(mean_squared_error(y_val, y_pred_val)),3)

    print(f'MLP Training RMSE: {rmse_train:.4f}')
    print(f'MLP Validation RMSE: {rmse_val:.4f}')
    results.append(["MLP", recipe_name, params,rmse_train, rmse_val])

# Creating DataFrame with results
results_df = pd.DataFrame(results, columns=["Model", "Recipe", "Params", "RMSE_Train", "RMSE_Validation"])

############################################ Finding the best model:
# Find rows with the smallest RMSE for training and validation
min_rmse_train = results_df.loc[results_df['RMSE_Train'].idxmin()]
min_rmse_val = results_df.loc[results_df['RMSE_Validation'].idxmin()]

# Display results
print("\nBest result on the training set:")
print(min_rmse_train)
print("\nBest result on the validation set:")
print(min_rmse_val)

## ----- Displaying results and saving them to CSV
print('#===============================================================================|')
print(results_df)
results_df.to_csv("results.csv", index=False)

# Saving these results to a CSV file
best_results_df = pd.DataFrame([min_rmse_train, min_rmse_val])
best_results_df.to_csv("best_results.csv", index=False)
print('#===============================================================================|')


## ------ Best models for each recipe
# Grouping by recipe and selecting the row with the smallest RMSE_Validation
best_models_per_recipe = results_df.loc[results_df.groupby("Recipe")["RMSE_Validation"].idxmin()]
"""
# Displaying results
print("\nBest models for each recipe:")
print(best_models_per_recipe)
"""
# Saving results to a CSV file
best_models_per_recipe.to_csv("best_models_per_recipe.csv", index=False)

print("done")