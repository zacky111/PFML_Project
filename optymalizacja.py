import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import math
import warnings
from fancyimpute import KNN
from sklearn.metrics import mean_squared_error
import os

import receptury
import modele


## ------------------------------- ładowanie plików z danymi
folder_name = "datasets"
X_train = pd.read_csv(os.path.join(folder_name, "X_train.csv"))
X_val = pd.read_csv(os.path.join(folder_name, "X_val.csv"))
X_test = pd.read_csv(os.path.join(folder_name, "X_test.csv"))

y_train = pd.read_csv(os.path.join(folder_name, "y_train.csv")).squeeze()  # Zamiana na Series
y_val = pd.read_csv(os.path.join(folder_name, "y_val.csv")).squeeze()
y_test = pd.read_csv(os.path.join(folder_name, "y_test.csv")).squeeze()

results=[]

#-----------------------------------------------------------------------------------
#################################### Main pętla ####################################

recepturyList=[receptury.recipe_spotify_col,
               receptury.recipe_tiktok_col,
               receptury.recipe_youtube_col,
               receptury.recipe_all]

for recipe in recepturyList:

    recipe_name = "+".join(recipe)  # Łączenie nazw kolumn w nazwę receptury

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

    results.append(["Linear Regression", recipe_name, rmse_train, rmse_val])


    ##-------------------- Wielomian

    print ("---------Wielomian")

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

        print(f"Polynomial Regression Training RMSE (degree={degree}):", rmse_train_poly)
        print(f"Polynomial Regression Validation RMSE (degree={degree}):", rmse_val_poly)
        results.append([f"Polynomial Regression (degree={degree})", recipe_name, rmse_train_poly, rmse_val_poly])

    ##-------------------- SVM

    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error
    import numpy as np


    print ("---------SVM")
    # Tworzenie i trenowanie modelu SVM
    svm = SVR(kernel='rbf', C=1.0, epsilon=0.1)  # Parametry można dostosować

    svm.fit(X_train[recipe], y_train)

    # Predykcja na zbiorach treningowym i walidacyjnym
    y_pred_train = svm.predict(X_train[recipe])
    y_pred_val = svm.predict(X_val[recipe])

    # Obliczenie RMSE dla treningu i walidacji
    rmse_train = round(np.sqrt(mean_squared_error(y_train, y_pred_train)),3)
    rmse_val = round(np.sqrt(mean_squared_error(y_val, y_pred_val)),3)

    print(f'SVM Training RMSE: {rmse_train:.4f}')
    print(f'SVM Validation RMSE: {rmse_val:.4f}')
    results.append(["SVM", recipe_name, rmse_train, rmse_val])


    ##-------------------- MLP

    print ("--------- MLP")
    warnings.filterwarnings("ignore", category=UserWarning)

    mlp = MLPRegressor(
        hidden_layer_sizes=(10,10), activation='logistic', solver='lbfgs',
        max_iter=2000, random_state=1, learning_rate_init=0.001,
        early_stopping=True, n_iter_no_change=10
    )

    mlp.fit(X_train[recipe], y_train)
    y_pred_train = mlp.predict(X_train[recipe])
    y_pred_val = mlp.predict(X_val[recipe])

    rmse_train = round(np.sqrt(mean_squared_error(y_train, y_pred_train)),3)
    rmse_val = round(np.sqrt(mean_squared_error(y_val, y_pred_val)),3)

    print(f'MLP Training RMSE: {rmse_train:.4f}')
    print(f'MLP Validation RMSE: {rmse_val:.4f}')
    results.append(["MLP", recipe_name, rmse_train, rmse_val])

# Tworzenie DataFrame z wynikami
results_df = pd.DataFrame(results, columns=["Model", "Recipe", "RMSE_Train", "RMSE_Validation"])

print(results_df)

# Eksport do pliku CSV
results_df.to_csv("results.csv", index=False)