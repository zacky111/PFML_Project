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
from sklearn.pipeline import make_pipeline
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


## --------------------------------- nan in "X_test" rows

print("X_test: ")
print(X_test)
print(X_test.isnull().sum())
#mean value of each column in training set
median=X_test.select_dtypes(include=['number']).median() #### <----------------- tu ew. miejsce na zmianę czy ma to być mean, czy mediana

#X_test_filled = X_test.copy()


for column in X_test.select_dtypes(include=['number']).columns:
    if column in median:
        X_test[column] = X_test[column].fillna(median[column])

print(X_test)



## import best models
#top_4= pd.read_csv(os.path.join("best_results.csv"))
#print(top_4)

##reading through them and rewritting into dictionaries

##Model1
Model1_parametry = {
    'solver': 'sgd',
    'n_iter_no_change': 20,
    'max_iter': 2000,
    'learning_rate_init': 0.001,
    'learning_rate': 'constant',
    'hidden_layer_sizes': (50, 50),
    'early_stopping': True,
    'alpha': 0.001,
    'activation': 'logistic'
}
Model1 =MLPRegressor(**Model1_parametry)
Model1_receptura=receptury.recipe_all

##Model2
Model2_parametry = {
    'degree': 4,
    'include_bias': False  # Wartość False, jeśli nie chcesz dodawać biasu
}

Model2_receptura = receptury.recipe_spotify_col
Model2_poly = PolynomialFeatures(degree=Model2_parametry['degree'])

X_train_poly_model2 = Model2_poly.fit_transform(X_train[Model2_receptura])
X_test_poly_model2 = Model2_poly.transform(X_test[Model2_receptura])

Model2=LinearRegression()

##Model3
Model3_parametry = {
    'degree': 3,
    'include_bias': False
}

Model3_receptura = receptury.recipe_spotify_col
Model3 = make_pipeline(PolynomialFeatures(degree=Model3_parametry['degree']), LinearRegression())


#X_train_poly_model3 = Model3_poly.fit_transform(X_train[Model3_receptura])
#X_test_poly_model3 = Model3_poly.transform(X_test[Model3_receptura])


##Model4
Model4_parametry = {
    'solver': 'sgd',
    'n_iter_no_change': 10,
    'max_iter': 3000,
    'learning_rate_init': 0.001,
    'learning_rate': 'constant',
    'hidden_layer_sizes': (10,),  
    'early_stopping': True,
    'alpha': 0.0001,
    'activation': 'relu'
}

Model4_receptura = receptury.recipe_spotify_col
Model4 = MLPRegressor(**Model4_parametry)


######################################################################### training models

Model1.fit(X_train[Model1_receptura],y_train)
Model2.fit(X_train_poly_model2,y_train)
Model3.fit(X_train[Model3_receptura],y_train)
Model4.fit(X_train[Model4_receptura],y_train)

################# testing
print("Model1:")
y_train_pred_model1 = Model1.predict(X_train[Model1_receptura])
y_test_pred_model1 = Model1.predict(X_test[Model1_receptura])

rmse_train_model1= round(np.sqrt(mean_squared_error(y_train, y_train_pred_model1)),3)
rmse_val_model1 = round(np.sqrt(mean_squared_error(y_test, y_test_pred_model1)),3)

print(f"Polynomial Regression Training RMSE:", rmse_train_model1)
print(f"Polynomial Regression Validation RMSE:", rmse_val_model1)


print("Model2:")
y_train_pred_model2 = Model2.predict(X_train_poly_model2)
y_test_pred_model2 = Model2.predict(X_test_poly_model2)

rmse_train_model2= round(np.sqrt(mean_squared_error(y_train, y_train_pred_model2)),3)
rmse_val_model2 = round(np.sqrt(mean_squared_error(y_test, y_test_pred_model2)),3)

print(f"Polynomial Regression Training RMSE:", rmse_train_model2)
print(f"Polynomial Regression Validation RMSE:", rmse_val_model2)


print("Model3:")
y_train_pred_model3 = Model3.predict(X_train[Model3_receptura])
y_test_pred_model3 = Model3.predict(X_test[Model3_receptura])


rmse_train_model3= round(np.sqrt(mean_squared_error(y_train, y_train_pred_model3)),3)
rmse_val_model3 = round(np.sqrt(mean_squared_error(y_test, y_test_pred_model3)),3)

print(f"Polynomial Regression Training RMSE:", rmse_train_model3)
print(f"Polynomial Regression Validation RMSE:", rmse_val_model3)


print("Model4:")
y_train_pred_model4 = Model4.predict(X_train[Model4_receptura])
y_test_pred_model4 = Model4.predict(X_test[Model4_receptura])

rmse_train_model4= round(np.sqrt(mean_squared_error(y_train, y_train_pred_model1)),3)
rmse_val_model4 = round(np.sqrt(mean_squared_error(y_test, y_test_pred_model1)),3)

print(f"Polynomial Regression Training RMSE:", rmse_train_model1)
print(f"Polynomial Regression Validation RMSE:", rmse_val_model1)

## wizualizacja histogramów track score predyktowanego
# Importowanie bibliotek
import matplotlib.pyplot as plt
import seaborn as sns

# Lista modeli i ich predykcji
model_names = ['Model1: MLP', 'Model2: Polynomial Model', 'Model3: Polynomial Model', 'Model4: MLP']
predictions = [y_test_pred_model1, y_test_pred_model2, y_test_pred_model3, y_test_pred_model4]

# Iterowanie przez modele i wyświetlanie histogramów
for model_name, y_pred in zip(model_names, predictions):
    plt.figure(figsize=(12, 6))

    # Histogram dla rzeczywistych wartości
    sns.histplot(y_test, bins=40, kde=True, color='blue', label='Real', alpha=0.5)
    sns.histplot(y_pred, bins=40, kde=True, color='green', label='Predicted', alpha=0.5)
    plt.title(f'Comparison of actual vs. predicted values for track scores - {model_name}')
    plt.xlim(0, 300)
    plt.xlabel('Track Score')
    plt.ylabel('Frequency')
    plt.legend(title="Legenda")

    # Wyświetlenie wykresów
    plt.tight_layout()
    plt.show()



"""

# Importowanie bibliotek
import matplotlib.pyplot as plt
import seaborn as sns

# Lista modeli i ich predykcji
model_names = ['Model1: MLP', 'Model2: Polynomial Model', 'Model3: Polynomial Model', 'Model4: MLP']
predictions = [y_test_pred_model1, y_test_pred_model2, y_test_pred_model3, y_test_pred_model4]

# Tworzenie wspólnej figury
plt.figure(figsize=(16, 12))

# Iterowanie przez modele i tworzenie podwykresów
for i, (model_name, y_pred) in enumerate(zip(model_names, predictions), 1):
    plt.subplot(2, 2, i)
    sns.histplot(y_test, bins=40, kde=True, color='blue', label='Rzeczywiste', alpha=0.5)
    sns.histplot(y_pred, bins=40, kde=True, color='green', label='Przewidywane', alpha=0.5)
    plt.title(f'Rzeczywiste vs Przewidywane - {model_name}')
    plt.xlabel('Track Score')
    plt.ylabel('Frequency')
    plt.legend()

# Dostosowanie odstępów między podwykresami
plt.tight_layout()  
# Wyświetlenie wykresu
plt.show()

"""
