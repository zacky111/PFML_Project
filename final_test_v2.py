import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('best_resultsz.csv')  # Replace with your CSV file path

# Function to create and train models
def create_and_train_model(model_name, params_str):
    """
    Create and train the model based on the name and parameters string.
    
    Parameters:
    - model_name: The model type (e.g., 'LinearRegression')
    - params_str: The parameters of the model as a string.
    
    Returns:
    - model: The trained model
    """
    # Convert string parameters to actual parameter dictionary using eval
    params = eval(params_str) if params_str != '{}' else {}

    # Create the model based on the model name
    if model_name == 'LinearRegression':
        model = LinearRegression(**params)
    elif model_name == 'SVM':
        model = SVR(**params)
    elif model_name == 'MLPRegressor':
        model = MLPRegressor(**params)
    elif model_name == 'PolynomialRegression':
        degree = params.get('degree', 2)  # Default to degree 2 if not provided
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Generate random data to simulate training (Replace with real data)
    X_train = np.random.rand(100, 5)  # 100 samples, 5 features
    y_train = np.random.rand(100)  # 100 target values

    # Train the model
    model.fit(X_train, y_train)
    
    return model

# Iterate through each row in the CSV and create/train the corresponding model
trained_models = {}

for index, row in data.iterrows():
    model_name = row['model']
    params_str = row['parameters']
    rmse = row['rmse']
    
    # Create and train the model
    model = create_and_train_model(model_name, params_str)
    
    # Store the trained model
    trained_models[model_name] = model
    
    # Print details about the trained model
    print(f"Model: {model_name}")
    print(f"Parameters: {params_str}")
    print(f"RMSE: {rmse}")
    print(f"Trained Model: {model}")
    print("-" * 50)

# Optionally, save trained models to disk (e.g., using joblib)
import joblib
for model_name, model in trained_models.items():
    joblib.dump(model, f"{model_name}_trained_model.pkl")


