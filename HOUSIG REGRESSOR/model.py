import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, HuberRegressor)
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline # data leakages
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Load dataset
data=pd.read_csv(r'C:\Users\HP\Desktop\ALL DATA SET\USA_Housing.csv')

# Preprocessing
X = data.drop(['Price', 'Address'], axis=1)
y=data['Price']

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Robust Regression": HuberRegressor(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Elastic Net": ElasticNet(),
    "Polynomial Regression": Pipeline([('poly', PolynomialFeatures(degree=2)),
                                       ('linear', LinearRegression())]),
    "SGDRegressor": SGDRegressor(),
    "ANN": MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000),
    "Random Forest": RandomForestRegressor(),
    "SVM": SVR(),
    "LGBM": lgb.LGBMRegressor(),
    "XGBoost": xgb.XGBRegressor(),
    "KNN": KNeighborsRegressor()
    
}

# train the models and evaluate

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append((name, mae, mse, r2))
    
    
    
    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "R2 Score": r2
    })
    
    with open(f"{name}.pkl", "wb") as f:
        pickle.dump(model, f)
        
# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)  
results_df.to_csv("model_evaluation_results.csv", index=False)
print("Model have been trained and save as pickle files and evaluation results saved to model_evaluation_results.csv")



