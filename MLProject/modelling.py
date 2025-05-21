import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random
import numpy as np
import sys

if __name__ == "__main__":
    train_path = sys.argv[1]
    test_path = sys.argv[2]

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

X_train, y_train = train_data.drop("Revenue", axis=1), train_data["Revenue"]
X_test, y_test = test_data.drop("Revenue", axis=1), test_data["Revenue"]
input_example = X_train[0:5]


with mlflow.start_run():
    
    n_estimators = 200
    max_depth = 20
    mlflow.autolog()
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = model.score(X_test, y_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    n = len(y_test)                  
    p = X_test.shape[1]
    adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("adjusted_r2", adjusted_r2)
    mlflow.sklearn.log_model(
        sk_model = model,
        artifact_path="model",
        input_example=input_example
    )