import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random
import numpy as np
import sys

if __name__ == "__main__":
    train_path = sys.argv[1]
    test_path = sys.argv[2]

TRAIN_PATH = "supplement_sales_preprocessed/sales_train_preprocessed.csv"
train_data = pd.read_csv(TRAIN_PATH)
TEST_PATH = "supplement_sales_preprocessed/sales_test_preprocessed.csv"
test_data = pd.read_csv(TEST_PATH)

X_train, X_test, y_train, y_test = train_test_split(
    train_data.drop("Revenue", axis=1), train_data["Revenue"], test_size=0.2, random_state=42
)
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
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(
        sk_model = model,
        artifact_path="model",
        input_example=input_example
    )