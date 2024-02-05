# Databricks notebook source
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# COMMAND ----------
folder_path = "/dbfs/mnt/archive/blog" #please add your folder path 
df = pd.read_csv(folder_path+'breast_cancer.csv')
# COMMAND ----------
# MAGIC %pip install mlflow
# COMMAND ----------
# COMMAND ----------
X = df.drop(df.columns[1], axis=1)
y = df.iloc[:,1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# COMMAND ----------
print (X,y)
# COMMAND ----------
# Set the experiment name
experiment_name = "/Users/user_name/name_of_experiment"
# Check if the experiment already exists
if not mlflow.get_experiment_by_name(experiment_name):
    # If not, create it
    mlflow.create_experiment(experiment_name)
# Set the experiment to track
mlflow.set_experiment(experiment_name)
# COMMAND ----------
# List of hyperparameters to try
n_estimators_options = [50, 100, 150]
max_depth_options = [2, 4, 6]
for n_estimators in n_estimators_options:
    for max_depth in max_depth_options:
        with mlflow.start_run():
            # Create and train the model
            rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            rf.fit(X_train, y_train)
            
            # Make predictions and calculate accuracy
            predictions = rf.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            
            # Log parameters, metrics, and the model
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(rf, "model")
            
            print(f"Model with n_estimators={n_estimators}, max_depth={max_depth} has accuracy: {accuracy}")