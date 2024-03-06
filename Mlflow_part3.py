# Databricks notebook source
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split

# COMMAND ----------
folder_path = "/dbfs/mnt/archive/"
model_path = folder_path+"packaged_models/python_model.pkl"

# COMMAND ----------

df = pd.read_csv(folder_path+'breast_cancer.csv')
local_df = df.copy() 
local_df.iloc[:, 2:] = df.iloc[:, 2:] * 1.1

# COMMAND ----------

local_X = local_df.drop(df.columns[1], axis=1)
local_y = local_df.iloc[:,1]
local_X_train, local_X_test, local_y_train, local_y_test = train_test_split(local_X, local_y, test_size=0.2, random_state=42)

# COMMAND ----------

model_wrapper = joblib.load(model_path)
model = model_wrapper.model
print("Model type:", type(model))
print(f"Model structure: {model}")
print("Model's parameters:", model.get_params())

# If the model is a tree-based model we can get more details about its structure
if hasattr(model, 'estimators_'):
    print(f"Number of trees: {len(model.estimators_)}")
    print(f"Depth of the first tree: {model.estimators_[0].tree_.max_depth}")

if hasattr(model, 'feature_importances_'):
    print("Feature importances of model:", model.feature_importances_)

if hasattr(model, 'coef_'):
    print("For linear models, coefficients are:", model.coef_)

if hasattr(model, 'support_vectors_'):
    print("For SVM model, support vectors are:", model.support_vectors_)

# Training score
if hasattr(model, 'score'):
    print(f"Training score: {model.score(local_X_train, local_y_train)}")

# Class labels
if hasattr(model, 'classes_'):
    print(f"Class labels: {model.classes_}")

# Cross-validation results (if using a model from a cross-validation process)
if hasattr(model, 'cv_results_'):
    print(f"Cross-validation results: {model.cv_results_}")

# Decision function (if applicable)
if hasattr(model, 'decision_function'):
    decision_boundaries = model.decision_function(local_X_test)
    print(f"Decision function output: {decision_boundaries}")

# For predict_probabilities
if hasattr(model, 'predict_proba'):
    probabilities = model.predict_proba(local_X_test)
    print(f"Prediction probabilities: {probabilities}")

# COMMAND ----------
model.fit(local_X_train, local_y_train)
retrained_model_path = folder_path + "packaged_models/retrained_python_model.pkl"
joblib.dump(model, retrained_model_path)

# COMMAND ----------
predictions = model.predict(local_X_test)
print(f"Predictions: {predictions}")
accuracy = accuracy_score(local_y_test, predictions)
print ('Accuracy', accuracy)

# COMMAND ----------


