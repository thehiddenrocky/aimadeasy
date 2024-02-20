# Databricks notebook source
import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
import shutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlflow.pyfunc import PythonModel

# COMMAND ----------

folder_path = "/dbfs/mnt/archive/recipe-optimization/Mercerizator/latest/working/learn/"
df = pd.read_csv(folder_path+'breast_cancer.csv')

# COMMAND ----------

df

# COMMAND ----------

# MAGIC %pip install mlflow

# COMMAND ----------

X = df.drop(df.columns[1], axis=1)
y = df.iloc[:,1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# COMMAND ----------

print (X,y)

# COMMAND ----------

# Set the experiment name
experiment_name = "/Users/akshenndra.garg@ahlstrom-munksjo.com/aks_breast_cancer_mlflow_learn"

# Check if the experiment already exists
if not mlflow.get_experiment_by_name(experiment_name):
    # If not, create it
    mlflow.create_experiment(experiment_name)

# Set the experiment to track
mlflow.set_experiment(experiment_name)

# COMMAND ----------

# MAGIC %fs ls

# COMMAND ----------

dbutils.fs.mkdirs("/FileStore/mlflow_envs/")

# COMMAND ----------

yaml_file_path = "/dbfs/FileStore/mlflow_envs/learn_mlflow_env_cancer.yaml"

# COMMAND ----------

print(f"mlflow version: {mlflow.__version__}")
print(f"pandas version: {pd.__version__}")
print(f"scikit-learn version: {sklearn.__version__}")

# COMMAND ----------

yaml_content = """
name: mlflow-env
channels:
  - defaults
dependencies:
  - python = 3.8
  - mlflow = 2.3.0
  - pandas = 1.3.5
  - scikit-learn = 1.1.0
"""

with open(yaml_file_path, "w") as file:
    file.write(yaml_content)

# COMMAND ----------

# MAGIC %fs ls /FileStore/mlflow_envs/

# COMMAND ----------

best_accuracy = 0
best_model = None

n_estimators_options = [50, 100, 150]
max_depth_options = [2, 4, 6]

for n_estimators in n_estimators_options:
    for max_depth in max_depth_options:
        with mlflow.start_run():
            rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            rf.fit(X_train, y_train)
            
            predictions = rf.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(rf, "model")
            
            print(f"Model with n_estimators={n_estimators}, max_depth={max_depth} has accuracy: {accuracy}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = rf

if best_model is not None:
    class BestModelWrapper(PythonModel):
        def __init__(self, model):
            self.model = model

        def predict(self, context, model_input):
            return self.model.predict(model_input)

    best_model_wrapper = BestModelWrapper(best_model)

    conda_env_path = yaml_file_path

    packaging_dir = "/dbfs/mnt/archive/recipe-optimization/Mercerizator/latest/working/learn/packaged_models"

    shutil.rmtree(packaging_dir, ignore_errors=True)

    mlflow.pyfunc.save_model(
        path=packaging_dir,
        python_model=best_model_wrapper,
        conda_env=conda_env_path
    )

# COMMAND ----------

model_path = "/dbfs/mnt/archive/recipe-optimization/Mercerizator/latest/working/learn/packaged_models/MLmodel"
with open(model_path, 'r') as file:
    mlmodel_content = file.read()

print(mlmodel_content)


# COMMAND ----------


