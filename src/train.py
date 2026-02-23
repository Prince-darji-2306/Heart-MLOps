import os
import joblib
import mlflow
import mlflow.catboost
import pandas as pd
from promote import promote_if_better
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Paths
DATA_PATH = "data/train.csv"
MODEL_PATH = "models/catboost_model.pkl"

# Load dataset
df = pd.read_csv(DATA_PATH)
X = df.drop("Heart Disease", axis=1)
y = df["Heart Disease"]

# Split dataset
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pools for CatBoost
train_pool = Pool(X_train, y_train)
valid_pool = Pool(X_valid, y_valid)

# Hyperparameters
params = {
    "iterations": 2000,
    "learning_rate": 0.05,
    "depth": 2,
    "l2_leaf_reg": 15,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "random_seed": 42,
    "early_stopping_rounds": 300,
    "verbose": 200
}

# Start MLflow experiment
mlflow.set_experiment("heart_disease_catboost")

with mlflow.start_run():
    # Log hyperparameters
    mlflow.log_params(params)

    # Train model
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    # Predictions & metrics
    y_pred = model.predict(X_valid)
    y_pred_proba = model.predict_proba(X_valid)[:, 1]
    acc = accuracy_score(y_valid, y_pred)
    auc = roc_auc_score(y_valid, y_pred_proba)

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("auc", auc)

    # Save model
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    
    mlflow.catboost.log_model(model, "catboost_model")
    run_id = mlflow.active_run().info.run_id
    promote_if_better(auc, run_id)

print(f"Training finished. Accuracy: {acc:.4f}, AUC: {auc:.4f}")
