from mlflow.tracking import MlflowClient
import mlflow

# mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

model_version = client.get_model_version_by_alias(
    "HeartDiseaseCatBoost-V1",
    "production"
)

print(model_version.version)