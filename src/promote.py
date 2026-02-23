import subprocess
from mlflow.tracking import MlflowClient

MODEL_NAME = "HeartDiseaseCatBoost-V1"


def promote_if_better(new_auc, run_id):
    client = MlflowClient()

    # Check if model exists
    try:
        client.get_registered_model(MODEL_NAME)
    except:
        client.create_registered_model(MODEL_NAME)

    # Check if production alias exists
    try:
        prod_version = client.get_model_version_by_alias(
            MODEL_NAME, "production"
        )

        prod_run = client.get_run(prod_version.run_id)
        prod_auc = float(prod_run.data.metrics["auc"])

        print(f"Current Production AUC: {prod_auc}")
        print(f"New Model AUC: {new_auc}")

        if new_auc > prod_auc:
            print("New model is better. Promoting...")

            git_push()

            # Register new version
            mv = client.create_model_version(
                name=MODEL_NAME,
                source=f"runs:/{run_id}/catboost_model",
                run_id=run_id
            )

            # Set alias production to new version
            client.set_registered_model_alias(
                MODEL_NAME, "production", mv.version
            )

            print("Promotion completed.")

        else:
            print("New model is NOT better. Keeping current production.")

    except Exception:
        print(Exception)
        # No production model exists yet
        print("No production model found. Registering first model...")

        # mv = client.create_model_version(
        #     name=MODEL_NAME,
        #     source=f"runs:/{run_id}/catboost_model",
        #     run_id=run_id
        # )

        # client.set_registered_model_alias(
        #     MODEL_NAME, "production", mv.version
        # )

        # print("First model registered as production.")

def git_push():
    try:
        # Add all changes
        subprocess.run(["git", "add", "."], check=True)

        # Commit
        subprocess.run(
            ["git", "commit", "-m", "Auto promote model: New AUC is better"],
            check=True
        )

        # Push
        subprocess.run(["git", "push", "origin", "main"], check=True)

        print("Changes committed and pushed successfully 🚀")

    except subprocess.CalledProcessError as e:
        print("Git operation failed:", e)