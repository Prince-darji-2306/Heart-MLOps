# ❤️ Heart Disease Prediction - MLOps Pipeline

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://heart-mlops.streamlit.app/)

This repository demonstrates a complete end-to-end MLOps pipeline for predicting heart disease. It integrates robust data versioning, experiment tracking, automated model promotion, and a user-friendly Streamlit web application.

## 🚀 Live Demo
Access the application here: [Predict Heart Disease](https://heart-mlops.streamlit.app/)

## 🛠️ Tech Stack
-   **Model**: [CatBoost](https://catboost.ai/) (Gradient Boosting on Decision Trees)
-   **Experiment Tracking**: [MLflow](https://mlflow.org/)
-   **Data Versioning**: [DVC](https://dvc.org/)
-   **Frontend**: [Streamlit](https://streamlit.io/)
-   **Environment**: Python 3.10

## 🏗️ Project Structure
```text
Heart-MLOps/
├── app/
│   └── main.py          # Streamlit web application
├── data/
│   └── train.csv        # Dataset (Versioned via DVC)
├── models/
│   └── catboost_model.pkl # Trained model instance
├── src/
│   ├── train.py         # Model training script with MLflow logging
│   └── promote.py       # Automated model promotion logic
├── dvc.yaml             # DVC pipeline definition
├── mlflow.db            # Local MLflow tracking database
└── requirements.txt     # Project dependencies
```

## 🔄 MLOps Workflow

### 1. Data Versioning (DVC)
Data is managed using DVC to ensure reproducibility. Large datasets are tracked without bloating the Git repository.
-   `dvc.yaml`: Defines the training stage and its dependencies.

### 2. Experiment Tracking (MLflow)
Every training run is logged to MLflow, capturing:
-   **Hyperparameters**: learning rate, depth, iterations, etc.
-   **Metrics**: Accuracy, AUC (Area Under Curve).
-   **Artifacts**: The trained CatBoost model.

### 3. Automated Model Promotion
The `src/promote.py` script ensures that only the best-performing models reach "Production" status:
-   Compares the AUC of the new model against the current production model.
-   If the new model is better, it is registered in the **MLflow Model Registry** with a `production` alias.
-   Automatically commits and pushes changes to Git to trigger downstream deployments.

## 💻 Local Setup

### Installation
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Prince-darji-2306/Heart-MLOps.git
    cd Heart-MLOps
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Pipeline
-   **Train the model**:
    ```bash
    dvc repro
    ```
-   **Launch the Web App**:
    ```bash
    streamlit run app/main.py
    ```

## 📊 Dataset Reference
The model is trained on patient health metrics including age, cholesterol, maximum heart rate, and ST depression to predict the presence or absence of heart disease.

---
Built with ❤️ by Prince Darji
