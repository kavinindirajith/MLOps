import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model(X_train, y_train, seed=42):
    model = RandomForestClassifier(random_state=seed)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return acc, report

from mlflow.tracking import MlflowClient

def register_model(model, name="heart_attack_rf"):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("heart_attack")

    with mlflow.start_run() as run:
        mlflow.log_params(model.get_params())

        # Log the model and register it
        logged_model = mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=name
        )

        # Get the run ID and version from model URI
        client = MlflowClient()
        latest_versions = client.get_latest_versions(name, stages=["None"])
        if latest_versions:
            version = latest_versions[0].version

            # Transition it to Production stage
            client.transition_model_version_stage(
                name=name,
                version=version,
                stage="Production",
                archive_existing_versions=True  # optional: archives older Production model
            )
            print(f"✅ Registered model v{version} promoted to Production.")
        else:
            print("⚠️ Model registration failed or not found.")

def load_registered_model(name="heart_attack_rf", stage="Production"):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model_uri = f"models:/{name}/{stage}"
    print(f"Loading model from: {model_uri}")
    
    try:
        model = mlflow.sklearn.load_model(model_uri)
        print("✅ Model loaded successfully.")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise

