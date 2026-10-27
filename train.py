"""Train a model, log it to MLflow, and promote it only if it beats the current one."""
import json
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"


def train(n_estimators: int = 200, max_depth: int = 8, seed: int = 42) -> dict:
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=seed, stratify=data.target
    )

    mlflow.set_experiment("breast-cancer-classifier")
    with mlflow.start_run():
        params = {"n_estimators": n_estimators, "max_depth": max_depth, "seed": seed}
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=seed
        ).fit(X_train, y_train)

        preds = model.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1": float(f1_score(y_test, preds)),
        }
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

    return {
        "model": model,
        "metrics": metrics,
        "feature_names": list(data.feature_names),
        "target_names": list(data.target_names),
    }


def promote_if_better(result: dict) -> bool:
    """Save the model only if it matches or beats the currently deployed one."""
    new_acc = result["metrics"]["accuracy"]
    if METRICS_PATH.exists():
        old_acc = json.loads(METRICS_PATH.read_text())["accuracy"]
        if new_acc < old_acc:
            print(f"Rejected: new accuracy {new_acc:.4f} < current {old_acc:.4f}")
            return False

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": result["model"],
            "feature_names": result["feature_names"],
            "target_names": result["target_names"],
        },
        MODEL_PATH,
    )
    METRICS_PATH.write_text(json.dumps(result["metrics"], indent=2))
    print(f"Promoted: accuracy {new_acc:.4f}")
    return True


if __name__ == "__main__":
    promoted = promote_if_better(train())
    # Non-zero exit lets CI skip the deploy step when the model is not better.
    sys.exit(0 if promoted else 1)
