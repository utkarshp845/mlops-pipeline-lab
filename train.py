import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlflow.models import infer_signature

# Hyperparameters
C = 1.0
MAX_ITER = 200
SOLVER = "lbfgs"

def train():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    mlflow.set_experiment("iris-logistic")

    with mlflow.start_run() as run:
        model = LogisticRegression(C=C, max_iter=MAX_ITER, solver=SOLVER)
        model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, model.predict(X_test))

        mlflow.log_params({"C": C, "max_iter": MAX_ITER, "solver": SOLVER})
        mlflow.log_metric("accuracy", accuracy)

        signature = infer_signature(X_test, model.predict(X_test))
        mlflow.sklearn.log_model(model, "model", signature=signature, input_example=X_test[:1])

        os.makedirs("model", exist_ok=True)
        joblib.dump(model, "model/model.pkl")

        print(f"Run ID : {run.info.run_id}")
        print(f"Accuracy: {accuracy:.4f}")
        print("Model saved to model/model.pkl")

if __name__ == "__main__":
    train()
