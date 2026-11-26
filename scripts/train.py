"""train.py - Entraînement du modèle de défaut de paiement, suivi avec MLflow (script du TP 2, inchangé)."""
import argparse

import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "default"
EXPERIMENT = "credit-default"

# Dans Airflow il n'y a pas de .env : load_dotenv() ne fait rien et MLflow lit
# les variables MLFLOW_TRACKING_* définies dans le docker-compose.
load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/credit_clean.csv")
    parser.add_argument("--model", default="logreg", choices=["logreg", "rf"])
    parser.add_argument("--class_weight", default="none", choices=["none", "balanced"])
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)
    return parser.parse_args()


def build_model(args):
    cw = None if args.class_weight == "none" else args.class_weight
    if args.model == "logreg":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(class_weight=cw, max_iter=1000))
    return RandomForestClassifier(n_estimators=args.n_estimators,
                                  max_depth=args.max_depth,
                                  class_weight=cw, random_state=42, n_jobs=-1)


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_proba),
    }


def main():
    args = parse_args()

    df = pd.read_csv(args.data, index_col=0)
    X = df.drop(columns=[TARGET]).astype("float64")
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y)

    mlflow.set_experiment(EXPERIMENT)
    run_name = f"{args.model}-{args.class_weight}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(vars(args))

        model = build_model(args)
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        signature = mlflow.models.infer_signature(X_test, model.predict(X_test))
        info = mlflow.sklearn.log_model(model, name="model", signature=signature,
                                        input_example=X_test.head(3),
                                        skops_trusted_types=["sklearn.tree._tree.Tree"])
        mlflow.set_tag("model_uri", info.model_uri)

        for k, v in metrics.items():
            print(f"{k:>10} : {v:.3f}")
        print("Run :", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    main()
