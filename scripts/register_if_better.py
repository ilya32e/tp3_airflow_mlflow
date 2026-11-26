"""register_if_better.py - Champion / challenger : on ne remplace le modèle en service que s'il fait mieux.

Le challenger est le dernier run de train.py. Le champion est le modèle qui porte l'alias
« champion » dans le Registry. Les deux sont comparés sur le MÊME jeu de test
(mêmes données, même découpage que train.py), sinon la comparaison n'aurait pas de sens.
"""
import argparse
import shutil

import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

TARGET = "default"
EXPERIMENT = "credit-default"
MODEL_NAME = "credit-default-model"
METRIC = "recall"

load_dotenv()


def register_if_better(data_path, ref_path, test_size=0.3):
    client = MlflowClient()

    # 1. Challenger = run le plus récent de l'expérience (celui que train.py vient de créer)
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT],
                              order_by=["attributes.start_time DESC"], max_results=1)
    challenger = runs.iloc[0]
    challenger_score = challenger[f"metrics.{METRIC}"]
    print(f"Challenger : {challenger['tags.mlflow.runName']} ({METRIC} = {challenger_score:.3f})")

    # 2. Champion actuel, évalué sur le même jeu de test que le challenger
    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except MlflowException:
        champion = None

    if champion is None:
        print("Aucun champion : le challenger devient le premier champion.")
        better = True
    else:
        df = pd.read_csv(data_path, index_col=0)
        X = df.drop(columns=[TARGET]).astype("float64")
        y = df[TARGET]
        _, X_test, _, y_test = train_test_split(X, y, test_size=test_size,
                                                random_state=42, stratify=y)
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
        champion_score = recall_score(y_test, model.predict(X_test))
        print(f"Champion : version {champion.version} ({METRIC} = {champion_score:.3f} sur les mêmes données)")
        better = challenger_score > champion_score

    # 3. Décision
    if better:
        version = mlflow.register_model(challenger["tags.model_uri"], MODEL_NAME)
        client.set_registered_model_alias(MODEL_NAME, "champion", version.version)
        print(f"Nouveau champion : {MODEL_NAME} version {version.version}")
    else:
        print("Le champion reste en service.")

    # 4. Ces données deviennent la référence pour le prochain contrôle de dérive
    shutil.copy(data_path, ref_path)
    return better


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/credit_clean.csv")
    parser.add_argument("--ref", default="data/reference.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    register_if_better(args.data, args.ref)
