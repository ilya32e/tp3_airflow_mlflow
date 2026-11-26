"""check_drift.py - Monitoring : les nouvelles données ressemblent-elles à celles du dernier entraînement ?

Pour chaque variable, on mesure l'écart entre la moyenne des nouvelles données et celle
des données de référence, en nombre d'écarts-types (même idée qu'un « effet de taille »).
drift_score = l'écart de la variable qui a le plus bougé.
Au-delà du seuil (0,2 par défaut), on considère qu'il y a dérive et qu'il faut réentraîner.
"""
import argparse
import os

import mlflow
import pandas as pd
from dotenv import load_dotenv

TARGET = "default"
EXPERIMENT = "credit-monitoring"      # expérience MLflow dédiée au monitoring

load_dotenv()


def drift_by_column(new, ref):
    """Écart des moyennes, en nombre d'écarts-types de la référence, pour chaque variable."""
    features = [c for c in ref.columns if c != TARGET]
    ecarts = (new[features].mean() - ref[features].mean()).abs() / ref[features].std()
    return ecarts.sort_values(ascending=False)


def check_drift(new_path, ref_path, threshold):
    """Renvoie True s'il faut réentraîner (dérive détectée ou aucune référence)."""
    # Premier passage : aucun modèle n'a encore été entraîné par le pipeline
    if not os.path.exists(ref_path):
        print("Aucune donnée de référence : premier entraînement.")
        return True

    new = pd.read_csv(new_path, index_col=0)
    ref = pd.read_csv(ref_path, index_col=0)
    ecarts = drift_by_column(new, ref)
    drift_score = float(ecarts.iloc[0])
    drift_detected = drift_score > threshold

    # Chaque contrôle devient un run MLflow : on suit la dérive dans le temps
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="drift-check"):
        mlflow.log_params({"new_data": new_path, "reference": ref_path, "threshold": threshold})
        mlflow.log_metric("drift_score", drift_score)
        mlflow.log_metrics({f"drift_{col}": float(v) for col, v in ecarts.items()})
        mlflow.log_metric("default_rate", float(new[TARGET].mean()))
        mlflow.set_tag("variable_max", ecarts.index[0])
        mlflow.set_tag("decision", "réentraîner" if drift_detected else "rien à faire")

    print("Variables qui ont le plus bougé :")
    print(ecarts.head(5).round(3).to_string())
    print(f"drift_score = {drift_score:.3f} (seuil {threshold}) -> "
          f"{'DÉRIVE : réentraînement' if drift_detected else 'pas de dérive : on garde le modèle'}")
    return drift_detected


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", default="data/credit_clean.csv", help="Données préparées du nouveau lot")
    parser.add_argument("--ref", default="data/reference.csv", help="Données du dernier entraînement")
    parser.add_argument("--threshold", type=float, default=0.2, help="Seuil de dérive")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    check_drift(args.new, args.ref, args.threshold)
