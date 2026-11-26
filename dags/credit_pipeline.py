"""credit_pipeline.py - TP 3 : entraînement continu du modèle de défaut de paiement.

préparer -> contrôler la dérive -> entraîner -> enregistrer si meilleur

Airflow orchestre (quand, dans quel ordre, relances, logs) ;
MLflow garde la mémoire (runs, métriques, modèles, dérive).
Les scripts sont ceux du TP 2 : le DAG ne fait que les appeler.
"""
import sys
from datetime import datetime

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import Param, dag, task

BASE = "/opt/airflow"                 # dossier du projet dans le conteneur
sys.path.insert(0, f"{BASE}/scripts")  # pour importer les fonctions des scripts

CLEAN = "data/credit_clean.csv"       # données préparées du lot en cours
REFERENCE = "data/reference.csv"      # données du dernier entraînement


@dag(
    schedule="@monthly",              # chaque mois, un nouveau lot de clients arrive
    start_date=datetime(2026, 11, 1),
    catchup=False,
    tags=["mlops", "tp3"],
    # Paramètres modifiables au lancement (bouton « Trigger » de l'interface)
    params={
        "raw_file": Param("data/default_of_credit_card_clients.csv", type="string",
                          description="Fichier brut du lot à traiter"),
        "drift_threshold": Param(0.2, type="number",
                                 description="Seuil de dérive au-delà duquel on réentraîne"),
    },
)
def credit_pipeline():

    # 1. Préparer : le script du TP 2, appelé tel quel
    prepare = BashOperator(
        task_id="prepare_data",
        bash_command=f"python scripts/prepare_data.py --raw {{{{ params.raw_file }}}} --out {CLEAN}",
        cwd=BASE,
    )

    # 2. Monitoring : s'il n'y a pas de dérive, les tâches suivantes sont sautées (skipped)
    @task.short_circuit
    def check_drift(params=None):
        from check_drift import check_drift as compute
        return compute(f"{BASE}/{CLEAN}", f"{BASE}/{REFERENCE}", params["drift_threshold"])

    # 3. Entraîner : toujours le script du TP 2 ; le run apparaît dans MLflow
    train = BashOperator(
        task_id="train",
        bash_command=(f"python scripts/train.py --data {CLEAN} "
                      "--model rf --class_weight balanced --n_estimators 200 --max_depth 8"),
        cwd=BASE,
    )

    # 4. Champion / challenger : on ne déplace l'alias que si le nouveau modèle fait mieux
    @task
    def register_if_better():
        from register_if_better import register_if_better as decide
        decide(f"{BASE}/{CLEAN}", f"{BASE}/{REFERENCE}")

    prepare >> check_drift() >> train >> register_if_better()


credit_pipeline()
