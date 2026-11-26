# TP 3 – MLflow dans Airflow (entraînement continu + monitoring)

1. Copier `.env.example` en `.env` et remplir les 3 variables MLflow (les mêmes qu'au TP 2).
2. Copier `default_of_credit_card_clients.csv` (TP 2) dans `data/`.
3. `docker compose up airflow-init` puis `docker compose up -d --build`
4. Interface Airflow : http://localhost:8080 (airflow / airflow), DAG `credit_pipeline`.

Pipeline : prepare_data -> check_drift -> train -> register_if_better
Lots de démonstration : `docker compose exec airflow-worker python scripts/simulate_batch.py --mode normal` (ou `--mode crise`)
