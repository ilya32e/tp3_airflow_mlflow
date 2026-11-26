# Image Airflow officielle + les bibliothèques du TP 2 (MLflow, scikit-learn)
FROM apache/airflow:3.4.0-python3.12

COPY requirements.txt /requirements.txt

# On réinstalle la même version d'Airflow avec les dépendances :
# pip ne peut pas mettre à jour ou casser Airflow par accident.
RUN pip install --no-cache-dir "apache-airflow==3.4.0" -r /requirements.txt
