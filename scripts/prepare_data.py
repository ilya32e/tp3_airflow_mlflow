"""prepare_data.py - Préparation du jeu de données credit-card-clients (script du TP 2, inchangé)."""
import argparse

import pandas as pd

TARGET = "default"
RAW_TARGET = "default payment next month"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/default_of_credit_card_clients.csv")
    parser.add_argument("--out", default="data/credit_clean.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.raw, header=1, index_col=0, encoding="utf-8-sig")
    df = df.rename(columns={RAW_TARGET: TARGET, "PAY_0": "PAY_1"})

    assert TARGET in df.columns, f"Colonne cible '{TARGET}' introuvable"
    assert df.isna().sum().sum() == 0, "Valeurs manquantes détectées"
    assert set(df[TARGET].unique()) <= {0, 1}, "La cible doit valoir 0 ou 1"

    df.to_csv(args.out)
    print(f"{df.shape[0]} lignes, {df.shape[1]} colonnes, "
          f"taux de défaut = {df[TARGET].mean():.1%} -> {args.out}")


if __name__ == "__main__":
    main()
