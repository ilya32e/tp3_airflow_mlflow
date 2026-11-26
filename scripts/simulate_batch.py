"""simulate_batch.py - Fabrique un « nouveau lot » de clients pour la démonstration du monitoring.

  --mode normal : 10 000 clients tirés au hasard dans le fichier d'origine (pas de dérive)
  --mode crise  : même tirage, puis une crise économique simulée :
                  plafonds de crédit réduits de 30 %, remboursements en baisse de 40 %,
                  et les jeunes clients très endettés font beaucoup plus souvent défaut.
Le fichier produit a le même format brut que l'original (double en-tête) :
il passe donc par prepare_data.py comme un vrai fichier reçu du fournisseur.
"""
import argparse

import numpy as np
import pandas as pd

RAW_TARGET = "default payment next month"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/default_of_credit_card_clients.csv")
    parser.add_argument("--mode", default="normal", choices=["normal", "crise"])
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    out = args.out or f"data/lot_{args.mode}.csv"

    # Lecture du fichier brut en gardant ses noms d'origine (PAY_0, cible avec espaces)
    raw = pd.read_csv(args.raw, header=1, index_col=0, encoding="utf-8-sig")
    lot = raw.sample(n=10_000, random_state=7)

    if args.mode == "crise":
        rng = np.random.default_rng(7)
        pay_amt = [c for c in lot.columns if c.startswith("PAY_AMT")]
        lot["LIMIT_BAL"] = (lot["LIMIT_BAL"] * 0.7).round(-4).clip(lower=10_000)
        lot[pay_amt] = (lot[pay_amt] * 0.6).round()
        fragiles = (lot["AGE"] < 35) & (lot["BILL_AMT1"] > 0.5 * lot["LIMIT_BAL"])
        lot.loc[fragiles & (rng.random(len(lot)) < 0.7), RAW_TARGET] = 1

    # Réécriture au format brut : première ligne X1...X23, Y comme dans le fichier d'origine
    first_line = "," + ",".join([f"X{i}" for i in range(1, lot.shape[1])] + ["Y"])
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        f.write(first_line + "\n")
        lot.to_csv(f, index_label="ID")
    print(f"{len(lot)} clients, taux de défaut = {lot[RAW_TARGET].mean():.1%} -> {out}")


if __name__ == "__main__":
    main()
