"""
04b_datasense17_eval.py — Avalia as 17 features do DataSense paper (Tabela 7)

Comparacao suplementar: nosso metodo (25 features, nao-supervisionado) vs
DataSense-17 (17 features, supervisionado), como referencia superior.

Uso:
    python experiments/04b_datasense17_eval.py
"""

import sys, os, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FULL_CSV, BENIGN_CSV, TABLES_DIR, DATASENSE_SELECTED_17
from src.preprocessing import DataSensePreprocessor
from src.evaluation import EvaluationPipeline


def run(csv_path: str) -> None:
    print(f"\n{'='*60}")
    print("EXPERIMENTO 04b — DataSense-17 vs Contrastivo (25 features)")
    print(f"{'='*60}")
    print(f"Dataset: {csv_path}\n")

    pp = DataSensePreprocessor(csv_path)
    X, y = pp.load()
    feature_names = pp.get_feature_names()
    print(f"Dataset: {X.shape[0]:,} x {X.shape[1]}")

    name_to_idx = {n: i for i, n in enumerate(feature_names)}

    # DataSense-17 (supervisionado — referencia do paper)
    ds17_indices = [name_to_idx[f] for f in DATASENSE_SELECTED_17 if f in name_to_idx]
    ds17_avail = [f for f in DATASENSE_SELECTED_17 if f in name_to_idx]
    print(f"\nDataSense-17: {len(ds17_indices)} de {len(DATASENSE_SELECTED_17)} features encontradas")
    if len(ds17_indices) < len(DATASENSE_SELECTED_17):
        missing = [f for f in DATASENSE_SELECTED_17 if f not in name_to_idx]
        print(f"  Ausentes: {missing}")

    # Contrastivo (nao-supervisionado — nosso metodo)
    contra_path = os.path.join(TABLES_DIR, "contrastive_result.json")
    with open(contra_path) as f:
        contra = json.load(f)
    contra_indices = [name_to_idx[f] for f in contra["selected_features"] if f in name_to_idx]

    feature_sets = {
        "datasense_17_supervised": ds17_indices,
        "contrastive_25_unsupervised": contra_indices,
        "all_features_71": list(range(len(feature_names))),
    }

    include_knn = X.shape[0] <= 50000
    pipe = EvaluationPipeline(include_knn=include_knn)
    label_cols = [c for c in ["label1", "label2"] if c in y.columns and y[c].nunique() >= 2]

    os.makedirs(TABLES_DIR, exist_ok=True)

    for label_col in label_cols:
        print(f"\n{'='*60}")
        print(f"Cenario: {label_col} | {y[label_col].nunique()} classes")
        print(f"{'='*60}")

        df = pipe.run_all(X, y, feature_sets, label_col=label_col, verbose=True)
        if df.empty:
            continue

        out = os.path.join(TABLES_DIR, f"datasense17_vs_contrastive_{label_col}.csv")
        df.to_csv(out, index=False)

        # Tabela resumida
        summary = (
            df.groupby("method")
            .agg(
                n_features=("n_features", "first"),
                f1_mean=("f1_macro_mean", "mean"),
                f1_std=("f1_macro_std", "mean"),
                acc_mean=("accuracy_mean", "mean"),
                mcc_mean=("mcc_mean", "mean"),
            )
            .sort_values("f1_mean", ascending=False)
        )
        print(f"\n{'Metodo':<35} {'#Feat':>5}  {'F1-macro':>12}  {'Acc':>8}  {'MCC':>8}")
        print(f"{'-'*35} {'-'*5}  {'-'*12}  {'-'*8}  {'-'*8}")
        for method, row in summary.iterrows():
            print(f"  {method:<33} {int(row.n_features):>5}  "
                  f"{row.f1_mean:.3f}+-{row.f1_std:.3f}  "
                  f"{row.acc_mean:.3f}     "
                  f"{row.mcc_mean:.3f}")
        print(f"\nResultados salvos em: {out}")

    print(f"\n{'='*60}")
    print("Avaliacao DataSense-17 vs Contrastivo concluida.")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=None)
    args = p.parse_args()
    csv_path = args.csv or (FULL_CSV if os.path.exists(FULL_CSV) else BENIGN_CSV)
    run(csv_path)
