"""
01_run_baseline_original.py — Baseline: k-Means + Silhouette bidirecional (dissertação 2022)

Valida o pipeline de pré-processamento e busca bidirecional antes do encoder contrastivo.
Usa dataset benign se o completo não estiver disponível.

Uso:
    python experiments/01_run_baseline_original.py
    python experiments/01_run_baseline_original.py --csv data/all_attack_samples.csv
"""

import sys
import os
import json
import time
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BENIGN_CSV, FULL_CSV, TABLES_DIR, CONTEXT_GROUPS
from src.preprocessing import DataSensePreprocessor
from src.baselines import BidirectionalSilhouetteFS


def run(csv_path: str) -> dict:
    print(f"\n{'='*60}")
    print(f"EXPERIMENTO 01 — Baseline k-Means + Silhouette Bidirecional")
    print(f"{'='*60}")
    print(f"Dataset: {csv_path}\n")

    # 1. Pré-processamento
    pp = DataSensePreprocessor(csv_path)
    X, y = pp.load()
    feature_names = pp.get_feature_names()
    context_groups = pp.get_context_groups()

    print(f"\nDataset carregado: {X.shape[0]:,} amostras × {X.shape[1]} features")
    if "label1" in y.columns:
        print(f"Classes (label1): {y['label1'].value_counts().to_dict()}")

    # 2. Busca bidirecional
    print()
    t0 = time.time()
    fs = BidirectionalSilhouetteFS(verbose=True)
    fs.fit(X)
    elapsed = time.time() - t0

    selected_names = fs.get_selected_features(feature_names)

    # 3. Relatório
    print(f"\n{'='*60}")
    print(f"RESULTADO")
    print(f"{'='*60}")
    print(f"Features selecionadas : {len(selected_names)} / {len(feature_names)}")
    print(f"Taxa de redução        : {1 - len(selected_names)/len(feature_names):.1%}")
    print(f"Silhouette score       : {fs.best_score_:.4f}")
    print(f"k ótimo                : {fs.best_k_}")
    print(f"Tempo de seleção       : {elapsed:.1f}s")
    print(f"Subconjuntos avaliados : {len(fs._cache)}")
    print(f"Iterações realizadas   : {len(fs.history_)}")

    print(f"\nFeatures selecionadas:")
    for i, name in enumerate(selected_names, 1):
        print(f"  {i:2d}. {name}")

    # Breakdown por grupo semântico
    idx_to_group: dict[int, str] = {}
    for g, indices in context_groups.items():
        for idx in indices:
            idx_to_group[idx] = g

    group_sel: dict[str, list[str]] = {}
    for idx in fs.selected_indices_:
        g = idx_to_group.get(idx, "Unknown")
        group_sel.setdefault(g, []).append(feature_names[idx])

    print(f"\nPor grupo semântico:")
    for g in sorted(group_sel.keys()):
        total_in_group = len([i for i, grp in idx_to_group.items() if grp == g])
        sel_in_group = len(group_sel[g])
        print(f"  {g}: {sel_in_group}/{total_in_group} -> {group_sel[g]}")

    # 4. Salvar resultados
    os.makedirs(TABLES_DIR, exist_ok=True)

    result = {
        "method": "kmeans_silhouette_bidirectional",
        "csv": os.path.basename(csv_path),
        "n_samples": int(X.shape[0]),
        "n_features_total": int(X.shape[1]),
        "n_features_selected": len(selected_names),
        "reduction_ratio": round(1 - len(selected_names) / len(feature_names), 4),
        "silhouette_score": round(float(fs.best_score_), 6),
        "best_k": int(fs.best_k_),
        "selection_time_s": round(elapsed, 2),
        "n_evaluations": len(fs._cache),
        "n_iterations": len(fs.history_),
        "selected_features": selected_names,
    }

    out_json = os.path.join(TABLES_DIR, "baseline_original_result.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nResultado salvo em: {out_json}")

    hist_df = pd.DataFrame(fs.history_)
    out_hist = os.path.join(TABLES_DIR, "baseline_original_history.csv")
    hist_df.to_csv(out_hist, index=False)
    print(f"Histórico salvo em : {out_hist}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=None,
        help="Caminho do CSV (padrão: usa completo se disponível, senão benign)",
    )
    args = parser.parse_args()

    if args.csv:
        csv_path = args.csv
    elif os.path.exists(FULL_CSV):
        print(f"Dataset completo encontrado: {FULL_CSV}")
        csv_path = FULL_CSV
    else:
        print(f"Dataset completo não encontrado. Usando benign: {BENIGN_CSV}")
        csv_path = BENIGN_CSV

    run(csv_path)
