"""
02_run_contrastive.py — Método proposto: Contrastive Encoder + Busca Bidirecional no Espaço Latente

Pipeline completo (sem labels em nenhum momento da seleção):
  1. Pré-processar DataSense
  2. Treinar encoder contrastivo (NT-Xent, não supervisionado)
  3. Busca bidirecional no espaço latente (masked forward pass + k-Means + Silhouette)
  4. Salvar encoder, features selecionadas e histórico

Uso:
    python experiments/02_run_contrastive.py
    python experiments/02_run_contrastive.py --csv data/benign_samples_1sec.csv.csv
    python experiments/02_run_contrastive.py --skip-train  # usa encoder já salvo
"""

import sys
import os
import json
import time
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BENIGN_CSV, FULL_CSV, TABLES_DIR, MODELS_DIR, MAX_SUBSAMPLE
from src.preprocessing import DataSensePreprocessor
from src.contrastive_fs import (
    train_contrastive_encoder,
    ContrastiveBidirectionalFS,
    get_representations,
    load_encoder,
)


ENCODER_PATH = os.path.join(MODELS_DIR, "encoder_datasense.pt")


def run(csv_path: str, skip_train: bool = False) -> dict:
    print(f"\n{'='*60}")
    print(f"EXPERIMENTO 02 — Contrastive Encoder + Busca Latente")
    print(f"{'='*60}")
    print(f"Dataset: {csv_path}\n")

    # ------------------------------------------------------------------
    # 1. Pré-processamento
    # ------------------------------------------------------------------
    pp = DataSensePreprocessor(csv_path)
    X, y = pp.load()
    feature_names = pp.get_feature_names()
    context_groups = pp.get_context_groups()

    print(f"Dataset carregado: {X.shape[0]:,} amostras x {X.shape[1]} features")
    if "label1" in y.columns:
        print(f"Classes (label1): {y['label1'].value_counts().to_dict()}")

    # ------------------------------------------------------------------
    # 2. Treinar encoder (ou carregar existente)
    # ------------------------------------------------------------------
    t_train_start = time.time()
    if skip_train and os.path.exists(ENCODER_PATH):
        print(f"\n[Encoder] Carregando encoder existente: {ENCODER_PATH}")
        encoder, _ = load_encoder(ENCODER_PATH)
        train_time = 0.0
        loss_history = []
    else:
        # Subsamplar para treino do encoder (eficiência em datasets grandes)
        if len(X) > MAX_SUBSAMPLE:
            print(f"\n[Encoder] Subsampling para treino: {MAX_SUBSAMPLE:,} / {len(X):,} amostras")
            X_train, _ = pp.subsample(X, y, n_samples=MAX_SUBSAMPLE, stratify_col="label2"
                                      if "label2" in y.columns else "label1")
        else:
            X_train = X
        print(f"\n[Encoder] Treinando encoder contrastivo em {len(X_train):,} amostras...")
        encoder, loss_history = train_contrastive_encoder(
            X_train,
            feature_names,
            save_path=ENCODER_PATH,
            verbose=True,
        )
        train_time = time.time() - t_train_start

    # ------------------------------------------------------------------
    # 3. Busca bidirecional no espaço latente
    # ------------------------------------------------------------------
    print("\n[Busca] Iniciando busca bidirecional no espaço latente...")
    t_search_start = time.time()
    fs = ContrastiveBidirectionalFS(encoder=encoder, verbose=True)
    fs.fit(X)
    search_time = time.time() - t_search_start

    selected_names = fs.get_selected_features(feature_names)
    total_time = train_time + search_time

    # ------------------------------------------------------------------
    # 4. Relatório
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"RESULTADO")
    print(f"{'='*60}")
    print(f"Features selecionadas : {len(selected_names)} / {len(feature_names)}")
    print(f"Taxa de reducao       : {1 - len(selected_names)/len(feature_names):.1%}")
    print(f"Silhouette (latente)  : {fs.best_score_:.4f}")
    print(f"k otimo               : {fs.best_k_}")
    print(f"Tempo treino encoder  : {train_time:.1f}s")
    print(f"Tempo busca           : {search_time:.1f}s")
    print(f"Tempo total           : {total_time:.1f}s")
    print(f"Subconjuntos avaliados: {len(fs._cache)}")
    print(f"Iteracoes             : {len(fs.history_)}")

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

    print(f"\nPor grupo semantico:")
    for g in sorted(group_sel.keys()):
        total_in_group = len([i for i, grp in idx_to_group.items() if grp == g])
        print(f"  {g}: {len(group_sel[g])}/{total_in_group} -> {group_sel[g]}")

    if loss_history:
        print(f"\nLoss encoder: {loss_history[0]:.4f} (ep.1) -> {loss_history[-1]:.4f} (ep.{len(loss_history)})")

    # ------------------------------------------------------------------
    # 5. Salvar resultados
    # ------------------------------------------------------------------
    os.makedirs(TABLES_DIR, exist_ok=True)

    result = {
        "method": "contrastive_bidirectional",
        "csv": os.path.basename(csv_path),
        "n_samples": int(X.shape[0]),
        "n_features_total": int(X.shape[1]),
        "n_features_selected": len(selected_names),
        "reduction_ratio": round(1 - len(selected_names) / len(feature_names), 4),
        "silhouette_latent": round(float(fs.best_score_), 6),
        "best_k": int(fs.best_k_),
        "train_time_s": round(train_time, 2),
        "search_time_s": round(search_time, 2),
        "total_time_s": round(total_time, 2),
        "n_evaluations": len(fs._cache),
        "n_iterations": len(fs.history_),
        "selected_features": selected_names,
        "loss_first": round(loss_history[0], 6) if loss_history else None,
        "loss_last": round(loss_history[-1], 6) if loss_history else None,
    }

    out_json = os.path.join(TABLES_DIR, "contrastive_result.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nResultado salvo em : {out_json}")

    hist_df = pd.DataFrame(fs.history_)
    out_hist = os.path.join(TABLES_DIR, "contrastive_history.csv")
    hist_df.to_csv(out_hist, index=False)
    print(f"Historico salvo em : {out_hist}")

    if loss_history:
        loss_df = pd.DataFrame({"epoch": range(1, len(loss_history) + 1), "loss": loss_history})
        out_loss = os.path.join(TABLES_DIR, "contrastive_loss_curve.csv")
        loss_df.to_csv(out_loss, index=False)
        print(f"Loss curve salva em: {out_loss}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None)
    parser.add_argument("--skip-train", action="store_true", help="Usar encoder ja salvo")
    args = parser.parse_args()

    if args.csv:
        csv_path = args.csv
    elif os.path.exists(FULL_CSV):
        print(f"Dataset completo encontrado: {FULL_CSV}")
        csv_path = FULL_CSV
    else:
        print(f"Usando dataset benign: {BENIGN_CSV}")
        csv_path = BENIGN_CSV

    run(csv_path, skip_train=args.skip_train)
