"""
06_run_xai_analysis.py — Análise XAI detalhada com SHAP

Pipeline:
  1. Carrega preprocessamento + encoder + resultado contrastivo
  2. Gera cluster labels no espaço latente
  3. Treina proxy RF e calcula SHAP values
  4. Gera: summary plot, heatmap, waterfall por cluster, t-SNE, relatório textual

Uso:
    python experiments/06_run_xai_analysis.py
    python experiments/06_run_xai_analysis.py --csv data/benign_samples_1sec.csv.csv
"""

import sys
import os
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BENIGN_CSV, FULL_CSV, FIGURES_DIR, TABLES_DIR, MODELS_DIR
from src.preprocessing import DataSensePreprocessor
from src.contrastive_fs import get_representations, load_encoder
from src.xai_explainer import ClusterSHAPExplainer, cluster_latent_space

ENCODER_PATH = os.path.join(MODELS_DIR, "encoder_datasense.pt")
CONTRASTIVE_RESULT = os.path.join(TABLES_DIR, "contrastive_result.json")


def run(csv_path: str) -> None:
    print(f"\n{'='*60}")
    print(f"EXPERIMENTO 06 — Análise XAI (SHAP + t-SNE)")
    print(f"{'='*60}")
    print(f"Dataset: {csv_path}\n")

    # ------------------------------------------------------------------
    # 1. Pré-processamento
    # ------------------------------------------------------------------
    pp = DataSensePreprocessor(csv_path)
    X, y = pp.load()
    feature_names = pp.get_feature_names()
    context_groups = pp.get_context_groups()

    print(f"Dataset: {X.shape[0]:,} x {X.shape[1]}")

    # ------------------------------------------------------------------
    # 2. Carregar encoder e resultado contrastivo
    # ------------------------------------------------------------------
    if not os.path.exists(ENCODER_PATH):
        print(f"ERRO: encoder não encontrado em {ENCODER_PATH}")
        print("Execute primeiro: python experiments/02_run_contrastive.py")
        sys.exit(1)

    print(f"\n[XAI] Carregando encoder: {ENCODER_PATH}")
    encoder, _ = load_encoder(ENCODER_PATH)

    # Carregar features selecionadas do resultado contrastivo
    selected_features: list[str] = []
    best_k = 5
    if os.path.exists(CONTRASTIVE_RESULT):
        with open(CONTRASTIVE_RESULT, encoding="utf-8") as f:
            result = json.load(f)
        selected_features = result["selected_features"]
        best_k = result["best_k"]
        print(f"[XAI] Features selecionadas carregadas: {len(selected_features)}")
        print(f"[XAI] k do resultado contrastivo: {best_k}")
    else:
        print(f"[XAI] Resultado contrastivo não encontrado. Usando todas as features.")
        selected_features = feature_names
        best_k = 5

    # ------------------------------------------------------------------
    # 3. Obter representações latentes e cluster labels
    # ------------------------------------------------------------------
    print(f"\n[XAI] Extraindo representações latentes (128d)...")
    H = get_representations(encoder, X)
    print(f"[XAI] Shape representações: {H.shape}")

    print(f"[XAI] Clusterizando no espaço latente (k={best_k})...")
    cluster_labels = cluster_latent_space(H, n_clusters=best_k)
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"[XAI] Distribuição de clusters:")
    for c, n in zip(unique, counts):
        print(f"  Cluster {c}: {n:,} amostras ({n/len(cluster_labels):.1%})")

    # ------------------------------------------------------------------
    # 4. Montar X com features selecionadas para o proxy
    # ------------------------------------------------------------------
    sel_indices = [feature_names.index(f) for f in selected_features if f in feature_names]
    X_selected = X[:, sel_indices]
    sel_feature_names = [feature_names[i] for i in sel_indices]
    print(f"\n[XAI] X_selected shape: {X_selected.shape}")

    # ------------------------------------------------------------------
    # 5. Treinar proxy e calcular SHAP
    # ------------------------------------------------------------------
    explainer = ClusterSHAPExplainer(
        feature_names=sel_feature_names,
        context_groups=context_groups,
    )
    explainer.fit(X_selected, cluster_labels)

    # ------------------------------------------------------------------
    # 6. Gerar visualizações
    # ------------------------------------------------------------------
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print(f"\n[XAI] Gerando visualizações...")

    # SHAP summary (beeswarm)
    explainer.plot_shap_summary(FIGURES_DIR)

    # Heatmap features x clusters
    explainer.plot_feature_heatmap(FIGURES_DIR)

    # Waterfall por cluster
    for c in range(best_k):
        explainer.plot_cluster_waterfall(c, FIGURES_DIR)

    # t-SNE — usar label1 como true labels se disponível e não for tudo benign
    true_labels = None
    if "label1" in y.columns:
        lbl_vals = y["label1"].values
        if len(np.unique(lbl_vals)) > 1:
            true_labels = lbl_vals
        else:
            # Benign-only: usar device se disponível, senão None
            true_labels = None

    print(f"\n[XAI] Gerando t-SNE (pode demorar ~2 min)...")
    explainer.plot_tsne_latent(
        representations=H,
        cluster_labels=cluster_labels,
        true_labels=true_labels,
        output_dir=FIGURES_DIR,
    )

    # ------------------------------------------------------------------
    # 7. Relatório textual
    # ------------------------------------------------------------------
    print(f"\n[XAI] Gerando relatório textual...")
    explainer.generate_text_report(selected_features, TABLES_DIR)

    print(f"\n{'='*60}")
    print(f"Figuras salvas em : {FIGURES_DIR}")
    print(f"Relatório salvo em: {TABLES_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    if args.csv:
        csv_path = args.csv
    elif os.path.exists(FULL_CSV):
        csv_path = FULL_CSV
    else:
        csv_path = BENIGN_CSV

    run(csv_path)
