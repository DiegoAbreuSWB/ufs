"""
08_run_k_sweep.py — Avaliação com k variável: 10, 15, 20, 25 features

Objetivo: mostrar que com k pequeno (10 features), métodos que preservam
estrutura discriminativa (VICReg, k-Means+Sil, MCFS) mantêm performance,
enquanto métodos de ranking simples (Variance, Laplacian) degradam mais.

Cada método de ranking (Variance, Laplacian, MCFS, SPEC) é rankeado uma
única vez; depois fatiamos os top-k para cada valor de k avaliado.
Os métodos bidirecionais (ContrastiveBidirectionalFS, BidirectionalSilhouetteFS)
também rodam uma vez e seus índices são fatiados por k.

Uso:
    python experiments/08_run_k_sweep.py
    python experiments/08_run_k_sweep.py --csv data/dataset_attack_benign.csv
"""

import sys
import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BENIGN_CSV, FULL_CSV, TABLES_DIR, FIGURES_DIR,
    CONTEXT_GROUPS, MODELS_DIR,
)
from src.preprocessing import DataSensePreprocessor
from src.contrastive_fs import (
    ContrastiveBidirectionalFS,
    load_encoder,
    train_contrastive_encoder,
)
from src.baselines import (
    BidirectionalSilhouetteFS,
    VarianceFS,
    LaplacianScoreFS,
    SPECFS,
    MCFSFS,
)
from src.evaluation import EvaluationPipeline

K_VALUES = [10, 15, 20, 25]

ENCODER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "models", "encoder_datasense_vicreg.pt",
)
# Fallback: encoder salvo pelo exp 02 (pode ser NT-Xent ou VICReg)
ENCODER_PATH_FALLBACK = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "models", "encoder_datasense.pt",
)


def _load_or_train_encoder(X, feature_names):
    for path in [ENCODER_PATH, ENCODER_PATH_FALLBACK]:
        if os.path.exists(path):
            print(f"[K-Sweep] Carregando encoder: {path}")
            enc, _ = load_encoder(path)
            return enc
    print("[K-Sweep] Encoder não encontrado — treinando VICReg agora...")
    enc, _ = train_contrastive_encoder(
        X, feature_names, CONTEXT_GROUPS,
        save_path=ENCODER_PATH, verbose=True,
    )
    return enc


def _variance_rank(X: np.ndarray, indices: list[int]) -> list[int]:
    """Ordena `indices` do mais variável ao menos variável em X."""
    variances = X[:, indices].var(axis=0)
    order = np.argsort(variances)[::-1]
    return [indices[i] for i in order]


def _load_bidirectional_result(json_path: str, feature_names: list[str]) -> list[int] | None:
    """Carrega selected_features de um JSON salvo e retorna índices."""
    if not os.path.exists(json_path):
        return None
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    with open(json_path, encoding="utf-8") as f:
        d = json.load(f)
    names = d.get("selected_features", [])
    return [name_to_idx[n] for n in names if n in name_to_idx]


def _run_all_methods(X, feature_names):
    """
    Retorna um ranking completo de features por método.
    - Métodos de ranking: lista completa do melhor ao pior
    - Métodos bidirecionais: carrega resultado salvo do JSON, rankeando
      internamente por variância (para ser justo no fatiamento por k)
    """
    n_feat = len(feature_names)
    rankings = {}

    # 1. Ranking simples: VarianceFS
    print("[K-Sweep] Variância...")
    t0 = time.time()
    vfs = VarianceFS()
    vfs.fit(X)
    rankings["variance"] = vfs.select_top_k(n_feat)
    print(f"  -> {time.time()-t0:.1f}s")

    # 2. Ranking: LaplacianScore
    print("[K-Sweep] Laplacian Score...")
    t0 = time.time()
    lfs = LaplacianScoreFS(sample_size=5000)
    try:
        lfs.fit(X)
        rankings["laplacian"] = lfs.select_top_k(n_feat)
    except Exception as e:
        print(f"  -> ERRO: {e}")
    print(f"  -> {time.time()-t0:.1f}s")

    # 3. Ranking: SPEC
    print("[K-Sweep] SPEC...")
    t0 = time.time()
    sfs = SPECFS()
    try:
        sfs.fit(X)
        rankings["spec"] = sfs.select_top_k(n_feat)
    except Exception as e:
        print(f"  -> ERRO: {e}")
    print(f"  -> {time.time()-t0:.1f}s")

    # 4. Ranking: MCFS (k=8 clusters para label2)
    print("[K-Sweep] MCFS...")
    t0 = time.time()
    mfs = MCFSFS(n_clusters=8, sample_size=5000)
    try:
        mfs.fit(X)
        rankings["mcfs"] = mfs.select_top_k(n_feat)
    except Exception as e:
        print(f"  -> ERRO: {e}")
    print(f"  -> {time.time()-t0:.1f}s")

    # 5. k-Means+Sil bidirecional — carrega JSON salvo (evita 1h de busca)
    km_json = os.path.join(TABLES_DIR, "baseline_original_result.json")
    km_cached = _load_bidirectional_result(km_json, feature_names)
    if km_cached is not None:
        # Rankear as features selecionadas por variância para fatiamento justo
        ranked_km = _variance_rank(X, km_cached)
        # Completar com features não selecionadas (para k > natural_k)
        remaining = [i for i in rankings["variance"] if i not in set(km_cached)]
        rankings["kmeans_sil"] = ranked_km + remaining
        print(f"[K-Sweep] k-Means+Sil: {len(km_cached)} features carregadas do JSON")
    else:
        print("[K-Sweep] k-Means+Sil bidirecional (rodando do zero)...")
        t0 = time.time()
        kmfs = BidirectionalSilhouetteFS(verbose=False)
        kmfs.fit(X)
        ranked_km = _variance_rank(X, kmfs.selected_indices_)
        remaining = [i for i in rankings["variance"] if i not in set(kmfs.selected_indices_)]
        rankings["kmeans_sil"] = ranked_km + remaining
        print(f"  -> {len(kmfs.selected_indices_)} features | {time.time()-t0:.1f}s")

    # 6. VICReg — carrega JSON salvo (evita 2.5h de busca)
    vicreg_json = os.path.join(TABLES_DIR, "contrastive_result.json")
    vicreg_cached = _load_bidirectional_result(vicreg_json, feature_names)
    if vicreg_cached is not None:
        ranked_vicreg = _variance_rank(X, vicreg_cached)
        remaining = [i for i in rankings["variance"] if i not in set(vicreg_cached)]
        rankings["vicreg"] = ranked_vicreg + remaining
        print(f"[K-Sweep] VICReg: {len(vicreg_cached)} features carregadas do JSON")
    else:
        print("[K-Sweep] ContrastiveBidirectionalFS (rodando do zero)...")
        t0 = time.time()
        encoder = _load_or_train_encoder(X, feature_names)
        cfs = ContrastiveBidirectionalFS(encoder=encoder, verbose=False)
        cfs.fit(X)
        ranked_vicreg = _variance_rank(X, cfs.selected_indices_)
        remaining = [i for i in rankings["variance"] if i not in set(cfs.selected_indices_)]
        rankings["vicreg"] = ranked_vicreg + remaining
        print(f"  -> {len(cfs.selected_indices_)} features | {time.time()-t0:.1f}s")

    return rankings


def _build_feature_sets_for_k(rankings, k, n_feat):
    """
    Para um k dado, constrói o dict feature_sets esperado por EvaluationPipeline.
    Para métodos de ranking: top-k direto.
    Para métodos bidirecionais: primeiros k do conjunto selecionado (ou tudo se <k).
    Inclui baseline "all_features".
    """
    feature_sets = {}

    for method, indices in rankings.items():
        if len(indices) >= k:
            feature_sets[f"{method}_k{k}"] = list(indices[:k])
        else:
            # Seleção natural < k: usar tudo (marcar no nome)
            feature_sets[f"{method}_k{k}"] = list(indices)

    feature_sets["all_features"] = []  # lista vazia = todas

    return feature_sets


def run(csv_path):
    print(f"\n{'='*60}")
    print(f"EXPERIMENTO 08 — K Sweep (k = {K_VALUES})")
    print(f"{'='*60}")
    print(f"Dataset: {csv_path}\n")

    pp = DataSensePreprocessor(csv_path)
    X, y = pp.load()
    feature_names = pp.get_feature_names()
    n_feat = len(feature_names)
    print(f"Dataset: {X.shape[0]:,} × {X.shape[1]}")

    label_cols = [c for c in ["label1", "label2"] if c in y.columns and y[c].nunique() >= 2]
    if not label_cols:
        print("Nenhuma label_col válida encontrada. Abortando.")
        return

    # Rodar cada método UMA vez
    print(f"\n--- Computando rankings de features (1 vez) ---")
    rankings = _run_all_methods(X, feature_names)

    # Avaliar para cada k
    pipe = EvaluationPipeline(include_knn=False)
    all_results = []

    for k in K_VALUES:
        print(f"\n{'-'*50}")
        print(f"K = {k} features")
        print(f"{'-'*50}")

        feature_sets = _build_feature_sets_for_k(rankings, k, n_feat)

        for label_col in label_cols:
            df = pipe.run_all(X, y, feature_sets, label_col=label_col, verbose=True)
            if not df.empty:
                df["k"] = k
                df["label_col"] = label_col
                all_results.append(df)

    if not all_results:
        print("Nenhum resultado gerado.")
        return

    results_df = pd.concat(all_results, ignore_index=True)

    os.makedirs(TABLES_DIR, exist_ok=True)
    out_csv = os.path.join(TABLES_DIR, "k_sweep_results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\n[K-Sweep] Resultados salvos: {out_csv}")
    print(results_df[results_df["classifier"] == "random_forest"][
        ["method", "k", "label_col", "n_features", "f1_macro_mean"]
    ].to_string(index=False))

    for label_col in label_cols:
        _plot_k_sweep(results_df, label_col=label_col, output_dir=FIGURES_DIR)


def _plot_k_sweep(df, label_col, output_dir):
    df2 = df[df["label_col"] == label_col].copy()
    df_rf = df2[df2["classifier"] == "random_forest"].copy()
    if df_rf.empty:
        return

    # Extrair nome base (remover _k{k} do sufixo)
    import re
    df_rf["method_base"] = df_rf["method"].str.replace(r"_k\d+$", "", regex=True)

    method_styles = {
        "vicreg":      ("#185FA5", "o", "-",  2.5),
        "kmeans_sil":  ("#1D9E75", "s", "--", 1.8),
        "mcfs":        ("#BA7517", "^", "--", 1.8),
        "spec":        ("#6A2585", "D", "--", 1.5),
        "variance":    ("#A32D2D", "x", ":",  1.5),
        "laplacian":   ("#534AB7", "P", ":",  1.5),
        "all_features":("#888780", ".", "-",  1.2),
    }

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    for base, group in df_rf.groupby("method_base"):
        g = group.sort_values("k")
        color, marker, ls, lw = method_styles.get(base, ("#888780", "o", "-", 1.2))
        ax.plot(
            g["k"], g["f1_macro_mean"],
            marker=marker, linestyle=ls, linewidth=lw,
            color=color, label=base, markersize=7,
        )
        ax.fill_between(
            g["k"],
            g["f1_macro_mean"] - g["f1_macro_std"],
            g["f1_macro_mean"] + g["f1_macro_std"],
            alpha=0.12, color=color,
        )

    ax.set_xlabel("Número de features selecionadas (k)", fontsize=12)
    ax.set_ylabel("F1-macro (Random Forest, 5-fold CV)", fontsize=12)
    ax.set_title(f"Degradação por k — {label_col}", fontsize=13)
    ax.set_xticks(K_VALUES)
    ax.legend(title="Método", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_ylim(0.6, 1.02)
    plt.tight_layout()

    out = os.path.join(output_dir, f"k_sweep_{label_col}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[K-Sweep] Gráfico salvo: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()
    csv_path = args.csv or (FULL_CSV if os.path.exists(FULL_CSV) else BENIGN_CSV)
    run(csv_path)
