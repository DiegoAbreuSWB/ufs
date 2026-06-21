"""
05_run_ablation.py — Ablation Study: contribuicao de cada componente do metodo proposto

Isola o impacto de:
  (1) Encoder contrastivo vs. espaco original de features
  (2) Busca bidirecional vs. metodo ranking estatico
  (3) Combinacao completa (proposta) vs. baselines individuais

Configuracoes avaliadas:
  A — all_features          : 71 features sem selecao (trivial baseline)
  B — variance_top_k        : ranking estatico (variancia) — sem encoder
  C — laplacian_score_top_k : melhor baseline grafico classico
  D — kmeans_sil_raw        : bidirecional no espaco original (dissertacao 2022)
  E — contrastive_proposed  : encoder + bidirecional no espaco latente (proposta)

Uso:
    python experiments/05_run_ablation.py
    python experiments/05_run_ablation.py --label label1   # so cenario binario
"""

import sys
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BENIGN_CSV, FULL_CSV, TABLES_DIR, FIGURES_DIR
from src.preprocessing import DataSensePreprocessor
from src.evaluation import EvaluationPipeline


# =============================================================================
# Carregar configuracoes de ablation
# =============================================================================

def load_ablation_configs(feature_names: list[str], tables_dir: str) -> dict[str, list[int] | None]:
    """
    Monta os conjuntos de features para cada configuracao de ablation.
    Retorna dict: config_name -> list[int] de indices.
    """
    name_to_idx = {n: i for i, n in enumerate(feature_names)}

    def _resolve(names: list[str]) -> list[int]:
        return [name_to_idx[n] for n in names if n in name_to_idx]

    configs: dict[str, list[int] | None] = {}

    # A — Todas as features
    configs["A_all_features"] = list(range(len(feature_names)))

    # B — Variance ranking (top-k)
    p = os.path.join(tables_dir, "baselines_results.json")
    k_ref = 25
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        k_ref = d.get("k_contrastive", 25)
        for method in ["variance", "laplacian_score", "spec", "mcfs", "ndfs"]:
            if method in d["baselines"] and d["baselines"][method].get("status") == "ok":
                feats = d["baselines"][method].get("selected_features", [])
                configs[f"B_{method}"] = _resolve(feats)

    # C — k-Means Silhouette bidirecional no espaco original
    p2 = os.path.join(tables_dir, "baseline_original_result.json")
    if os.path.exists(p2):
        with open(p2, encoding="utf-8") as f:
            d2 = json.load(f)
        feats_raw = d2.get("selected_features", [])
        configs["C_kmeans_sil_raw"] = _resolve(feats_raw)
        print(f"[Ablation] Config C (k-Means Sil raw): {len(configs['C_kmeans_sil_raw'])} features "
              f"(csv: {d2.get('csv', '?')})")
    else:
        print(f"[Ablation] baseline_original_result.json nao encontrado — Config C indisponivel")

    # D — Metodo proposto (contrastivo + bidirecional no espaco latente)
    p3 = os.path.join(tables_dir, "contrastive_result.json")
    if os.path.exists(p3):
        with open(p3, encoding="utf-8") as f:
            d3 = json.load(f)
        feats_contra = d3.get("selected_features", [])
        configs["D_contrastive_proposed"] = _resolve(feats_contra)
        print(f"[Ablation] Config D (Contrastive proposed): {len(configs['D_contrastive_proposed'])} features")
    else:
        print(f"[Ablation] contrastive_result.json nao encontrado — Config D indisponivel")

    return configs, k_ref


# =============================================================================
# Tabela de ablation formatada
# =============================================================================

def print_ablation_table(df: pd.DataFrame, label_col: str) -> None:
    if df.empty:
        return
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
        .reset_index()
    )

    print(f"\n{'='*80}")
    print(f"ABLATION STUDY — {label_col}")
    print(f"{'='*80}")
    print(f"{'Config':<32} {'#Feat':>6}  {'F1-macro':>12}  {'Accuracy':>10}  {'MCC':>8}")
    print(f"{'-'*32} {'-'*6}  {'-'*12}  {'-'*10}  {'-'*8}")
    for _, row in summary.iterrows():
        print(
            f"  {row['method']:<30} {int(row.n_features):>6}  "
            f"{row.f1_mean:.3f}+-{row.f1_std:.3f}  "
            f"{row.acc_mean:.3f}       "
            f"{row.mcc_mean:.3f}"
        )

    # Destacar ganho relativo do contrastivo vs. melhor baseline
    methods = summary["method"].tolist()
    baseline_rows = [r for r in methods if "contrastive" not in r and "all_features" not in r]
    contrastive_rows = [r for r in methods if "contrastive" in r]

    if contrastive_rows and baseline_rows:
        best_baseline_f1 = summary[summary["method"].isin(baseline_rows)]["f1_mean"].max()
        contrastive_f1 = summary[summary["method"] == contrastive_rows[0]]["f1_mean"].values
        if len(contrastive_f1):
            delta = contrastive_f1[0] - best_baseline_f1
            print(f"\n  Ganho do metodo proposto vs. melhor baseline: {delta:+.3f} F1-macro")


# =============================================================================
# Plot radar-chart ou grouped bar
# =============================================================================

def plot_ablation_bar(df: pd.DataFrame, label_col: str, output_dir: str) -> None:
    if df.empty:
        return
    os.makedirs(output_dir, exist_ok=True)

    pivot = df.pivot_table(
        index="method", columns="classifier",
        values="f1_macro_mean", aggfunc="mean"
    )
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    pivot_std = df.pivot_table(
        index="method", columns="classifier",
        values="f1_macro_std", aggfunc="mean"
    ).loc[pivot.index]

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(pivot))
    n_clf = len(pivot.columns)
    width = 0.8 / n_clf
    colors = plt.cm.tab10.colors

    for i, clf in enumerate(pivot.columns):
        offset = (i - n_clf / 2 + 0.5) * width
        ax.bar(
            x + offset, pivot[clf], width,
            yerr=pivot_std[clf], capsize=3,
            label=clf, color=colors[i], alpha=0.85,
        )

    # Destacar a melhor configuracao (contrastivo)
    for xi, method in enumerate(pivot.index):
        if "contrastive" in method:
            ax.axvline(xi, color="red", linewidth=1.5, linestyle="--", alpha=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("F1-macro (media +- std, 5-fold CV)")
    ax.set_title(f"Ablation Study — Contribuicao dos Componentes ({label_col})")
    ax.legend(title="Classificador", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = os.path.join(output_dir, f"ablation_barplot_{label_col}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Ablation] Grafico salvo: {out}")


def plot_component_delta(all_results: dict[str, pd.DataFrame], output_dir: str) -> None:
    """
    Grafico de delta F1 mostrando contribuicao incremental de cada componente:
    all_features -> melhor_baseline -> kmeans_sil_raw -> contrastive_proposed
    """
    os.makedirs(output_dir, exist_ok=True)

    for label_col, df in all_results.items():
        if df.empty:
            continue
        summary = (
            df.groupby("method")["f1_macro_mean"].mean()
            .sort_values(ascending=False)
        )

        # Identificar os pontos-chave do ablation
        steps = []
        step_labels = []
        for method in ["A_all_features", "C_kmeans_sil_raw"]:
            if method in summary.index:
                steps.append((method, float(summary[method])))

        # Melhor baseline estatico (B_*)
        b_methods = [m for m in summary.index if m.startswith("B_")]
        if b_methods:
            best_b = max(b_methods, key=lambda m: float(summary[m]))
            steps.insert(1, (best_b, float(summary[best_b])))

        # Proposta
        d_methods = [m for m in summary.index if m.startswith("D_")]
        if d_methods:
            steps.append((d_methods[0], float(summary[d_methods[0]])))

        if len(steps) < 2:
            continue

        names = [s[0] for s in steps]
        vals = [s[1] for s in steps]
        deltas = [0.0] + [vals[i] - vals[i-1] for i in range(1, len(vals))]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Barras absolutas
        colors_abs = ["#aec6cf"] * len(vals)
        if d_methods:
            colors_abs[-1] = "#ff6b6b"
        ax1.barh(range(len(names)), vals, color=colors_abs, edgecolor="gray")
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels(names, fontsize=8)
        ax1.set_xlabel("F1-macro medio")
        ax1.set_title(f"F1-macro por Configuracao ({label_col})")
        ax1.set_xlim(max(0, min(vals) - 0.05), 1.0)
        ax1.axvline(vals[0], color="gray", linestyle="--", alpha=0.5, label="all_features")
        ax1.grid(axis="x", alpha=0.3)

        # Deltas
        delta_colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]
        ax2.barh(range(len(names)), deltas, color=delta_colors, edgecolor="gray")
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names, fontsize=8)
        ax2.set_xlabel("Delta F1-macro vs. passo anterior")
        ax2.set_title(f"Ganho Incremental por Componente ({label_col})")
        ax2.axvline(0, color="black", linewidth=0.8)
        ax2.grid(axis="x", alpha=0.3)
        for i, d in enumerate(deltas):
            ax2.text(d, i, f"  {d:+.3f}", va="center", fontsize=8)

        plt.suptitle("Ablation Study — ContrastiveXAI-FS", fontsize=12, y=1.02)
        plt.tight_layout()
        out = os.path.join(output_dir, f"ablation_delta_{label_col}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Ablation] Delta chart salvo: {out}")


# =============================================================================
# Main
# =============================================================================

def run(csv_path: str, label_cols: list[str] | None = None) -> None:
    print(f"\n{'='*60}")
    print(f"EXPERIMENTO 05 — Ablation Study")
    print(f"{'='*60}")
    print(f"Dataset: {csv_path}\n")

    pp = DataSensePreprocessor(csv_path)
    X, y = pp.load()
    feature_names = pp.get_feature_names()
    print(f"Dataset: {X.shape[0]:,} x {X.shape[1]}")

    # Detectar label cols
    if label_cols is None:
        available = []
        for col in ["label1", "label2"]:
            if col in y.columns and y[col].nunique() >= 2:
                available.append(col)
                print(f"  label '{col}': {y[col].nunique()} classes")
        label_cols = available[:2]

    if not label_cols:
        print("Nenhum label com >= 2 classes. Encerrando.")
        return

    # Carregar configuracoes
    ablation_configs, k_ref = load_ablation_configs(feature_names, TABLES_DIR)
    print(f"\nConfiguracoes de ablation ({len(ablation_configs)} total):")
    for name, indices in ablation_configs.items():
        if indices is not None:
            print(f"  {name}: {len(indices)} features")
        else:
            print(f"  {name}: N/A (PCA)")

    # Avaliar cada cenario
    include_knn = X.shape[0] <= 50000
    if not include_knn:
        print(f"[Ablation] KNN desabilitado para dataset grande ({X.shape[0]:,} amostras)")
    pipe = EvaluationPipeline(include_knn=include_knn)
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    all_results: dict[str, pd.DataFrame] = {}

    for label_col in label_cols:
        df = pipe.run_all(X, y, ablation_configs, label_col=label_col, verbose=True)
        if df.empty:
            continue

        # Salvar
        out_csv = os.path.join(TABLES_DIR, f"ablation_{label_col}.csv")
        df.to_csv(out_csv, index=False)
        print(f"\n[Ablation] Resultados salvos: {out_csv}")

        print_ablation_table(df, label_col)
        plot_ablation_bar(df, label_col, FIGURES_DIR)
        all_results[label_col] = df

    # Plot de deltas (ganho incremental)
    if all_results:
        plot_component_delta(all_results, FIGURES_DIR)

    print(f"\n{'='*60}")
    print(f"Ablation concluido.")
    print(f"Tabelas: {TABLES_DIR}")
    print(f"Figuras : {FIGURES_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None)
    parser.add_argument("--label", nargs="+", default=None)
    args = parser.parse_args()

    if args.csv:
        csv_path = args.csv
    elif os.path.exists(FULL_CSV):
        csv_path = FULL_CSV
    else:
        csv_path = BENIGN_CSV

    run(csv_path, label_cols=args.label)
