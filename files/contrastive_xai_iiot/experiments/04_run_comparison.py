"""
04_run_comparison.py — Tabela comparativa: todos os métodos × classificadores × cenários

Carrega feature sets pré-computados (baselines_results.json + contrastive_result.json +
baseline_original_result.json) e avalia cada conjunto com 5-fold Stratified CV.

Com dados benign-only usa device_name (38 dispositivos) como label proxy para validação.
Com dataset completo usa label1 (binary) e label2 (8-class).

Uso:
    python experiments/04_run_comparison.py
    python experiments/04_run_comparison.py --csv data/all_attack_samples.csv
    python experiments/04_run_comparison.py --label device_name   # forçar label proxy
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
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BENIGN_CSV, FULL_CSV, TABLES_DIR, FIGURES_DIR
from src.preprocessing import DataSensePreprocessor
from src.evaluation import EvaluationPipeline


def load_feature_sets(feature_names: list[str], tables_dir: str) -> dict[str, list[int] | None]:
    """
    Carrega índices de features de cada método a partir dos JSONs salvos.
    Retorna dict: method_name -> list[int] de índices em feature_names.
    """
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    sets: dict[str, list[int] | None] = {}

    def _resolve(names: list[str]) -> list[int]:
        return [name_to_idx[n] for n in names if n in name_to_idx]

    # Método proposto (contrastivo)
    p = os.path.join(tables_dir, "contrastive_result.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        sets["contrastive_proposed"] = _resolve(d["selected_features"])

    # Baseline dissertação (k-Means + Silhouette)
    p = os.path.join(tables_dir, "baseline_original_result.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        sets["kmeans_silhouette"] = _resolve(d["selected_features"])

    # Baselines UFS
    p = os.path.join(tables_dir, "baselines_results.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        for name, res in d["baselines"].items():
            if res.get("status") == "ok" and "selected_features" in res:
                sets[name] = _resolve(res["selected_features"])
            elif name == "pca":
                sets["pca"] = None  # PCA transforma, não seleciona

    # All features (baseline trivial)
    sets["all_features"] = list(range(len(feature_names)))

    return sets


def plot_comparison(df: pd.DataFrame, label_col: str, output_dir: str) -> None:
    """Gráfico de barras: F1-macro por método × classificador."""
    if df.empty:
        return
    os.makedirs(output_dir, exist_ok=True)

    pivot = df.pivot_table(
        index="method", columns="classifier",
        values="f1_macro_mean", aggfunc="mean"
    )
    pivot_std = df.pivot_table(
        index="method", columns="classifier",
        values="f1_macro_std", aggfunc="mean"
    )

    # Ordenar por melhor F1 médio entre classificadores
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    pivot_std = pivot_std.loc[pivot.index]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(pivot))
    n_clf = len(pivot.columns)
    width = 0.8 / n_clf
    colors = plt.cm.tab10.colors

    for i, clf in enumerate(pivot.columns):
        offset = (i - n_clf / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, pivot[clf], width,
            yerr=pivot_std[clf], capsize=3,
            label=clf, color=colors[i], alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("F1-macro (média ± std, 5-fold CV)")
    ax.set_title(f"Comparação de Métodos UFS — {label_col}")
    ax.legend(title="Classificador", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_dir, f"comparison_barplot_{label_col}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Comparison] Gráfico salvo: {out}")


def print_comparison_table(df: pd.DataFrame, label_col: str) -> None:
    """Imprime tabela comparativa resumida no terminal."""
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
    )
    print(f"\n{'='*75}")
    print(f"TABELA COMPARATIVA — {label_col}")
    print(f"{'='*75}")
    print(f"{'Método':<28} {'#Feat':>6}  {'F1-macro':>10}  {'Accuracy':>10}  {'MCC':>8}")
    print(f"{'-'*28} {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}")
    for method, row in summary.iterrows():
        print(
            f"  {method:<26} {int(row.n_features):>6}  "
            f"{row.f1_mean:.3f}±{row.f1_std:.3f}  "
            f"{row.acc_mean:.3f}        "
            f"{row.mcc_mean:.3f}"
        )


def run(csv_path: str, label_cols: list[str] | None = None) -> None:
    print(f"\n{'='*60}")
    print(f"EXPERIMENTO 04 — Tabela Comparativa (5-fold CV)")
    print(f"{'='*60}")
    print(f"Dataset: {csv_path}\n")

    # ------------------------------------------------------------------
    # 1. Pré-processamento
    # ------------------------------------------------------------------
    pp = DataSensePreprocessor(csv_path)
    X, y = pp.load()
    feature_names = pp.get_feature_names()
    print(f"Dataset: {X.shape[0]:,} x {X.shape[1]}")

    # Determinar quais label cols usar
    if label_cols is None:
        # Auto-detect: preferir label1/label2; se benign-only usar device_name
        available = []
        for col in ["label1", "label2", "device_name"]:
            if col in y.columns:
                n_cls = y[col].nunique()
                if n_cls >= 2:
                    available.append(col)
                    print(f"  label '{col}': {n_cls} classes")
        label_cols = available[:2]  # max 2 cenários por rodada

    if not label_cols:
        print("Nenhum label com >= 2 classes encontrado. Encerrando.")
        return

    # ------------------------------------------------------------------
    # 2. Carregar feature sets
    # ------------------------------------------------------------------
    feature_sets = load_feature_sets(feature_names, TABLES_DIR)
    print(f"\nMétodos carregados: {list(feature_sets.keys())}")

    # ------------------------------------------------------------------
    # 3. Avaliar cada cenário
    # ------------------------------------------------------------------
    # KNN é O(n²) na predição — desabilitar para datasets grandes (>50k amostras)
    include_knn = X.shape[0] <= 50000
    if not include_knn:
        print(f"[Eval] KNN desabilitado para dataset grande ({X.shape[0]:,} amostras)")
    pipe = EvaluationPipeline(include_knn=include_knn)
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    for label_col in label_cols:
        df = pipe.run_all(X, y, feature_sets, label_col=label_col, verbose=True)
        if df.empty:
            continue

        # Salvar CSV
        out_csv = os.path.join(TABLES_DIR, f"comparison_{label_col}.csv")
        df.to_csv(out_csv, index=False)
        print(f"\n[Comparison] Resultados salvos: {out_csv}")

        # Tabela resumida
        print_comparison_table(df, label_col)

        # Gráfico
        plot_comparison(df, label_col, FIGURES_DIR)

    print(f"\n{'='*60}")
    print(f"Avaliação concluída.")
    print(f"Tabelas: {TABLES_DIR}")
    print(f"Figuras : {FIGURES_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None)
    parser.add_argument(
        "--label", nargs="+", default=None,
        help="Colunas de label a usar (ex: --label label1 label2)"
    )
    args = parser.parse_args()

    if args.csv:
        csv_path = args.csv
    elif os.path.exists(FULL_CSV):
        csv_path = FULL_CSV
    else:
        csv_path = BENIGN_CSV

    run(csv_path, label_cols=args.label)
