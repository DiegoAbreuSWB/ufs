"""
07_final_paper_table.py — Gera tabela final no formato do paper DataSense (Table 8 equivalent)

Combina resultados de classificacao (Exp 04) com metricas de selecao de features
para gerar a tabela comparativa completa do paper.

Uso:
    python experiments/07_final_paper_table.py
"""

import sys, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TABLES_DIR, FIGURES_DIR


# Metadados de selecao de features
SELECTION_META = {
    "contrastive_proposed": {
        "label": "ContrastiveXAI-FS (ours)",
        "n_features": 25,
        "reduction": 0.648,
        "silhouette": 0.9711,
        "supervised": False,
    },
    "kmeans_silhouette": {
        "label": "k-Means+Sil Bidirecional",
        "n_features": 46,
        "reduction": 0.352,
        "silhouette": 0.7360,
        "supervised": False,
    },
    "variance": {"label": "Variance Ranking", "n_features": 25, "reduction": 0.648, "silhouette": None, "supervised": False},
    "laplacian_score": {"label": "Laplacian Score", "n_features": 25, "reduction": 0.648, "silhouette": None, "supervised": False},
    "spec": {"label": "SPEC", "n_features": 25, "reduction": 0.648, "silhouette": None, "supervised": False},
    "mcfs": {"label": "MCFS", "n_features": 25, "reduction": 0.648, "silhouette": None, "supervised": False},
    "udfs": {"label": "UDFS", "n_features": 25, "reduction": 0.648, "silhouette": None, "supervised": False},
    "ndfs": {"label": "NDFS", "n_features": 25, "reduction": 0.648, "silhouette": None, "supervised": False},
    "all_features": {"label": "All Features (baseline)", "n_features": 71, "reduction": 0.0, "silhouette": None, "supervised": False},
    "datasense_17_supervised": {"label": "DataSense-17 (supervised ref)", "n_features": 17, "reduction": 0.760, "silhouette": None, "supervised": True},
}


def load_results(tables_dir: str) -> dict[str, pd.DataFrame]:
    """Load all comparison CSVs available."""
    results = {}
    for label_col in ["label1", "label2"]:
        for fname in [f"comparison_{label_col}.csv", f"datasense17_vs_contrastive_{label_col}.csv",
                      f"ablation_{label_col}.csv"]:
            p = os.path.join(tables_dir, fname)
            if os.path.exists(p):
                df = pd.read_csv(p)
                existing = results.get(label_col, pd.DataFrame())
                results[label_col] = pd.concat([existing, df], ignore_index=True)
    return results


def best_per_method(df: pd.DataFrame) -> pd.DataFrame:
    """Best classifier F1-macro per method (max across classifiers)."""
    return (
        df.groupby("method")
        .agg(
            n_features=("n_features", "first"),
            f1_mean=("f1_macro_mean", "max"),
            f1_std=("f1_macro_std", "first"),
            acc_mean=("accuracy_mean", "max"),
            mcc_mean=("mcc_mean", "max"),
            best_clf=("classifier", lambda x: x[df.loc[x.index, "f1_macro_mean"].idxmax()]),
        )
        .reset_index()
    )


def mean_across_classifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Mean F1-macro per method (averaged across classifiers)."""
    return (
        df.groupby("method")
        .agg(
            n_features=("n_features", "first"),
            f1_mean=("f1_macro_mean", "mean"),
            f1_std=("f1_macro_std", "mean"),
            acc_mean=("accuracy_mean", "mean"),
            mcc_mean=("mcc_mean", "mean"),
        )
        .reset_index()
    )


def format_paper_table(results: dict[str, pd.DataFrame]) -> None:
    """Print final comparison table in paper format."""
    print("\n" + "="*90)
    print("TABELA COMPARATIVA FINAL — ContrastiveXAI-FS vs Baselines UFS")
    print("Metrica: F1-macro (media nos classificadores, 5-fold Stratified CV)")
    print("="*90)

    label_names = {"label1": "Binary (benign/attack)", "label2": "8-class (attack types)"}

    all_dfs = []
    for label_col, df in results.items():
        if df.empty:
            continue
        summary = mean_across_classifiers(df)
        summary["scenario"] = label_col
        all_dfs.append(summary)

    if not all_dfs:
        print("Nenhum resultado encontrado.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)

    # Ordenar metodos (metodo proposto primeiro, depois baseline alphabetically)
    method_order = [
        "contrastive_proposed",
        "kmeans_silhouette",
        "variance",
        "laplacian_score",
        "spec",
        "mcfs",
        "udfs",
        "ndfs",
        "all_features",
        "datasense_17_supervised",
    ]

    scenarios = list(results.keys())
    pivot = combined.pivot_table(
        index="method", columns="scenario", values="f1_mean", aggfunc="first"
    )
    pivot_std = combined.pivot_table(
        index="method", columns="scenario", values="f1_std", aggfunc="first"
    )
    pivot_n = combined.pivot_table(
        index="method", columns="scenario", values="n_features", aggfunc="first"
    )
    pivot_mcc = combined.pivot_table(
        index="method", columns="scenario", values="mcc_mean", aggfunc="first"
    )

    # Header
    header = f"  {'Metodo':<36} {'#Feat':>5}"
    for sc in scenarios:
        header += f"  {'F1('+label_names.get(sc, sc)[:12]+')':>18}"
    header += f"  {'MCC(bin)':>10}"
    print(header)
    print("-"*90)

    for method in method_order:
        if method not in pivot.index:
            continue
        meta = SELECTION_META.get(method, {})
        label = meta.get("label", method)
        sup = "*" if meta.get("supervised") else " "
        n_feat = int(pivot_n.loc[method, scenarios[0]]) if scenarios[0] in pivot_n.columns else "?"
        row = f"  {sup}{label:<35} {n_feat:>5}"
        for sc in scenarios:
            if sc in pivot.columns and method in pivot.index:
                f1 = pivot.loc[method, sc]
                std = pivot_std.loc[method, sc]
                row += f"  {f1:.3f}+-{std:.3f}   "
            else:
                row += f"  {'n/a':>18}"
        if "label1" in pivot_mcc.columns and method in pivot_mcc.index:
            row += f"  {pivot_mcc.loc[method, 'label1']:>10.3f}"
        print(row)

    print("="*90)
    print("* = metodo supervisionado (usa labels durante selecao) — referencia apenas")

    # Salvar
    out = os.path.join(TABLES_DIR, "final_paper_table.csv")
    combined.to_csv(out, index=False)
    print(f"\nTabela completa salva em: {out}")


def plot_final_comparison(results: dict[str, pd.DataFrame]) -> None:
    """Radar chart comparing methods across scenarios."""
    if "label1" not in results or results["label1"].empty:
        return

    df = results["label1"]
    pivot = df.pivot_table(index="method", columns="classifier", values="f1_macro_mean", aggfunc="mean")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(pivot))
    n_clf = len(pivot.columns)
    width = 0.8 / n_clf
    colors = plt.cm.Set2.colors

    for i, clf in enumerate(pivot.columns):
        offset = (i - n_clf / 2 + 0.5) * width
        ax.bar(x + offset, pivot[clf], width, label=clf, color=colors[i % len(colors)], alpha=0.85)

    ax.set_xticks(x)
    labels = [SELECTION_META.get(m, {}).get("label", m)[:18] for m in pivot.index]
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("F1-macro (5-fold CV)")
    ax.set_title("ContrastiveXAI-FS vs Baselines UFS — Binary Classification (benign vs attack)")
    ax.legend(title="Classificador", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = os.path.join(FIGURES_DIR, "final_comparison_label1.png")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grafico salvo: {out}")


def run() -> None:
    print(f"\n{'='*60}")
    print("EXPERIMENTO 07 — Tabela Final para o Paper")
    print(f"{'='*60}")

    results = load_results(TABLES_DIR)
    if not results:
        print("Nenhum resultado encontrado. Execute Exp 04 primeiro.")
        return

    print(f"Cenarios encontrados: {list(results.keys())}")
    for sc, df in results.items():
        print(f"  {sc}: {len(df)} linhas | metodos: {sorted(df['method'].unique())}")

    format_paper_table(results)
    plot_final_comparison(results)


if __name__ == "__main__":
    run()
