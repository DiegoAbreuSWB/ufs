"""
09_generate_paper_figures.py — Figuras finais formatadas para o paper

Figuras geradas:
  fig2_tsne_comparison.png  — t-SNE: encoder anterior vs VICReg (se disponível)
  fig3_comparison_heatmap.{pdf,png} — heatmap comparativo label1 + label2
  fig4_k_sweep.{pdf,png}    — lineplot F1 por k (do experimento 08)
  fig5_shap_summary.png     — SHAP summary reaproveitado com estilo paper

Requer:
  - results/tables/final_paper_table.csv  (exp 07)
  - results/tables/k_sweep_results.csv    (exp 08)
  - results/figures/shap_summary.png      (exp 06, se existir)

Uso:
    python experiments/09_generate_paper_figures.py
"""

import sys
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TABLES_DIR, FIGURES_DIR

# Diretório de saída exclusivo para o paper
PAPER_FIGS_DIR = os.path.join(FIGURES_DIR, "paper_ready")

# Paleta de cores consistente com 08_run_k_sweep
METHOD_STYLES = {
    "vicreg":           ("#185FA5", "o", "-",  2.5, "ContrastiveXAI-FS (ours)"),
    "kmeans_sil":       ("#1D9E75", "s", "--", 1.8, "k-Means+Sil (Abreu 2022)"),
    "mcfs":             ("#BA7517", "^", "--", 1.8, "MCFS"),
    "spec":             ("#6A2585", "D", "--", 1.5, "SPEC"),
    "variance":         ("#A32D2D", "x", ":",  1.5, "Variance"),
    "laplacian":        ("#534AB7", "P", ":",  1.5, "Laplacian Score"),
    "all_features":     ("#888780", ".", "-",  1.2, "All Features (baseline)"),
    "datasense_17":     ("#C24B99", "*", "-",  1.5, "DataSense-17 (supervised†)"),
}

# Mapeamento de nomes do CSV para nomes display
METHOD_DISPLAY = {
    "contrastive_proposed": "ContrastiveXAI-FS (ours)",
    "kmeans_silhouette": "k-Means+Sil (Abreu 2022)",
    "mcfs": "MCFS",
    "variance": "Variance",
    "spec": "SPEC",
    "laplacian_score": "Laplacian Score",
    "ndfs": "NDFS",
    "udfs": "UDFS",
    "all_features": "All Features (baseline)",
    "datasense_17_supervised": "DataSense-17 (supervised†)",
}

# Estilo global para o paper
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

K_VALUES = [10, 15, 20, 25]


# =============================================================================
# fig3 — Heatmap comparativo label1 + label2
# =============================================================================

def fig3_comparison_heatmap():
    p = os.path.join(TABLES_DIR, "final_paper_table.csv")
    if not os.path.exists(p):
        print(f"[fig3] {p} não encontrado — rodar exp 07 primeiro.")
        return

    try:
        import seaborn as sns
    except ImportError:
        print("[fig3] seaborn não instalado: pip install seaborn")
        return

    df = pd.read_csv(p)

    methods_order = [
        "contrastive_proposed",
        "kmeans_silhouette",
        "mcfs",
        "variance",
        "spec",
        "laplacian_score",
        "ndfs",
        "udfs",
        "all_features",
        "datasense_17_supervised",
    ]
    labels_order = ["label1", "label2"]

    # Filtrar e pivotar
    df_filt = df[df["method"].isin(methods_order)].copy()
    if df_filt.empty:
        print("[fig3] Nenhum método reconhecido no final_paper_table.csv.")
        return

    pivot_f1 = df_filt.pivot_table(
        index="method", columns="scenario", values="f1_mean", aggfunc="mean"
    ).reindex(index=[m for m in methods_order if m in df_filt["method"].values],
              columns=[l for l in labels_order if l in df_filt["scenario"].values])

    ylabels = [METHOD_DISPLAY.get(m, m) for m in pivot_f1.index]
    xlabels = {
        "label1": "Binary (label1)",
        "label2": "8-class (label2)",
    }
    col_labels = [xlabels.get(c, c) for c in pivot_f1.columns]

    fig, ax = plt.subplots(figsize=(8, len(pivot_f1) * 0.6 + 1.5))
    im = sns.heatmap(
        pivot_f1,
        annot=True, fmt=".3f",
        cmap="RdYlGn", vmin=0.80, vmax=0.95,
        linewidths=0.5, linecolor="white",
        xticklabels=col_labels,
        yticklabels=ylabels,
        ax=ax,
        cbar_kws={"label": "F1-macro (avg classifiers, 5-fold CV)", "shrink": 0.8},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Destacar linha do método proposto
    if "contrastive_proposed" in pivot_f1.index:
        row_idx = list(pivot_f1.index).index("contrastive_proposed")
        ax.add_patch(plt.Rectangle(
            (0, row_idx), len(pivot_f1.columns), 1,
            fill=False, edgecolor="#185FA5", lw=2.5,
        ))

    fig.text(0.01, -0.02, "† supervised — uses labels during feature selection",
             fontsize=9, style="italic")

    _save(fig, "fig3_comparison_heatmap")
    print("[fig3] Heatmap comparativo gerado.")


# =============================================================================
# fig4 — K-sweep lineplot
# =============================================================================

def fig4_k_sweep():
    p = os.path.join(TABLES_DIR, "k_sweep_results.csv")
    if not os.path.exists(p):
        print(f"[fig4] {p} não encontrado — rodar exp 08 primeiro.")
        return

    df = pd.read_csv(p)
    df_rf = df[df["classifier"] == "random_forest"].copy()
    if df_rf.empty:
        print("[fig4] Sem resultados de random_forest no k_sweep_results.csv.")
        return

    # Extrair nome base (remover _k{k} do sufixo)
    df_rf["method_base"] = df_rf["method"].str.replace(r"_k\d+$", "", regex=True)

    label_cols = df_rf["label_col"].unique() if "label_col" in df_rf.columns else ["label1"]

    for label_col in label_cols:
        df_lbl = df_rf[df_rf["label_col"] == label_col] if "label_col" in df_rf.columns else df_rf

        fig, ax = plt.subplots(figsize=(9, 5.5))

        for base, group in df_lbl.groupby("method_base"):
            g = group.sort_values("k")
            style = METHOD_STYLES.get(base, ("#888780", "o", "-", 1.2, base))
            color, marker, ls, lw, label = style
            ax.plot(
                g["k"], g["f1_macro_mean"],
                marker=marker, linestyle=ls, linewidth=lw,
                color=color, label=label, markersize=7,
            )
            ax.fill_between(
                g["k"],
                g["f1_macro_mean"] - g["f1_macro_std"],
                g["f1_macro_mean"] + g["f1_macro_std"],
                alpha=0.10, color=color,
            )

        ax.set_xlabel("Número de features selecionadas (k)", fontsize=12)
        ax.set_ylabel("F1-macro (Random Forest, 5-fold CV)", fontsize=12)
        ax.set_xticks(K_VALUES)
        ax.legend(title="Método", bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.grid(alpha=0.3)
        ax.set_ylim(0.6, 1.02)

        _save(fig, f"fig4_k_sweep_{label_col}")
        print(f"[fig4] K-sweep {label_col} gerado.")


# =============================================================================
# fig5 — SHAP summary (se existir)
# =============================================================================

def fig5_shap_summary():
    src = os.path.join(FIGURES_DIR, "shap_summary.png")
    if not os.path.exists(src):
        print(f"[fig5] {src} não encontrado — rodar exp 06 primeiro.")
        return

    # Reaproveitamos a figura existente; aqui apenas copiamos para paper_ready
    import shutil
    dst = os.path.join(PAPER_FIGS_DIR, "fig5_shap_summary.png")
    shutil.copy2(src, dst)
    print(f"[fig5] Copiado: {src} -> {dst}")


# =============================================================================
# fig2 — t-SNE comparison (requer exp 06 re-rodado com VICReg)
# =============================================================================

def fig2_tsne_comparison():
    tsne_path = os.path.join(FIGURES_DIR, "tsne_clusters_vs_labels.png")
    if not os.path.exists(tsne_path):
        print(f"[fig2] {tsne_path} não encontrado — rodar exp 06 com VICReg primeiro.")
        return

    import shutil
    dst = os.path.join(PAPER_FIGS_DIR, "fig2_tsne_vicreg.png")
    shutil.copy2(tsne_path, dst)
    print(f"[fig2] Copiado: {tsne_path} -> {dst}")


# =============================================================================
# Helpers
# =============================================================================

def _save(fig, name):
    os.makedirs(PAPER_FIGS_DIR, exist_ok=True)
    for ext in ["png", "pdf"]:
        path = os.path.join(PAPER_FIGS_DIR, f"{name}.{ext}")
        fig.savefig(path, format=ext)
    plt.close(fig)
    print(f"  Salvo: {PAPER_FIGS_DIR}/{name}.{{png,pdf}}")


# =============================================================================
# Entry point
# =============================================================================

def run():
    os.makedirs(PAPER_FIGS_DIR, exist_ok=True)
    print(f"\n{'='*60}")
    print("EXPERIMENTO 09 — Figuras para o Paper")
    print(f"{'='*60}")
    print(f"Output: {PAPER_FIGS_DIR}\n")

    fig2_tsne_comparison()
    fig3_comparison_heatmap()
    fig4_k_sweep()
    fig5_shap_summary()

    print(f"\n[Exp 09] Concluído. Figuras em: {PAPER_FIGS_DIR}")
    print("Pendentes (requerem re-execução de exp 02 e 06 com VICReg):")
    print("  fig2_tsne_comparison — precisa exp 06 re-rodado com VICReg")
    print("  fig5_shap_summary   — precisa exp 06 re-rodado com VICReg")


if __name__ == "__main__":
    run()
