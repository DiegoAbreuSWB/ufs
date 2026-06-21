"""
11_english_figures.py — Regenerate paper figures in English as PDF

Figures produced:
  fig_ablation_delta_{label}.{pdf,png}  — ablation incremental-gain chart
  fig_ksweep_{label}.{pdf,png}          — k-sweep lineplot (label1 + label2)
  fig_tsne.{pdf,png}                    — t-SNE latent representations

All figures read from existing CSVs / saved encoder — no experiment is re-run.
Only the t-SNE recomputes from the encoder (approx. 3–5 min).
"""

import sys, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    FULL_CSV, BENIGN_CSV, TABLES_DIR, FIGURES_DIR, MODELS_DIR,
    CONTEXT_GROUPS,
)

OUT_DIR = os.path.join(FIGURES_DIR, "paper_ready")
os.makedirs(OUT_DIR, exist_ok=True)

K_VALUES = [10, 15, 20, 25]

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

# Human-readable names for ablation method codes
ABLATION_LABELS = {
    "A_all_features":      "A: All Features (71)",
    "B_mcfs":              "B: MCFS (25)",
    "B_spec":              "B: SPEC (25)",
    "B_variance":          "B: Variance (25)",
    "B_laplacian_score":   "B: Laplacian Score (25)",
    "B_ndfs":              "B: NDFS (25)",
    "B_udfs":              "B: UDFS (25)",
    "C_kmeans_sil_raw":    "C: k-Means+Sil (46)",
    "D_contrastive_proposed": "D: ContrastiveXAI-FS (63) [ours]",
}

METHOD_STYLES = {
    "vicreg":       ("#185FA5", "o", "-",  2.5, "ContrastiveXAI-FS (ours)"),
    "kmeans_sil":   ("#1D9E75", "s", "--", 1.8, "k-Means+Sil (Abreu 2022)"),
    "mcfs":         ("#BA7517", "^", "--", 1.8, "MCFS"),
    "spec":         ("#6A2585", "D", "--", 1.5, "SPEC"),
    "variance":     ("#A32D2D", "x", ":",  1.5, "Variance"),
    "laplacian":    ("#534AB7", "P", ":",  1.5, "Laplacian Score"),
    "all_features": ("#888780", ".", "-",  1.2, "All Features (baseline)"),
}

SCENARIO_LABEL = {
    "label1": "Binary Classification (label1)",
    "label2": "8-Class Attack Classification (label2)",
}


def _save(fig, name):
    for ext in ["pdf", "png"]:
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(path, format=ext)
    plt.close(fig)
    print(f"  Saved: {OUT_DIR}/{name}.{{pdf,png}}")


# =============================================================================
# Figure: Ablation incremental-gain chart (one per label)
# =============================================================================

def fig_ablation():
    for label_col in ["label1", "label2"]:
        p = os.path.join(TABLES_DIR, f"ablation_{label_col}.csv")
        if not os.path.exists(p):
            print(f"[ablation] {p} not found — skipping.")
            continue

        df = pd.read_csv(p)
        summary = (
            df.groupby("method")["f1_macro_mean"].mean()
        )

        # Build ordered steps: A → best-B → C → D
        steps = []
        for m in ["A_all_features", "C_kmeans_sil_raw"]:
            if m in summary.index:
                steps.append((m, float(summary[m])))

        b_methods = [m for m in summary.index if m.startswith("B_")]
        if b_methods:
            best_b = max(b_methods, key=lambda m: float(summary[m]))
            steps.insert(1, (best_b, float(summary[best_b])))

        d_methods = [m for m in summary.index if m.startswith("D_")]
        if d_methods:
            steps.append((d_methods[0], float(summary[d_methods[0]])))

        if len(steps) < 2:
            print(f"[ablation] Not enough steps for {label_col}.")
            continue

        names_raw = [s[0] for s in steps]
        vals      = [s[1] for s in steps]
        deltas    = [0.0] + [vals[i] - vals[i-1] for i in range(1, len(vals))]
        names_eng = [ABLATION_LABELS.get(n, n) for n in names_raw]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        # Left: absolute F1
        colors_abs = ["#aec6cf"] * len(vals)
        colors_abs[-1] = "#ff6b6b"   # proposed method
        ax1.barh(range(len(names_eng)), vals, color=colors_abs, edgecolor="gray", height=0.55)
        ax1.set_yticks(range(len(names_eng)))
        ax1.set_yticklabels(names_eng, fontsize=9)
        ax1.set_xlabel("Mean F1-macro (RF+DT, 5-fold CV)")
        ax1.set_title(f"F1-macro by Configuration\n({SCENARIO_LABEL[label_col]})")
        ax1.set_xlim(max(0, min(vals) - 0.06), 1.0)
        ax1.axvline(vals[0], color="gray", linestyle="--", alpha=0.5)
        ax1.grid(axis="x", alpha=0.3)

        # Right: incremental deltas
        delta_colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]
        ax2.barh(range(len(names_eng)), deltas, color=delta_colors, edgecolor="gray", height=0.55)
        ax2.set_yticks(range(len(names_eng)))
        ax2.set_yticklabels(names_eng, fontsize=9)
        ax2.set_xlabel("Delta F1-macro vs. previous step")
        ax2.set_title(f"Incremental Gain per Component\n({SCENARIO_LABEL[label_col]})")
        ax2.axvline(0, color="black", linewidth=0.8)
        ax2.grid(axis="x", alpha=0.3)
        for i, d in enumerate(deltas):
            offset = 0.0005 if d >= 0 else -0.0005
            ha = "left" if d >= 0 else "right"
            ax2.text(d + offset, i, f"{d:+.3f}", va="center", ha=ha, fontsize=9)

        fig.suptitle("Ablation Study — ContrastiveXAI-FS", fontsize=13, y=1.01)
        plt.tight_layout()
        _save(fig, f"fig_ablation_{label_col}")
        print(f"[ablation] {label_col} done.")


# =============================================================================
# Figure: K-sweep lineplot
# =============================================================================

def fig_ksweep():
    p = os.path.join(TABLES_DIR, "k_sweep_results.csv")
    if not os.path.exists(p):
        print(f"[ksweep] {p} not found — skipping.")
        return

    df = pd.read_csv(p)
    df_rf = df[df["classifier"] == "random_forest"].copy()
    df_rf["method_base"] = df_rf["method"].str.replace(r"_k\d+$", "", regex=True)

    label_cols = df_rf["label_col"].unique() if "label_col" in df_rf.columns else ["label1"]

    for label_col in sorted(label_cols):
        df_lbl = df_rf[df_rf["label_col"] == label_col]

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

        ax.set_xlabel("Number of selected features (k)", fontsize=12)
        ax.set_ylabel("F1-macro (Random Forest, 5-fold CV)", fontsize=12)
        ax.set_title(SCENARIO_LABEL[label_col])
        ax.set_xticks(K_VALUES)
        ax.legend(title="Method", bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.grid(alpha=0.3)
        ax.set_ylim(0.6, 1.02)

        _save(fig, f"fig_ksweep_{label_col}")
        print(f"[ksweep] {label_col} done.")


# =============================================================================
# Figure: t-SNE latent representations
# =============================================================================

def fig_tsne():
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    from src.preprocessing import DataSensePreprocessor
    from src.contrastive_fs import load_encoder, get_representations

    # Load data
    csv_path = FULL_CSV if os.path.exists(FULL_CSV) else BENIGN_CSV
    print(f"[tsne] Loading data: {csv_path}")
    pp = DataSensePreprocessor(csv_path)
    X, y = pp.load()
    feature_names = pp.get_feature_names()

    # Load VICReg encoder
    encoder_path = os.path.join(MODELS_DIR, "encoder_datasense.pt")
    if not os.path.exists(encoder_path):
        print(f"[tsne] Encoder not found: {encoder_path}")
        return
    print(f"[tsne] Loading encoder: {encoder_path}")
    enc, _ = load_encoder(encoder_path)

    # Load selected features to align with bidirectional result
    contra_path = os.path.join(TABLES_DIR, "contrastive_result.json")
    with open(contra_path, encoding="utf-8") as f:
        d = json.load(f)
    best_k = d.get("best_k", 13)

    # Get representations
    print(f"[tsne] Computing representations ({X.shape[0]:,} samples)...")
    H = get_representations(enc, X)

    # K-Means on latent space
    print(f"[tsne] k-Means with k={best_k}...")
    km = KMeans(n_clusters=best_k, random_state=42, n_init=5)
    cluster_labels = km.fit_predict(H)

    # True labels (binary)
    true_labels = None
    if "label1" in y.columns and y["label1"].nunique() == 2:
        true_labels = y["label1"].values

    # Sub-sample for t-SNE
    n_points = 5000
    rng = np.random.default_rng(42)
    idx = rng.choice(len(H), size=min(n_points, len(H)), replace=False)
    H_sub   = H[idx]
    cl_sub  = cluster_labels[idx]
    tl_sub  = true_labels[idx] if true_labels is not None else None

    print(f"[tsne] Running t-SNE on {len(idx)} points...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=1000)
    H_2d = tsne.fit_transform(H_sub)

    n_cols = 2 if tl_sub is not None else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]

    palette = plt.cm.tab10.colors + plt.cm.tab20b.colors[:3]

    # Left: learned clusters
    for c in np.unique(cl_sub):
        mask = cl_sub == c
        axes[0].scatter(
            H_2d[mask, 0], H_2d[mask, 1],
            s=5, alpha=0.5, label=f"Cluster {c}",
            color=palette[c % len(palette)],
        )
    axes[0].set_title("t-SNE — Learned Clusters (Latent k-Means)")
    axes[0].legend(markerscale=3, fontsize=8)
    axes[0].axis("off")

    # Right: ground-truth labels
    if tl_sub is not None:
        unique_tl = np.unique(tl_sub)
        label_colors = {"attack": "#4C72B0", "benign": "#DD8452"}
        for lbl in unique_tl:
            mask = tl_sub == lbl
            color = label_colors.get(str(lbl), palette[0])
            axes[1].scatter(
                H_2d[mask, 0], H_2d[mask, 1],
                s=5, alpha=0.5, label=str(lbl), color=color,
            )
        axes[1].set_title("t-SNE — Ground-Truth Labels")
        axes[1].legend(markerscale=3, fontsize=9, loc="best")
        axes[1].axis("off")

    fig.suptitle("Latent Representations of the Contrastive Encoder (VICReg)", fontsize=13)
    plt.tight_layout()
    _save(fig, "fig_tsne")
    print("[tsne] done.")


# =============================================================================
# Entry point
# =============================================================================

def run():
    print(f"\n{'='*60}")
    print("EXPERIMENT 11 — English paper figures")
    print(f"{'='*60}")
    print(f"Output: {OUT_DIR}\n")

    print("--- Ablation delta charts ---")
    fig_ablation()

    print("\n--- K-sweep lineplots ---")
    fig_ksweep()

    print("\n--- t-SNE latent representations ---")
    fig_tsne()

    print(f"\n[Exp 11] Done. Files in: {OUT_DIR}")


if __name__ == "__main__":
    run()
