"""
xai_explainer.py — Módulo 3: XAI via SHAP no contexto não supervisionado

Pipeline:
  1. Recebe X (features originais) e cluster_labels (do k-Means no espaço latente)
  2. Treina RandomForest proxy: X → cluster_labels
  3. Aplica shap.TreeExplainer → SHAP values (n_samples, n_features, n_clusters)
  4. Gera visualizações e relatório de coerência semântica

Referência: C-SHAP (Wójcik et al., 2025) — SHAP + k-Means clustering
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # sem display no Windows
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SHAP_PROXY_ESTIMATORS,
    SHAP_PROXY_MAX_DEPTH,
    SHAP_MAX_SAMPLES,
    RANDOM_STATE,
    FIGURES_DIR,
    CONTEXT_GROUPS,
)


# =============================================================================
# Helper: obter cluster labels a partir do encoder
# =============================================================================

def cluster_latent_space(
    representations: np.ndarray,
    n_clusters: int,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Roda k-Means nas representações latentes e retorna labels."""
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    return km.fit_predict(representations)


# =============================================================================
# ClusterSHAPExplainer
# =============================================================================

class ClusterSHAPExplainer:
    """
    Explica clusters não supervisionados via SHAP sobre modelo proxy.

    Uso:
        explainer = ClusterSHAPExplainer(feature_names, context_groups)
        explainer.fit(X_selected, cluster_labels)
        explainer.plot_shap_summary(output_dir)
        explainer.plot_feature_heatmap(output_dir)
        explainer.generate_text_report(selected_features, output_dir)
    """

    def __init__(
        self,
        feature_names: list[str],
        context_groups: dict[str, list[int]] | None = None,
        n_estimators: int = SHAP_PROXY_ESTIMATORS,
        max_depth: int = SHAP_PROXY_MAX_DEPTH,
        max_samples: int = SHAP_MAX_SAMPLES,
        random_state: int = RANDOM_STATE,
    ):
        self.feature_names = feature_names
        self.context_groups = context_groups or {}
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_samples = max_samples
        self.random_state = random_state

        self.proxy_: RandomForestClassifier | None = None
        self.proxy_accuracy_: float = 0.0
        # shap_values_: (n_samples, n_features, n_clusters) após fit
        self.shap_values_: np.ndarray | None = None
        # X e labels usados no fit (subamostrados)
        self.X_fit_: np.ndarray | None = None
        self.labels_fit_: np.ndarray | None = None
        self.n_clusters_: int = 0

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, cluster_labels: np.ndarray) -> "ClusterSHAPExplainer":
        """
        Treina proxy RF e calcula SHAP values.
        X: (n_samples, n_features) — features originais normalizadas
        cluster_labels: (n_samples,) — saída do k-Means no espaço latente
        """
        n = len(X)
        self.n_clusters_ = len(np.unique(cluster_labels))

        # Sub-amostrar para eficiência do SHAP
        if n > self.max_samples:
            rng = np.random.default_rng(self.random_state)
            # Estratificado por cluster
            idx = []
            per_cluster = self.max_samples // self.n_clusters_
            for c in np.unique(cluster_labels):
                c_idx = np.where(cluster_labels == c)[0]
                chosen = rng.choice(c_idx, size=min(per_cluster, len(c_idx)), replace=False)
                idx.extend(chosen.tolist())
            idx = np.array(idx)
        else:
            idx = np.arange(n)

        self.X_fit_ = X[idx]
        self.labels_fit_ = cluster_labels[idx]

        print(f"[SHAP] Treinando proxy RF ({self.n_estimators} trees, max_depth={self.max_depth})...")
        print(f"[SHAP] Sub-amostra: {len(idx)} / {n} | Clusters: {self.n_clusters_}")

        self.proxy_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.proxy_.fit(self.X_fit_, self.labels_fit_)

        preds = self.proxy_.predict(self.X_fit_)
        self.proxy_accuracy_ = accuracy_score(self.labels_fit_, preds)
        print(f"[SHAP] Acuracia do proxy (treino): {self.proxy_accuracy_:.1%}")
        if self.proxy_accuracy_ < 0.85:
            print(f"[SHAP] ATENCAO: acuracia < 85% — explicacoes podem ser fracas")

        print(f"[SHAP] Calculando SHAP values...")
        explainer = shap.TreeExplainer(self.proxy_)
        sv = explainer.shap_values(self.X_fit_)

        # Normalizar para (n_samples, n_features, n_clusters)
        if isinstance(sv, list):
            # sklearn RF multiclass retorna lista de arrays (n_samples, n_features)
            self.shap_values_ = np.stack(sv, axis=2)
        elif sv.ndim == 3:
            self.shap_values_ = sv
        else:
            # Binário: (n_samples, n_features) → expandir
            self.shap_values_ = sv[:, :, np.newaxis]

        print(f"[SHAP] SHAP values shape: {self.shap_values_.shape}")
        return self

    # ------------------------------------------------------------------
    # Importâncias derivadas
    # ------------------------------------------------------------------

    def global_importance(self) -> pd.Series:
        """Média de |SHAP| por feature (todas as classes)."""
        mean_abs = np.abs(self.shap_values_).mean(axis=(0, 2))
        return pd.Series(mean_abs, index=self.feature_names).sort_values(ascending=False)

    def cluster_importance(self) -> pd.DataFrame:
        """DataFrame (features × clusters) com média de |SHAP| por cluster."""
        data = np.abs(self.shap_values_).mean(axis=0)  # (n_features, n_clusters)
        cols = [f"cluster_{c}" for c in range(self.n_clusters_)]
        return pd.DataFrame(data, index=self.feature_names, columns=cols)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_shap_summary(self, output_dir: str) -> str:
        """Beeswarm plot global (top 20 features por |SHAP| médio)."""
        os.makedirs(output_dir, exist_ok=True)
        # Usar valores do primeiro cluster para o beeswarm (mais legível)
        shap_2d = self.shap_values_[:, :, 0]

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_2d,
            self.X_fit_,
            feature_names=self.feature_names,
            max_display=20,
            show=False,
            plot_type="dot",
        )
        plt.title("SHAP Summary Plot — Importância Global das Features", pad=12)
        plt.tight_layout()
        out = os.path.join(output_dir, "shap_summary.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] shap_summary.png salvo")
        return out

    def plot_feature_heatmap(self, output_dir: str) -> str:
        """Heatmap: features × clusters (top 25 features), cor = SHAP médio abs."""
        os.makedirs(output_dir, exist_ok=True)
        df = self.cluster_importance()

        # Top 25 features por importância global
        top_feat = self.global_importance().head(25).index.tolist()
        df_top = df.loc[top_feat]

        fig, ax = plt.subplots(figsize=(max(6, self.n_clusters_ * 1.2), 10))
        sns.heatmap(
            df_top,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": "Mean |SHAP|"},
        )
        ax.set_title("SHAP por Cluster — Top 25 Features", pad=12)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Feature")
        plt.tight_layout()
        out = os.path.join(output_dir, "cluster_heatmap.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] cluster_heatmap.png salvo")
        return out

    def plot_cluster_waterfall(self, cluster_id: int, output_dir: str) -> str:
        """Waterfall plot para a amostra mais representativa de um cluster."""
        os.makedirs(output_dir, exist_ok=True)
        mask = self.labels_fit_ == cluster_id
        if not mask.any():
            print(f"[SHAP] Cluster {cluster_id} vazio, pulando waterfall")
            return ""

        # Pegar a amostra com maior SHAP total para esse cluster (mais representativa)
        shap_cluster = self.shap_values_[mask, :, cluster_id]
        rep_idx = np.argmax(np.abs(shap_cluster).sum(axis=1))
        abs_idx = np.where(mask)[0][rep_idx]

        sv_single = self.shap_values_[abs_idx, :, cluster_id]
        base_val = self.proxy_.predict_proba(self.X_fit_[abs_idx : abs_idx + 1])[0, cluster_id]

        expl = shap.Explanation(
            values=sv_single,
            base_values=float(base_val),
            data=self.X_fit_[abs_idx],
            feature_names=self.feature_names,
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(expl, max_display=15, show=False)
        plt.title(f"SHAP Waterfall — Cluster {cluster_id}", pad=12)
        plt.tight_layout()
        out = os.path.join(output_dir, f"waterfall_cluster_{cluster_id}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] waterfall_cluster_{cluster_id}.png salvo")
        return out

    def plot_tsne_latent(
        self,
        representations: np.ndarray,
        cluster_labels: np.ndarray,
        true_labels: np.ndarray | None = None,
        output_dir: str | None = None,
        n_points: int = 5000,
    ) -> str:
        """
        t-SNE 2D das representações latentes.
        Esquerda: colorido por cluster k-Means.
        Direita: colorido por labels reais (se disponível).
        """
        out_dir = output_dir or FIGURES_DIR
        os.makedirs(out_dir, exist_ok=True)

        # Sub-amostrar para t-SNE (O(n log n) mas lento para >10k)
        n = len(representations)
        if n > n_points:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(n, size=n_points, replace=False)
        else:
            idx = np.arange(n)

        H_sub = representations[idx]
        cl_sub = cluster_labels[idx]
        tl_sub = true_labels[idx] if true_labels is not None else None

        print(f"[SHAP] Calculando t-SNE em {len(idx)} pontos...")
        tsne = TSNE(n_components=2, random_state=self.random_state, perplexity=40, max_iter=1000)
        H_2d = tsne.fit_transform(H_sub)

        n_cols = 2 if tl_sub is not None else 1
        fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6))
        if n_cols == 1:
            axes = [axes]

        # Plot 1: clusters aprendidos
        unique_cl = np.unique(cl_sub)
        palette = plt.cm.tab10.colors
        for c in unique_cl:
            mask = cl_sub == c
            axes[0].scatter(
                H_2d[mask, 0], H_2d[mask, 1],
                s=5, alpha=0.5, label=f"Cluster {c}",
                color=palette[c % len(palette)],
            )
        axes[0].set_title("t-SNE — Clusters Aprendidos (k-Means latente)")
        axes[0].legend(markerscale=3, fontsize=8)
        axes[0].axis("off")

        # Plot 2: labels reais (se disponível)
        if tl_sub is not None:
            unique_tl = np.unique(tl_sub)
            for i, lbl in enumerate(unique_tl):
                mask = tl_sub == lbl
                axes[1].scatter(
                    H_2d[mask, 0], H_2d[mask, 1],
                    s=5, alpha=0.5, label=str(lbl),
                    color=palette[i % len(palette)],
                )
            axes[1].set_title("t-SNE — Labels Reais")
            axes[1].legend(markerscale=3, fontsize=7, loc="best", ncol=2)
            axes[1].axis("off")

        plt.suptitle("Representações Latentes do Encoder Contrastivo", fontsize=13, y=1.02)
        plt.tight_layout()
        out = os.path.join(out_dir, "tsne_clusters_vs_labels.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] tsne_clusters_vs_labels.png salvo")
        return out

    # ------------------------------------------------------------------
    # Relatório textual
    # ------------------------------------------------------------------

    def generate_text_report(
        self,
        selected_features: list[str],
        output_dir: str,
    ) -> str:
        """
        Relatório textual com:
        - Top 10 features globais por SHAP
        - Top 5 features por cluster
        - Comparação: selecionadas vs não-selecionadas
        - Coerência semântica por Context Group
        """
        os.makedirs(output_dir, exist_ok=True)
        lines = []

        lines.append("=" * 70)
        lines.append("RELATÓRIO XAI — ContrastiveXAI-FS para IIoT")
        lines.append("=" * 70)
        lines.append(f"Proxy RF acurácia: {self.proxy_accuracy_:.1%}")
        lines.append(f"Clusters: {self.n_clusters_} | Features totais: {len(self.feature_names)}")
        lines.append(f"Features selecionadas: {len(selected_features)}")
        lines.append("")

        # Top 10 globais
        gi = self.global_importance()
        lines.append("─" * 70)
        lines.append("TOP 10 FEATURES GLOBAIS (média |SHAP|)")
        lines.append("─" * 70)
        for rank, (feat, score) in enumerate(gi.head(10).items(), 1):
            sel_marker = "[SEL]" if feat in selected_features else "     "
            lines.append(f"  {rank:2d}. {sel_marker} {feat:<45} {score:.5f}")
        lines.append("")

        # Top 5 por cluster
        ci = self.cluster_importance()
        lines.append("─" * 70)
        lines.append("TOP 5 FEATURES POR CLUSTER")
        lines.append("─" * 70)
        for c in range(self.n_clusters_):
            col = f"cluster_{c}"
            top5 = ci[col].sort_values(ascending=False).head(5)
            n_samples_c = (self.labels_fit_ == c).sum()
            lines.append(f"\n  Cluster {c}  ({n_samples_c} amostras):")
            for feat, score in top5.items():
                sel_marker = "[SEL]" if feat in selected_features else "     "
                lines.append(f"    {sel_marker} {feat:<45} {score:.5f}")

        # Cobertura de selected vs não-selected no top 10 global
        top10 = set(gi.head(10).index.tolist())
        sel_set = set(selected_features)
        not_sel = set(self.feature_names) - sel_set
        in_top10_selected = top10 & sel_set
        in_top10_not_selected = top10 & not_sel

        lines.append("")
        lines.append("─" * 70)
        lines.append("COMPARAÇÃO: SELECIONADAS vs NÃO SELECIONADAS")
        lines.append("─" * 70)
        lines.append(f"  Features no top-10 SHAP que FORAM selecionadas    : {len(in_top10_selected)}/10")
        lines.append(f"  Features no top-10 SHAP que NÃO foram selecionadas: {len(in_top10_not_selected)}/10")
        if in_top10_not_selected:
            lines.append(f"  Não-selecionadas no top-10: {sorted(in_top10_not_selected)}")
        lines.append("")

        # Coerência semântica por Context Group
        lines.append("─" * 70)
        lines.append("COERÊNCIA SEMÂNTICA POR CONTEXT GROUP")
        lines.append("─" * 70)

        feat_to_group = {}
        for grp, indices in self.context_groups.items():
            for idx in indices:
                if idx < len(self.feature_names):
                    feat_to_group[self.feature_names[idx]] = grp

        group_importance: dict[str, float] = {}
        for feat, score in gi.items():
            grp = feat_to_group.get(feat, "Unknown")
            group_importance[grp] = group_importance.get(grp, 0.0) + score

        lines.append(f"  {'Grupo':<30} {'SHAP Total':>12}  {'Selected':>10}")
        lines.append(f"  {'-'*30} {'-'*12}  {'-'*10}")
        for grp, total in sorted(group_importance.items(), key=lambda x: -x[1]):
            grp_feats = [f for f, g in feat_to_group.items() if g == grp]
            n_sel = sum(1 for f in grp_feats if f in sel_set)
            n_total = len(grp_feats)
            lines.append(f"  {grp:<30} {total:>12.5f}  {n_sel}/{n_total}")

        lines.append("")
        lines.append("─" * 70)
        lines.append("INTERPRETAÇÃO SEMÂNTICA DOS CLUSTERS")
        lines.append("─" * 70)
        for c in range(self.n_clusters_):
            col = f"cluster_{c}"
            top3 = ci[col].sort_values(ascending=False).head(3)
            dominant_groups = []
            for feat in top3.index:
                grp = feat_to_group.get(feat, "Unknown")
                if grp not in dominant_groups:
                    dominant_groups.append(grp)
            lines.append(f"  Cluster {c}: dominado por [{', '.join(dominant_groups)}]")
            lines.append(f"    Features chave: {', '.join(top3.index.tolist())}")

        lines.append("")
        lines.append("=" * 70)

        report_text = "\n".join(lines)
        out_path = os.path.join(output_dir, "xai_report.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        try:
            print(report_text)
        except UnicodeEncodeError:
            print(report_text.encode("ascii", "replace").decode("ascii"))
        print(f"\n[SHAP] Relatorio salvo em: {out_path}")
        return out_path
