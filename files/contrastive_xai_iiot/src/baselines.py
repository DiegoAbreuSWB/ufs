"""
baselines.py — Todos os métodos baseline de seleção de features não supervisionada.

Fase 2: BidirectionalSilhouetteFS (dissertação original)
Fase 6: VarianceFS, LaplacianScoreFS, SPECFS, MCFSFS, UDFSFS, NDFSFS, PCAFS
"""

import sys
import os
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    K_RANGE,
    MAX_SEARCH_ITERATIONS,
    SILHOUETTE_SAMPLE_SIZE,
    RANDOM_STATE,
)


class BidirectionalSilhouetteFS:
    """
    Busca bidirecional (SFS ↔ SBS) com k-Means + Silhouette como fitness.
    Opera sobre features brutas normalizadas — baseline da dissertação (2022).

    Estratégia de eficiência:
    - Sub-amosta X UMA VEZ para SILHOUETTE_SAMPLE_SIZE antes da busca.
    - Todas as avaliações internas usam essa sub-amostra fixa.
    - Cache de frozenset → score evita reavaliações.
    - k ótimo determinado UMA VEZ com todas as features.
    """

    def __init__(
        self,
        k_range=K_RANGE,
        max_iter=MAX_SEARCH_ITERATIONS,
        sample_size: int = SILHOUETTE_SAMPLE_SIZE,
        random_state: int = RANDOM_STATE,
        verbose: bool = True,
    ):
        self.k_range = k_range
        self.max_iter = max_iter
        self.sample_size = sample_size
        self.random_state = random_state
        self.verbose = verbose

        self._cache: dict[frozenset, float] = {}
        self.best_k_: int = 2
        self.selected_indices_: list[int] = []
        self.best_score_: float = -1.0
        self.history_: list[dict] = []

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _subsample(self, X: np.ndarray) -> np.ndarray:
        if len(X) <= self.sample_size:
            return X
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(X), size=self.sample_size, replace=False)
        return X[idx]

    def _kmeans_silhouette(self, Xs: np.ndarray, k: int) -> float:
        """Roda k-Means e retorna Silhouette. Retorna -1 se colapso de cluster."""
        km = KMeans(n_clusters=k, n_init=5, random_state=self.random_state)
        labels = km.fit_predict(Xs)
        if len(np.unique(labels)) < 2:
            return -1.0
        # sample_size limita o cálculo O(n²) do Silhouette — crítico para performance
        sil_sample = min(2000, len(Xs))
        return float(silhouette_score(Xs, labels, sample_size=sil_sample, random_state=self.random_state))

    def _find_best_k(self, X_sub: np.ndarray) -> int:
        """Determina k ótimo testando K_RANGE sobre sub-amostra com todas as features."""
        if self.verbose:
            print("[BidirectionalFS] Determinando k ótimo...")
        best_k, best_sil = 2, -1.0
        for k in self.k_range:
            sil = self._kmeans_silhouette(X_sub, k)
            if sil > best_sil:
                best_sil, best_k = sil, k
        if self.verbose:
            print(f"[BidirectionalFS] k ótimo = {best_k} (silhouette = {best_sil:.4f})")
        return best_k

    def _evaluate(self, X_sub: np.ndarray, indices: frozenset) -> float:
        """Avalia subconjunto via k-Means + Silhouette com cache."""
        if indices in self._cache:
            return self._cache[indices]
        idx_list = sorted(indices)
        score = self._kmeans_silhouette(X_sub[:, idx_list], self.best_k_)
        self._cache[indices] = score
        return score

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "BidirectionalSilhouetteFS":
        """
        Executa busca bidirecional sobre X (já normalizado).
        Armazena selected_indices_, best_score_, history_.
        """
        n_features = X.shape[1]
        all_features = frozenset(range(n_features))

        # Sub-amostra fixa para toda a busca
        X_sub = self._subsample(X)
        if self.verbose:
            print(f"[BidirectionalFS] Sub-amostra: {len(X_sub)} / {len(X)} amostras")

        # k ótimo determinado uma única vez
        self.best_k_ = self._find_best_k(X_sub)

        # Inicializar com TODAS as features
        selected = set(range(n_features))
        best_score = self._evaluate(X_sub, frozenset(selected))

        if self.verbose:
            print(f"[BidirectionalFS] Score inicial (68 features): {best_score:.4f}")
            print(f"[BidirectionalFS] Iniciando busca bidirecional...")

        t0 = time.time()
        for iteration in range(self.max_iter):
            best_action = None
            best_new_score = best_score

            # Fase backward: tentar remover cada feature selecionada
            if len(selected) > 2:
                for f in list(selected):
                    score = self._evaluate(X_sub, frozenset(selected - {f}))
                    if score > best_new_score:
                        best_new_score = score
                        best_action = ("remove", f)

            # Fase forward: tentar adicionar cada feature não selecionada
            for f in all_features - frozenset(selected):
                score = self._evaluate(X_sub, frozenset(selected | {f}))
                if score > best_new_score:
                    best_new_score = score
                    best_action = ("add", f)

            self.history_.append({
                "iteration": iteration,
                "n_features": len(selected),
                "score": best_score,
                "action": str(best_action),
            })

            if best_action is None:
                if self.verbose:
                    elapsed = time.time() - t0
                    print(
                        f"[BidirectionalFS] Convergiu na iteração {iteration} | "
                        f"Score: {best_score:.4f} | Features: {len(selected)} | {elapsed:.0f}s"
                    )
                break

            if best_action[0] == "remove":
                selected.discard(best_action[1])
            else:
                selected.add(best_action[1])
            best_score = best_new_score

            if self.verbose and iteration % 5 == 0:
                elapsed = time.time() - t0
                act, feat = best_action
                print(
                    f"[BidirectionalFS] Iter {iteration:3d} | {act} f{feat:2d} | "
                    f"Score: {best_score:.4f} | Features: {len(selected)} | {elapsed:.0f}s"
                )

        self.selected_indices_ = sorted(selected)
        self.best_score_ = best_score
        return self

    def get_selected_features(
        self, feature_names: list[str] | None = None
    ) -> list:
        if feature_names is not None:
            return [feature_names[i] for i in self.selected_indices_]
        return self.selected_indices_


# =============================================================================
# Fase 6 — Baselines UFS com interface uniforme
# =============================================================================

# -----------------------------------------------------------------------------
# Utilitários compartilhados pelos métodos de grafo
# -----------------------------------------------------------------------------

def _build_knn_graph(X: np.ndarray, k_neighbors: int = 5) -> object:
    """
    Constrói grafo k-NN simétrico com pesos Gaussianos.
    Retorna matriz de adjacência esparsa scipy.
    """
    from sklearn.neighbors import kneighbors_graph
    import scipy.sparse as sp

    # Grafo direcionado com distâncias
    A = kneighbors_graph(X, k_neighbors, mode="distance", include_self=False)

    # Converter distâncias para pesos Gaussianos: w = exp(-d²/t)
    t = (A.data.mean() ** 2) if len(A.data) > 0 else 1.0
    A.data = np.exp(-A.data ** 2 / t)

    # Simetrizar: A = (A + Aᵀ) / 2
    A = (A + A.T) / 2
    return A


def _graph_laplacian(A, normalized: bool = False):
    """Retorna Laplaciano (L ou L_sym) a partir de adjacência esparsa."""
    from scipy.sparse.csgraph import laplacian
    return laplacian(A, normed=normalized)


# -----------------------------------------------------------------------------
# Classe base para métodos de ranking de features
# -----------------------------------------------------------------------------

class _RankingFS:
    """Base para métodos que produzem um score por feature (maior = melhor)."""

    scores_: np.ndarray  # (n_features,) após fit
    selected_indices_: list[int] = []

    def select_top_k(self, k: int) -> list[int]:
        self.selected_indices_ = list(np.argsort(self.scores_)[::-1][:k])
        return self.selected_indices_

    def get_selected_features(
        self, k: int, feature_names: list[str] | None = None
    ) -> list:
        indices = self.select_top_k(k)
        if feature_names is not None:
            return [feature_names[i] for i in indices]
        return indices


# -----------------------------------------------------------------------------
# 1. Variance — Filter univariate
# -----------------------------------------------------------------------------

class VarianceFS(_RankingFS):
    """Ranking por variância amostral. Maior variância = feature mais informativa."""

    def fit(self, X: np.ndarray) -> "VarianceFS":
        self.scores_ = np.var(X, axis=0)
        return self


# -----------------------------------------------------------------------------
# 2. Laplacian Score — He, Cai, Niyogi (NIPS 2005)
# -----------------------------------------------------------------------------

class LaplacianScoreFS(_RankingFS):
    """
    Laplacian Score: seleciona features que melhor preservam estrutura local do grafo k-NN.
    Score menor = melhor → invertemos para padronizar interface (maior = melhor).
    """

    def __init__(self, k_neighbors: int = 5, sample_size: int = 5000, random_state: int = RANDOM_STATE):
        self.k_neighbors = k_neighbors
        self.sample_size = sample_size
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "LaplacianScoreFS":
        # Sub-amostra para eficiência
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        idx = rng.choice(n, size=min(self.sample_size, n), replace=False)
        Xs = X[idx]

        A = _build_knn_graph(Xs, self.k_neighbors)
        L = _graph_laplacian(A, normalized=False)
        D_diag = np.asarray(A.sum(axis=1)).ravel()  # grau de cada nó

        scores = np.zeros(Xs.shape[1])
        for i in range(Xs.shape[1]):
            f = Xs[:, i]
            # Centrar pela média ponderada pelo grau
            D_sum = D_diag.sum()
            if D_sum == 0:
                scores[i] = 0.0
                continue
            f_tilde = f - (f * D_diag).sum() / D_sum
            Lf = L @ f_tilde
            denom = (f_tilde * D_diag * f_tilde).sum()
            scores[i] = (f_tilde @ Lf) / denom if abs(denom) > 1e-10 else np.inf

        # Inverter: menor LS = melhor → maior score = melhor
        self.scores_ = -scores
        return self


# -----------------------------------------------------------------------------
# 3. SPEC — Zhao & Liu (ICML 2007)
# -----------------------------------------------------------------------------

class SPECFS(_RankingFS):
    """
    SPEC: usa Laplaciano normalizado L_sym = D^{-1/2} L D^{-1/2}.
    Score = fᵀ L_sym f (menor = mais suave no grafo = melhor feature).
    Invertemos para interface uniforme.
    """

    def __init__(self, k_neighbors: int = 5, sample_size: int = 5000, random_state: int = RANDOM_STATE):
        self.k_neighbors = k_neighbors
        self.sample_size = sample_size
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "SPECFS":
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        idx = rng.choice(n, size=min(self.sample_size, n), replace=False)
        Xs = X[idx]

        A = _build_knn_graph(Xs, self.k_neighbors)
        L_sym = _graph_laplacian(A, normalized=True)

        scores = np.zeros(Xs.shape[1])
        for i in range(Xs.shape[1]):
            f = Xs[:, i]
            norm_sq = f @ f
            if norm_sq < 1e-10:
                scores[i] = np.inf
                continue
            f_norm = f / np.sqrt(norm_sq)
            scores[i] = float(f_norm @ (L_sym @ f_norm))

        self.scores_ = -scores  # menor spec = melhor → inverter
        return self


# -----------------------------------------------------------------------------
# 4. MCFS — Multi-Cluster Feature Selection, Cai et al. (KDD 2010)
# -----------------------------------------------------------------------------

class MCFSFS(_RankingFS):
    """
    MCFS: embedding espectral + regressão LASSO por eigenvector.
    Score_i = max |w_ij| sobre os k eigenvectors.
    Referência: Cai, Zhang, He. KDD 2010.
    """

    def __init__(
        self,
        n_clusters: int = 5,
        k_neighbors: int = 5,
        lasso_alpha: float = 0.01,
        sample_size: int = 5000,
        random_state: int = RANDOM_STATE,
    ):
        self.n_clusters = n_clusters
        self.k_neighbors = k_neighbors
        self.lasso_alpha = lasso_alpha
        self.sample_size = sample_size
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "MCFSFS":
        from scipy.sparse.linalg import eigsh
        from sklearn.linear_model import Lasso

        rng = np.random.default_rng(self.random_state)
        n = len(X)
        idx = rng.choice(n, size=min(self.sample_size, n), replace=False)
        Xs = X[idx]

        A = _build_knn_graph(Xs, self.k_neighbors)
        L = _graph_laplacian(A, normalized=False)

        # Eigenvectors do Laplaciano (menores eigenvalues, exceto 0)
        k = min(self.n_clusters + 1, Xs.shape[0] - 2)
        # sigma=1e-10 usa shift-invert — muito mais estável para menores eigenvalues
        eigenvalues, eigenvectors = eigsh(L, k=k, sigma=1e-10, which="LM", maxiter=5000)
        # Pular o primeiro (eigenvalue ≈ 0, eigenvector constante)
        Y = eigenvectors[:, 1:]  # (n_sub, n_clusters)

        # Para cada eigenvector: LASSO(Xs → y_j)
        W = np.zeros((Xs.shape[1], Y.shape[1]))
        for j in range(Y.shape[1]):
            lasso = Lasso(alpha=self.lasso_alpha, max_iter=1000, random_state=self.random_state)
            lasso.fit(Xs, Y[:, j])
            W[:, j] = np.abs(lasso.coef_)

        self.scores_ = W.max(axis=1)  # max absoluto por feature
        return self


# -----------------------------------------------------------------------------
# 5. UDFS — Yang et al. (IJCAI 2011) — versão simplificada
# -----------------------------------------------------------------------------

class UDFSFS(_RankingFS):
    """
    UDFS (simplificado): resolve problema espectral em M = XᵀLX.
    Score_i = norma da i-ésima linha nos k eigenvectors de M.
    Captura contribuição discriminativa de cada feature.
    Referência: Yang et al. IJCAI 2011.
    """

    def __init__(
        self,
        n_clusters: int = 5,
        k_neighbors: int = 5,
        sample_size: int = 5000,
        random_state: int = RANDOM_STATE,
    ):
        self.n_clusters = n_clusters
        self.k_neighbors = k_neighbors
        self.sample_size = sample_size
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "UDFSFS":
        from scipy.linalg import eigh

        rng = np.random.default_rng(self.random_state)
        n = len(X)
        idx = rng.choice(n, size=min(self.sample_size, n), replace=False)
        Xs = X[idx]

        A = _build_knn_graph(Xs, self.k_neighbors)
        L = _graph_laplacian(A, normalized=True)
        L_dense = np.asarray(L.todense())

        # M = XᵀLX: (d, d) — projeção Laplaciana no espaço de features
        M = Xs.T @ L_dense @ Xs

        # Eigenvectors dos menores eigenvalues de M (direções discriminativas)
        k = min(self.n_clusters, M.shape[0] - 1)
        eigenvalues, eigenvectors = eigh(M, subset_by_index=[0, k - 1])
        # eigenvectors: (d, k)

        # Score = norma-L2 por linha (contribuição de cada feature)
        self.scores_ = np.linalg.norm(eigenvectors, axis=1)
        return self


# -----------------------------------------------------------------------------
# 6. NDFS — Li et al. (AAAI 2012) — versão simplificada
# -----------------------------------------------------------------------------

class NDFSFS(_RankingFS):
    """
    NDFS (simplificado): similar ao UDFS mas usa NMF para obter
    representação não-negativa da atribuição de clusters.
    Score_i = norma da i-ésima linha nos componentes NMF de M.
    Referência: Li et al. AAAI 2012.
    """

    def __init__(
        self,
        n_clusters: int = 5,
        k_neighbors: int = 5,
        sample_size: int = 5000,
        random_state: int = RANDOM_STATE,
    ):
        self.n_clusters = n_clusters
        self.k_neighbors = k_neighbors
        self.sample_size = sample_size
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "NDFSFS":
        from sklearn.decomposition import NMF

        rng = np.random.default_rng(self.random_state)
        n = len(X)
        idx = rng.choice(n, size=min(self.sample_size, n), replace=False)
        Xs = X[idx]

        A = _build_knn_graph(Xs, self.k_neighbors)
        L = _graph_laplacian(A, normalized=True)
        L_dense = np.asarray(L.todense())

        # M = XᵀLX projetado e deslocado para não-negativo
        M = Xs.T @ L_dense @ Xs
        M_pos = M - M.min()  # shift para não-negativo (requisito NMF)

        k = min(self.n_clusters, M_pos.shape[0] - 1)
        nmf = NMF(n_components=k, random_state=self.random_state, max_iter=300)
        W = nmf.fit_transform(M_pos)  # (d, k)

        self.scores_ = np.linalg.norm(W, axis=1)
        return self


# -----------------------------------------------------------------------------
# 7. PCA — Redução de dimensionalidade
# -----------------------------------------------------------------------------

class PCAFS:
    """
    PCA mantendo n_components que explicam 95% da variância.
    Interface diferente dos métodos de ranking: transforma X → X_pca.
    """

    def __init__(self, variance_threshold: float = 0.95, random_state: int = RANDOM_STATE):
        self.variance_threshold = variance_threshold
        self.random_state = random_state
        self.n_components_: int = 0
        self.pca_ = None

    def fit(self, X: np.ndarray) -> "PCAFS":
        from sklearn.decomposition import PCA
        self.pca_ = PCA(n_components=self.variance_threshold, random_state=self.random_state)
        self.pca_.fit(X)
        self.n_components_ = self.pca_.n_components_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca_.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


# =============================================================================
# Registry — interface uniforme para todos os baselines
# =============================================================================

BASELINE_REGISTRY: dict[str, type] = {
    "variance":              VarianceFS,
    "laplacian_score":       LaplacianScoreFS,
    "spec":                  SPECFS,
    "mcfs":                  MCFSFS,
    "udfs":                  UDFSFS,
    "ndfs":                  NDFSFS,
    "pca":                   PCAFS,
    "kmeans_silhouette":     BidirectionalSilhouetteFS,
}


def get_baseline(name: str, **kwargs):
    """Instancia um baseline pelo nome. Ex: get_baseline('laplacian_score', k_neighbors=7)"""
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Baseline desconhecido: '{name}'. Disponíveis: {list(BASELINE_REGISTRY)}")
    return BASELINE_REGISTRY[name](**kwargs)
