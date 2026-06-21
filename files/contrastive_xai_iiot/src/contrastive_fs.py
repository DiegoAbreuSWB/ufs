"""
contrastive_fs.py — Módulo 2: Contrastive Feature Selection (VICReg)

Componentes:
  ContrastiveEncoder        — MLP backbone + projector (expander architecture)
  vicreg_loss               — VICReg loss (Bardes et al., ICLR 2022)
  train_contrastive_encoder — loop de treino com VICReg + GroupAwareAugmenter
  get_representations       — extrai embeddings backbone (128d) para o dataset
  load_encoder              — carrega checkpoint salvo
  ContrastiveBidirectionalFS — busca bidirecional no espaço latente
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ENCODER_HIDDEN_DIM,
    ENCODER_LATENT_DIM,
    ENCODER_PROJECTION_DIM,
    ENCODER_DROPOUT,
    ENCODER_EPOCHS,
    ENCODER_BATCH_SIZE,
    ENCODER_LR,
    VICREG_LAMBDA,
    VICREG_MU,
    VICREG_NU,
    VICREG_EPS,
    GROUP_MASK_PROB,
    FEAT_DROPOUT_PROB,
    AUGMENT_NOISE_SCALE,
    K_RANGE,
    MAX_SEARCH_ITERATIONS,
    SILHOUETTE_SAMPLE_SIZE,
    RANDOM_STATE,
    MODELS_DIR,
    CONTEXT_GROUPS,
)


# =============================================================================
# 1. Encoder
# =============================================================================

class ContrastiveEncoder(nn.Module):
    """
    Encoder MLP para contrastive learning no DataSense IIoT.

    Arquitetura:
      Encoder: input → 256 → BN → ReLU → Dropout → 128 → BN → ReLU
      Projector: 128 → 512 → BN → ReLU → 512  (expander, só usado no treino)

    forward(x)      → h (128d) — representação latente, usada na feature selection
    forward_proj(x) → (h, z)   — backbone + projeção, usado no loop de treino VICReg
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = ENCODER_HIDDEN_DIM,
        latent_dim: int = ENCODER_LATENT_DIM,
        proj_dim: int = ENCODER_PROJECTION_DIM,
        dropout: float = ENCODER_DROPOUT,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )

        # Projector expander — VICReg NÃO usa ReLU final nem normalização L2
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna representação latente h (128d). Usada em inference e feature selection."""
        return self.encoder(x)

    def forward_proj(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Retorna (h, z): backbone 128d e projeção 512d. Usado no treino VICReg."""
        h = self.encoder(x)
        z = self.projector(h)
        return h, z


# =============================================================================
# 2. VICReg Loss (Bardes et al., ICLR 2022)
# =============================================================================

def vicreg_loss(
    z: torch.Tensor,
    z_prime: torch.Tensor,
    lambda_: float = VICREG_LAMBDA,
    mu: float = VICREG_MU,
    nu: float = VICREG_NU,
    eps: float = VICREG_EPS,
) -> tuple[torch.Tensor, dict]:
    """
    VICReg loss com três termos independentes:
      - Invariância: MSE entre as representações das duas visões
      - Variância:   força std >= 1 em cada dimensão do batch (evita colapso)
      - Covariância: penaliza correlação entre dimensões diferentes

    Returns:
        loss total (scalar), dict com os três termos para logging
    """
    batch_size, dim = z.shape

    # 1. Invariância
    loss_inv = F.mse_loss(z, z_prime)

    # 2. Variância — cada dimensão deve ter std >= 1
    def variance_term(x: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(x.var(dim=0) + eps)
        return torch.mean(F.relu(1.0 - std))

    loss_var = (variance_term(z) + variance_term(z_prime)) / 2

    # 3. Covariância — off-diagonal da matriz de covariância deve ser ~0
    def covariance_term(x: torch.Tensor) -> torch.Tensor:
        x_centered = x - x.mean(dim=0)
        cov = (x_centered.T @ x_centered) / (batch_size - 1)
        off_diag = cov.pow(2)
        off_diag.fill_diagonal_(0)
        return off_diag.sum() / dim

    loss_cov = (covariance_term(z) + covariance_term(z_prime)) / 2

    loss = lambda_ * loss_inv + mu * loss_var + nu * loss_cov

    components = {
        "loss_inv": loss_inv.item(),
        "loss_var": loss_var.item(),
        "loss_cov": loss_cov.item(),
        "loss_total": loss.item(),
    }

    return loss, components


# =============================================================================
# 3. Training Loop VICReg
# =============================================================================

def train_contrastive_encoder(
    X: np.ndarray,
    feature_names: list[str],
    context_groups: dict = None,
    save_path: str = None,
    verbose: bool = True,
) -> tuple[ContrastiveEncoder, list[float]]:
    """
    Treina o encoder com VICReg loss e augmentação por grupos semânticos.
    Nunca usa labels — totalmente não supervisionado.

    Args:
        X: array (n_samples, n_features) normalizado
        feature_names: nomes das features na ordem de X
        context_groups: grupos semânticos (usa config.CONTEXT_GROUPS se None)
        save_path: onde salvar o checkpoint .pt
        verbose: imprimir progresso por época
    Returns:
        (encoder treinado, histórico de loss total por época)
    """
    from src.augmenter import GroupAwareAugmenter

    if context_groups is None:
        context_groups = CONTEXT_GROUPS

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"[VICReg] Device: {device}")
        print(f"[VICReg] Dataset: {X.shape[0]:,} amostras × {X.shape[1]} features")
        print(f"[VICReg] Épocas: {ENCODER_EPOCHS} | Batch: {ENCODER_BATCH_SIZE} | LR: {ENCODER_LR}")

    augmenter = GroupAwareAugmenter(
        feature_names=feature_names,
        context_groups=context_groups,
        group_mask_prob=GROUP_MASK_PROB,
        feat_dropout_prob=FEAT_DROPOUT_PROB,
        noise_scale=AUGMENT_NOISE_SCALE,
    )
    if verbose:
        print(f"[VICReg] Grupos semânticos: {len(augmenter.groups)}")
        print(f"[VICReg] Grupos mascarados por visão: 0–{augmenter.n_groups_to_mask}")

    encoder = ContrastiveEncoder(
        input_dim=X.shape[1],
        hidden_dim=ENCODER_HIDDEN_DIM,
        latent_dim=ENCODER_LATENT_DIM,
        proj_dim=ENCODER_PROJECTION_DIM,
        dropout=ENCODER_DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(
        encoder.parameters(), lr=ENCODER_LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=ENCODER_EPOCHS, eta_min=ENCODER_LR * 0.01
    )

    X_tensor = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(X_tensor),
        batch_size=ENCODER_BATCH_SIZE,
        shuffle=True,
        drop_last=True,  # VICReg precisa de batch completo para std/cov
    )

    loss_history: list[float] = []

    for epoch in range(1, ENCODER_EPOCHS + 1):
        encoder.train()
        epoch_losses: dict[str, list[float]] = {"total": [], "inv": [], "var": [], "cov": []}

        for (batch,) in loader:
            view_a, view_b = augmenter.augment_torch(batch)
            view_a = view_a.to(device)
            view_b = view_b.to(device)

            _, z_a = encoder.forward_proj(view_a)
            _, z_b = encoder.forward_proj(view_b)

            loss, components = vicreg_loss(
                z_a, z_b,
                lambda_=VICREG_LAMBDA,
                mu=VICREG_MU,
                nu=VICREG_NU,
                eps=VICREG_EPS,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses["total"].append(components["loss_total"])
            epoch_losses["inv"].append(components["loss_inv"])
            epoch_losses["var"].append(components["loss_var"])
            epoch_losses["cov"].append(components["loss_cov"])

        scheduler.step()

        avg_total = float(np.mean(epoch_losses["total"]))
        loss_history.append(avg_total)

        if verbose and (epoch == 1 or epoch % 10 == 0):
            lr_now = scheduler.get_last_lr()[0]
            avg_inv = float(np.mean(epoch_losses["inv"]))
            avg_var = float(np.mean(epoch_losses["var"]))
            avg_cov = float(np.mean(epoch_losses["cov"]))
            print(
                f"  Época {epoch:3d}/{ENCODER_EPOCHS} | "
                f"Total: {avg_total:.4f} | "
                f"Inv: {avg_inv:.4f} | "
                f"Var: {avg_var:.4f} | "
                f"Cov: {avg_cov:.4f} | "
                f"LR: {lr_now:.6f}"
            )

    encoder.eval()

    # Verificar colapso
    if verbose:
        with torch.no_grad():
            n_check = min(2000, len(X_tensor))
            sample = X_tensor[:n_check].to(device)
            h = encoder(sample)
            repr_std = h.std(dim=0).mean().item()
        status = "OK" if repr_std > 0.01 else "ALERTA — possível colapso"
        print(f"[VICReg] Std médio das representações: {repr_std:.4f} ({status})")

    if save_path:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save({
            "model_state_dict": encoder.state_dict(),
            "input_dim": X.shape[1],
            "hidden_dim": ENCODER_HIDDEN_DIM,
            "latent_dim": ENCODER_LATENT_DIM,
            "proj_dim": ENCODER_PROJECTION_DIM,
            "feature_names": feature_names,
            "loss_history": loss_history,
            "method": "vicreg",
        }, save_path)
        if verbose:
            print(f"[VICReg] Encoder salvo em: {save_path}")

    return encoder, loss_history


# =============================================================================
# 4. Extração de Representações
# =============================================================================

def get_representations(
    encoder: ContrastiveEncoder,
    X: np.ndarray,
    device: str | None = None,
    batch_size: int = 1024,
) -> np.ndarray:
    """
    Extrai embeddings backbone (128d) para todo o dataset X.
    Retorna array (n_samples, 128).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=False)

    reprs = []
    with torch.no_grad():
        for (batch,) in loader:
            h = encoder(batch.to(device))
            reprs.append(h.cpu().numpy())

    return np.vstack(reprs)


def load_encoder(
    path: str, device: str | None = None
) -> tuple[ContrastiveEncoder, list[str]]:
    """
    Carrega encoder salvo por train_contrastive_encoder.
    Suporta formato novo (VICReg, chave 'model_state_dict') e legado ('state_dict').
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(path, map_location=device)

    input_dim = ckpt["input_dim"]
    proj_dim = ckpt.get("proj_dim", ENCODER_PROJECTION_DIM)
    hidden_dim = ckpt.get("hidden_dim", ENCODER_HIDDEN_DIM)
    latent_dim = ckpt.get("latent_dim", ENCODER_LATENT_DIM)

    encoder = ContrastiveEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        proj_dim=proj_dim,
    ).to(device)

    # Suporte a ambos os formatos de checkpoint
    state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict")
    if state_dict is None:
        raise ValueError(f"Checkpoint em {path} não tem 'model_state_dict' nem 'state_dict'.")

    encoder.load_state_dict(state_dict, strict=False)
    encoder.eval()

    return encoder, ckpt.get("feature_names", [])


# =============================================================================
# 5. Busca Bidirecional no Espaço Latente
# =============================================================================

class ContrastiveBidirectionalFS:
    """
    Busca bidirecional (SFS <-> SBS) usando representações do encoder contrastivo.

    Estratégia de eficiência:
    - Treinar encoder UMA VEZ com todas as features.
    - Para avaliar subconjunto S: zerar features fora de S no input (masked forward).
    - Calcular k-Means + Silhouette nas representações latentes 128d.
    - Cache de frozenset -> score evita reavaliações.
    - k ótimo determinado uma vez com todas as features ativas.

    ~100x mais rápido que retreinar o encoder por subconjunto.
    """

    def __init__(
        self,
        encoder: ContrastiveEncoder,
        k_range=K_RANGE,
        max_iter: int = MAX_SEARCH_ITERATIONS,
        sample_size: int = SILHOUETTE_SAMPLE_SIZE,
        random_state: int = RANDOM_STATE,
        device: str | None = None,
        verbose: bool = True,
    ):
        self.encoder = encoder
        self.encoder.eval()
        self.k_range = k_range
        self.max_iter = max_iter
        self.sample_size = sample_size
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

        self._cache: dict[frozenset, float] = {}
        self.best_k_: int = 2
        self.selected_indices_: list[int] = []
        self.best_score_: float = -1.0
        self.history_: list[dict] = []

    def _subsample_idx(self, n: int) -> np.ndarray:
        if n <= self.sample_size:
            return np.arange(n)
        rng = np.random.default_rng(self.random_state)
        return rng.choice(n, size=self.sample_size, replace=False)

    def _get_latent(self, X_tensor: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
        """Forward pass com features mascaradas. Retorna backbone h (N, 128)."""
        masked = X_tensor * mask.to(self.device)
        reprs = []
        batch_size = 1024
        with torch.no_grad():
            for i in range(0, len(masked), batch_size):
                h = self.encoder(masked[i : i + batch_size])
                reprs.append(h.cpu().numpy())
        return np.vstack(reprs)

    def _kmeans_silhouette(self, H: np.ndarray, k: int) -> float:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        km = KMeans(n_clusters=k, n_init=5, random_state=self.random_state)
        labels = km.fit_predict(H)
        if len(np.unique(labels)) < 2:
            return -1.0
        sil_sample = min(2000, len(H))
        return float(silhouette_score(H, labels, sample_size=sil_sample,
                                     random_state=self.random_state))

    def _build_mask(self, indices: frozenset, n_features: int) -> torch.Tensor:
        mask = torch.zeros(1, n_features)
        for i in indices:
            mask[0, i] = 1.0
        return mask

    def _evaluate(
        self,
        X_tensor: torch.Tensor,
        sub_idx: np.ndarray,
        indices: frozenset,
        n_features: int,
    ) -> float:
        if indices in self._cache:
            return self._cache[indices]
        mask = self._build_mask(indices, n_features)
        H = self._get_latent(X_tensor[sub_idx], mask)
        score = self._kmeans_silhouette(H, self.best_k_)
        self._cache[indices] = score
        return score

    def _find_best_k(
        self, X_tensor: torch.Tensor, sub_idx: np.ndarray, n_features: int
    ) -> int:
        if self.verbose:
            print("[ContrastiveFS] Determinando k ótimo no espaço latente...")
        all_feat = frozenset(range(n_features))
        mask = self._build_mask(all_feat, n_features)
        H = self._get_latent(X_tensor[sub_idx], mask)
        best_k, best_sil = 2, -1.0
        for k in self.k_range:
            sil = self._kmeans_silhouette(H, k)
            if sil > best_sil:
                best_sil, best_k = sil, k
        if self.verbose:
            print(f"[ContrastiveFS] k ótimo = {best_k} (silhouette latente = {best_sil:.4f})")
        return best_k

    def fit(self, X: np.ndarray) -> "ContrastiveBidirectionalFS":
        """
        Executa busca bidirecional no espaço latente do encoder.
        X: array normalizado (n_samples, n_features).
        """
        import time
        n_features = X.shape[1]
        all_features = frozenset(range(n_features))

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        sub_idx = self._subsample_idx(len(X))

        if self.verbose:
            print(f"[ContrastiveFS] Sub-amostra: {len(sub_idx)} / {len(X)} amostras")

        self.best_k_ = self._find_best_k(X_tensor, sub_idx, n_features)

        selected = set(range(n_features))
        best_score = self._evaluate(X_tensor, sub_idx, frozenset(selected), n_features)

        if self.verbose:
            print(f"[ContrastiveFS] Score inicial (todas features): {best_score:.4f}")
            print(f"[ContrastiveFS] Iniciando busca bidirecional no espaço latente...")

        t0 = time.time()
        for iteration in range(self.max_iter):
            best_action = None
            best_new_score = best_score

            # Fase backward: tentar remover cada feature
            if len(selected) > 2:
                for f in list(selected):
                    score = self._evaluate(
                        X_tensor, sub_idx, frozenset(selected - {f}), n_features
                    )
                    if score > best_new_score:
                        best_new_score = score
                        best_action = ("remove", f)

            # Fase forward: tentar adicionar feature removida
            for f in all_features - frozenset(selected):
                score = self._evaluate(
                    X_tensor, sub_idx, frozenset(selected | {f}), n_features
                )
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
                        f"[ContrastiveFS] Convergiu na iteração {iteration} | "
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
                    f"[ContrastiveFS] Iter {iteration:3d} | {act} f{feat:2d} | "
                    f"Score: {best_score:.4f} | Features: {len(selected)} | {elapsed:.0f}s"
                )

        self.selected_indices_ = sorted(selected)
        self.best_score_ = best_score
        return self

    def get_selected_features(self, feature_names: list[str] | None = None) -> list:
        if feature_names is not None:
            return [feature_names[i] for i in self.selected_indices_]
        return self.selected_indices_
