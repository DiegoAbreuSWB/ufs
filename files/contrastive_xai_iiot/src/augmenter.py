"""
src/augmenter.py — Augmentação por grupos semânticos para dados IIoT

Estratégia: em vez de perturbar features individualmente com ruído fixo,
mascarar grupos semânticos inteiros (Context Groups do DataSense).
Duas visões da mesma amostra: ambas com todos os grupos, mas com
perturbações escaladas por grupo + masking aleatório de grupos.

Isso preserva a semântica: dois registros do mesmo tipo de ataque
continuam semanticamente equivalentes mesmo com grupos removidos.
"""

import numpy as np
import torch
from config import CONTEXT_GROUPS


class GroupAwareAugmenter:
    """
    Augmentação contrastiva por grupos semânticos.

    Operações aplicadas por visão (independentemente em view_a e view_b):
      1. Group masking: zerar completamente 1-2 grupos aleatórios (prob=GROUP_MASK_PROB)
      2. Feature noise: ruído gaussiano com std proporcional ao range do grupo
      3. Feature dropout: zerar features individuais (prob=FEAT_DROPOUT_PROB)

    Args:
        feature_names: lista de nomes das features (na ordem de X)
        context_groups: dict {grupo: [features]} do config.py
        group_mask_prob: probabilidade de zerar um grupo inteiro (default 0.3)
        feat_dropout_prob: probabilidade de zerar feature individual (default 0.1)
        noise_scale: escala do ruído gaussiano relativo ao std de cada feature (default 0.05)
    """

    def __init__(
        self,
        feature_names: list[str],
        context_groups: dict[str, list[str]] = None,
        group_mask_prob: float = 0.3,
        feat_dropout_prob: float = 0.1,
        noise_scale: float = 0.05,
    ):
        self.feature_names = feature_names
        self.group_mask_prob = group_mask_prob
        self.feat_dropout_prob = feat_dropout_prob
        self.noise_scale = noise_scale

        # Mapear grupos para índices de features
        if context_groups is None:
            context_groups = CONTEXT_GROUPS

        self.group_indices: dict[str, list[int]] = {}
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        for group_name, feat_list in context_groups.items():
            indices = [name_to_idx[f] for f in feat_list if f in name_to_idx]
            if indices:
                self.group_indices[group_name] = indices

        self.groups = list(self.group_indices.keys())
        self.n_groups = len(self.groups)
        self.n_features = len(feature_names)

        # Número máximo de grupos a mascarar por visão
        # Com 9 grupos, mascarar 2 ainda deixa 7 grupos intactos
        self.n_groups_to_mask = max(1, int(self.n_groups * group_mask_prob))

    def augment(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Gera dois pares de visões aumentadas para cada amostra em X.

        Args:
            X: array (n_samples, n_features), já normalizado
        Returns:
            (view_a, view_b): dois arrays (n_samples, n_features)
        """
        view_a = self._apply_augmentation(X.copy())
        view_b = self._apply_augmentation(X.copy())
        return view_a, view_b

    def augment_torch(
        self, X: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Versão para uso direto no loop de treino PyTorch."""
        X_np = X.cpu().numpy()
        va, vb = self.augment(X_np)
        device = X.device
        return (
            torch.tensor(va, dtype=torch.float32, device=device),
            torch.tensor(vb, dtype=torch.float32, device=device),
        )

    def _apply_augmentation(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]

        # 1. Group masking — para cada amostra, sortear grupos a mascarar
        # Vetorizado: criar máscara (n_samples, n_features)
        mask = np.ones((n_samples, self.n_features), dtype=np.float32)

        for i in range(n_samples):
            if self.n_groups > 0:
                n_mask = np.random.randint(0, self.n_groups_to_mask + 1)
                if n_mask > 0:
                    groups_to_mask = np.random.choice(
                        self.n_groups, size=n_mask, replace=False
                    )
                    for g_idx in groups_to_mask:
                        group_name = self.groups[g_idx]
                        for feat_idx in self.group_indices[group_name]:
                            mask[i, feat_idx] = 0.0

        X = X * mask

        # 2. Feature-level noise (proporcional ao std de cada feature no batch)
        feat_std = np.std(X, axis=0) + 1e-8
        noise = np.random.normal(0, self.noise_scale, X.shape) * feat_std
        X = X + noise

        # 3. Feature dropout individual
        feat_mask = np.random.binomial(
            1, 1 - self.feat_dropout_prob, X.shape
        ).astype(np.float32)
        X = X * feat_mask

        return X.astype(np.float32)
