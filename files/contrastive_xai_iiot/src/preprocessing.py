"""
preprocessing.py — Módulo 1: Pré-processamento do dataset DataSense (CIC IIoT 2025)
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import META_COLUMNS, LIST_COLUMNS, CONTEXT_GROUPS, RANDOM_STATE


class DataSensePreprocessor:
    """Carrega e prepara o dataset DataSense para seleção de features não supervisionada."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.scaler = StandardScaler()
        self._feature_names: list[str] = []
        self._zero_var_features: list[str] = []

    def load(self) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Retorna: (X_numeric, y_labels)

        X_numeric: array normalizado shape (n_samples, n_features)
        y_labels:  DataFrame com colunas label1..label4
        """
        print(f"[Preprocessor] Carregando {self.filepath}...")
        df = pd.read_csv(self.filepath, low_memory=False)
        print(f"[Preprocessor] Shape bruto: {df.shape}")

        # Separar labels + device_name (device_name usada como proxy em dados benign-only)
        label_cols = [c for c in ["label1", "label2", "label3", "label4", "device_name"] if c in df.columns]
        y_labels = df[label_cols].copy()

        # Remover colunas de metadados e de lista
        drop_cols = [c for c in META_COLUMNS + LIST_COLUMNS if c in df.columns]
        df_feat = df.drop(columns=drop_cols, errors="ignore")

        # Manter apenas numéricas
        df_feat = df_feat.select_dtypes(include=[np.number])

        # Remover features com variância zero
        stds = df_feat.std()
        zero_var = stds[stds == 0].index.tolist()
        if zero_var:
            print(f"[Preprocessor] Removendo {len(zero_var)} features com variância zero: {zero_var}")
        self._zero_var_features = zero_var
        df_feat = df_feat.drop(columns=zero_var, errors="ignore")

        # Tratar NaN — ausência de tráfego, preencher com 0
        nan_count = df_feat.isnull().sum().sum()
        if nan_count > 0:
            print(f"[Preprocessor] Substituindo {nan_count:,} NaN por 0")
        df_feat = df_feat.fillna(0)

        # Tratar Inf — substituir por max finito da coluna
        for col in df_feat.columns:
            mask_inf = np.isinf(df_feat[col])
            if mask_inf.any():
                finite_vals = df_feat[col][~mask_inf]
                max_finite = finite_vals.max() if len(finite_vals) > 0 else 0
                df_feat.loc[mask_inf, col] = max_finite

        self._feature_names = df_feat.columns.tolist()
        print(f"[Preprocessor] Features utilizáveis: {len(self._feature_names)}")

        # Normalizar com StandardScaler
        X = self.scaler.fit_transform(df_feat.values)

        return X, y_labels

    def get_feature_names(self) -> list[str]:
        return self._feature_names

    def get_zero_var_features(self) -> list[str]:
        return self._zero_var_features

    def get_context_groups(self) -> dict[str, list[int]]:
        """Retorna mapeamento grupo -> índices de features (apenas as que sobreviveram)."""
        name_to_idx = {name: i for i, name in enumerate(self._feature_names)}
        groups: dict[str, list[int]] = {}
        for group_name, group_cols in CONTEXT_GROUPS.items():
            indices = [name_to_idx[c] for c in group_cols if c in name_to_idx]
            if indices:
                groups[group_name] = indices
        return groups

    def subsample(
        self,
        X: np.ndarray,
        y: pd.DataFrame,
        n_samples: int = 100_000,
        stratify_col: str = "label2",
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """Amostragem estratificada para datasets grandes."""
        if len(X) <= n_samples:
            return X, y

        strat = y[stratify_col] if stratify_col in y.columns else None
        rng = np.random.default_rng(RANDOM_STATE)

        if strat is not None:
            classes, counts = np.unique(strat, return_counts=True)
            fracs = counts / counts.sum()
            alloc = np.maximum(1, (fracs * n_samples).astype(int))
            # Ajustar para bater exatamente n_samples
            diff = n_samples - alloc.sum()
            alloc[np.argmax(fracs)] += diff

            indices = []
            for cls, n in zip(classes, alloc):
                cls_idx = np.where(strat == cls)[0]
                chosen = rng.choice(cls_idx, size=min(n, len(cls_idx)), replace=False)
                indices.extend(chosen.tolist())
            indices = np.array(indices)
            rng.shuffle(indices)
        else:
            indices = rng.choice(len(X), size=n_samples, replace=False)

        return X[indices], y.iloc[indices].reset_index(drop=True)
