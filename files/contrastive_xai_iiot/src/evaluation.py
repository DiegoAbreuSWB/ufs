"""
evaluation.py — Módulo 4: Avaliação Comparativa via 5-fold Stratified CV

EvaluationPipeline avalia conjuntos de features de múltiplos métodos UFS com:
  - 4 classificadores: RandomForest, DecisionTree, KNN, XGBoost
  - Cenários configuráveis (binary, 8-class, 50-class, device)
  - Protocolo anti-leakage: filtros re-selecionam features por fold de treino
  - Métricas: F1-macro, accuracy, precision, recall, MCC, n_features, tempo
"""

import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, matthews_corrcoef
)
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CV_FOLDS, RANDOM_STATE, SCENARIOS

warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# Classificadores
# =============================================================================

def _make_classifiers(include_knn: bool = True) -> dict:
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=100, random_state=RANDOM_STATE,
            eval_metric="mlogloss", verbosity=0,
        )
    except ImportError:
        xgb = None

    clf = {
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    }
    if include_knn:
        clf["knn"] = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    if xgb is not None:
        clf["xgboost"] = xgb
    return clf


def _clone_classifier(clf):
    from sklearn.base import clone
    return clone(clf)


# =============================================================================
# Métricas por fold
# =============================================================================

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "f1_macro":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "accuracy":        float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "mcc":             float(matthews_corrcoef(y_true, y_pred)),
        }


# =============================================================================
# EvaluationPipeline
# =============================================================================

class EvaluationPipeline:
    """
    Avalia múltiplos conjuntos de features via 5-fold Stratified CV.

    Uso:
        pipe = EvaluationPipeline()
        results_df = pipe.run_all(X, y_labels, feature_sets, label_col="label1")
    """

    def __init__(self, n_splits: int = CV_FOLDS, random_state: int = RANDOM_STATE,
                 include_knn: bool = True):
        self.n_splits = n_splits
        self.random_state = random_state
        self.classifiers = _make_classifiers(include_knn=include_knn)

    def evaluate_feature_set(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_indices: list[int],
        method_name: str,
        n_features_total: int,
        selection_time: float = 0.0,
    ) -> list[dict]:
        """
        Avalia um conjunto de features com 5-fold CV × todos os classificadores.
        Retorna lista de dicts (um por clf × fold).
        """
        X_sel = X[:, feature_indices] if feature_indices else X
        n_classes = len(np.unique(y))

        if n_classes < 2:
            return []  # cenário trivial — pular

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        records = []

        for clf_name, clf_template in self.classifiers.items():
            fold_metrics = []
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_sel, y)):
                X_train, X_test = X_sel[train_idx], X_sel[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                clf = _clone_classifier(clf_template)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                fold_metrics.append(_compute_metrics(y_test, y_pred))

            # Agregar folds: média ± std
            agg = {}
            for metric in fold_metrics[0]:
                vals = [fm[metric] for fm in fold_metrics]
                agg[f"{metric}_mean"] = float(np.mean(vals))
                agg[f"{metric}_std"]  = float(np.std(vals))

            records.append({
                "method":           method_name,
                "classifier":       clf_name,
                "n_features":       len(feature_indices) if feature_indices else X.shape[1],
                "n_features_total": n_features_total,
                "reduction_ratio":  round(1 - len(feature_indices) / n_features_total, 4)
                                    if feature_indices else 0.0,
                "selection_time_s": selection_time,
                **agg,
            })

        return records

    def run_all(
        self,
        X: np.ndarray,
        y_labels: pd.DataFrame,
        feature_sets: dict[str, list[int]],
        label_col: str = "label1",
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Roda todos os métodos × classificadores para o cenário dado pelo label_col.

        feature_sets: {method_name: list_of_feature_indices}
          - Índice -1 indica "usar todas as features" (all_features baseline)
          - Índice None indica PCA (não suportado neste pipeline direto)
        """
        if label_col not in y_labels.columns:
            print(f"[Eval] Coluna '{label_col}' não encontrada em y_labels. Pulando.")
            return pd.DataFrame()

        y_raw = y_labels[label_col].values
        n_classes = len(np.unique(y_raw))
        if n_classes < 2:
            print(f"[Eval] '{label_col}' tem {n_classes} classe(s) — cenário trivial, pulando.")
            return pd.DataFrame()

        # Encoder para labels categóricas
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        n_total = X.shape[1]

        if verbose:
            print(f"\n[Eval] Cenário: {label_col} | {n_classes} classes | {self.n_splits}-fold CV")
            print(f"[Eval] Métodos a avaliar: {list(feature_sets.keys())}")
            print(f"[Eval] Classificadores: {list(self.classifiers.keys())}")

        all_records = []
        for method_name, indices in feature_sets.items():
            if indices is None:
                # PCA: pular (requer pipeline especial)
                continue
            if verbose:
                n_sel = len(indices) if indices else n_total
                print(f"[Eval] {method_name}: {n_sel} features...")

            t0 = time.time()
            records = self.evaluate_feature_set(
                X, y, indices, method_name, n_total,
                selection_time=feature_sets.get(f"_time_{method_name}", 0.0),
            )
            elapsed = time.time() - t0

            if records:
                all_records.extend(records)
                if verbose:
                    f1s = [r["f1_macro_mean"] for r in records]
                    print(f"  -> F1-macro: {min(f1s):.3f}–{max(f1s):.3f} | {elapsed:.1f}s")

        return pd.DataFrame(all_records)
