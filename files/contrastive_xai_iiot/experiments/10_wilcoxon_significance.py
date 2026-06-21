"""
10_wilcoxon_significance.py — Teste de Wilcoxon pareado entre metodos

Compara VICReg-63 vs All-71 e VICReg-63 vs DataSense-17
para label1 (binario) e label2 (8 classes).

NOTA TECNICA: Com n=5 folds, o Wilcoxon signed-rank nao consegue
atingir p<0.05 (o p minimo com n=5 e 0.0625). Por isso, este script
usa 10-fold CV *somente para o teste estatistico*. Os resultados do
paper (5-fold) sao reportados separadamente e nao sao alterados.

Uso:
    python experiments/10_wilcoxon_significance.py
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FULL_CSV, BENIGN_CSV, TABLES_DIR, DATASENSE_SELECTED_17
from src.preprocessing import DataSensePreprocessor

N_FOLDS_STAT = 10   # folds para o teste (precisamos n>=9 para p<0.05 ser atingivel)
N_FOLDS_PAPER = 5   # folds usados no paper (apenas para referencia)
RANDOM_STATE = 42


def _load_data():
    csv_path = FULL_CSV if os.path.exists(FULL_CSV) else BENIGN_CSV
    print(f"Dataset: {csv_path}")
    pp = DataSensePreprocessor(csv_path)
    X, y = pp.load()
    feature_names = pp.get_feature_names()
    print(f"Shape: {X.shape}")
    return X, y, feature_names


def _load_feature_sets(feature_names):
    name_to_idx = {n: i for i, n in enumerate(feature_names)}

    # VICReg-63
    contra_path = os.path.join(TABLES_DIR, "contrastive_result.json")
    with open(contra_path, encoding="utf-8") as f:
        contra = json.load(f)
    vicreg_indices = [name_to_idx[n] for n in contra["selected_features"] if n in name_to_idx]

    # All-71
    all_indices = list(range(len(feature_names)))

    # DataSense-17
    ds17_indices = [name_to_idx[f] for f in DATASENSE_SELECTED_17 if f in name_to_idx]
    found_ds17 = sum(1 for f in DATASENSE_SELECTED_17 if f in name_to_idx)

    print(f"\nFeature sets:")
    print(f"  VICReg-63 : {len(vicreg_indices)} features")
    print(f"  All-71    : {len(all_indices)} features")
    print(f"  DS-17     : {found_ds17}/{len(DATASENSE_SELECTED_17)} features encontradas")

    return {
        "vicreg_63": vicreg_indices,
        "all_71": all_indices,
        "datasense_17": ds17_indices,
    }


def _cv_f1_per_fold(X, y, feature_indices, n_folds, clf_template, label_col):
    """Retorna array de F1-macro por fold."""
    from sklearn.metrics import f1_score
    from sklearn.base import clone

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_sel = X[:, feature_indices] if feature_indices else X
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    f1_folds = []
    for train_idx, test_idx in skf.split(X_sel, y_enc):
        X_tr, X_te = X_sel[train_idx], X_sel[test_idx]
        y_tr, y_te = y_enc[train_idx], y_enc[test_idx]
        clf = clone(clf_template)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        f1_folds.append(f1_score(y_te, y_pred, average="macro", zero_division=0))

    return np.array(f1_folds)


def _wilcoxon_and_ttest(a, b, method_a, method_b):
    """
    a, b: arrays de F1 por fold (mesmos folds)
    Retorna dict com resultados de Wilcoxon e paired t-test.
    """
    diff = a - b
    n = len(diff)

    # Wilcoxon signed-rank (alternativa: a > b, ou seja, a e melhor)
    try:
        w_stat, w_p = stats.wilcoxon(a, b, alternative="greater")
        w_stat_two, w_p_two = stats.wilcoxon(a, b, alternative="two-sided")
    except ValueError as e:
        # zero_method="wilcox" falha se todos os diffs sao zero
        w_stat, w_p, w_stat_two, w_p_two = None, 1.0, None, 1.0
        print(f"  Wilcoxon warning: {e}")

    # Paired t-test (assume normalidade das diferencas)
    t_stat, t_p_two = stats.ttest_rel(a, b)
    t_p_one = t_p_two / 2 if t_stat > 0 else 1.0 - t_p_two / 2

    return {
        "method_a": method_a,
        "method_b": method_b,
        "n_folds": n,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff": float(np.mean(diff)),
        "std_diff": float(np.std(diff)),
        "wilcoxon_W": float(w_stat) if w_stat is not None else None,
        "wilcoxon_p_one": float(w_p),
        "wilcoxon_p_two": float(w_p_two),
        "ttest_t": float(t_stat),
        "ttest_p_one": float(t_p_one),
        "ttest_p_two": float(t_p_two),
        "sig_wilcoxon_005": w_p < 0.05,
        "sig_ttest_005": t_p_one < 0.05,
    }


def _print_result(r, scenario):
    sig_w = "*" if r["sig_wilcoxon_005"] else " "
    sig_t = "*" if r["sig_ttest_005"] else " "
    print(
        f"  {r['method_a']:12s} vs {r['method_b']:14s} | "
        f"diff={r['mean_diff']:+.4f} (std={r['std_diff']:.4f}) | "
        f"Wilcoxon p={r['wilcoxon_p_one']:.4f}{sig_w} | "
        f"t-test p={r['ttest_p_one']:.4f}{sig_t}"
    )


def run():
    print(f"\n{'='*70}")
    print("EXPERIMENTO 10 — Teste de Wilcoxon pareado")
    print(f"{'='*70}")
    print(f"Folds para o teste estatistico : {N_FOLDS_STAT}")
    print(f"Folds usados no paper (ref)    : {N_FOLDS_PAPER}")
    print("Hipotese alternativa           : VICReg-63 > comparador (one-sided)")
    print(f"{'='*70}")

    X, y, feature_names = _load_data()
    feat_sets = _load_feature_sets(feature_names)

    label_cols = [c for c in ["label1", "label2"]
                  if c in y.columns and y[c].nunique() >= 2]

    clf_rf = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                    random_state=RANDOM_STATE)
    clf_dt = DecisionTreeClassifier(random_state=RANDOM_STATE)

    all_records = []

    for label_col in label_cols:
        y_lbl = y[label_col].values
        n_cls = len(np.unique(y_lbl))
        print(f"\n{'='*70}")
        print(f"CENARIO: {label_col} ({n_cls} classes)")
        print(f"{'='*70}")

        # Coletar F1 por fold para cada metodo e cada clf
        fold_scores = {}
        for method_name, indices in feat_sets.items():
            for clf_name, clf in [("rf", clf_rf), ("dt", clf_dt)]:
                key = f"{method_name}_{clf_name}"
                print(f"  Computando {key} ({N_FOLDS_STAT}-fold)...", end="", flush=True)
                f1s = _cv_f1_per_fold(X, y_lbl, indices, N_FOLDS_STAT, clf, label_col)
                fold_scores[key] = f1s
                print(f" done — mean={np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

        # Comparacoes (usando RF como comparador principal)
        comparisons = [
            ("vicreg_63_rf", "all_71_rf",        "VICReg-63", "All-71"),
            ("vicreg_63_rf", "datasense_17_rf",  "VICReg-63", "DS-17"),
            ("vicreg_63_dt", "all_71_dt",        "VICReg-63-DT", "All-71-DT"),
            ("vicreg_63_dt", "datasense_17_dt",  "VICReg-63-DT", "DS-17-DT"),
        ]

        print(f"\n--- Resultados (one-sided: VICReg > comparador; * = p<0.05) ---")
        for key_a, key_b, name_a, name_b in comparisons:
            r = _wilcoxon_and_ttest(fold_scores[key_a], fold_scores[key_b], name_a, name_b)
            r["scenario"] = label_col
            all_records.append(r)
            _print_result(r, label_col)

    # Salvar CSV com todos os resultados
    out = os.path.join(TABLES_DIR, "wilcoxon_results.csv")
    df = pd.DataFrame(all_records)
    df.to_csv(out, index=False)
    print(f"\n[Exp 10] Resultados salvos: {out}")

    # Resumo executivo
    print(f"\n{'='*70}")
    print("RESUMO EXECUTIVO")
    print(f"{'='*70}")
    df_rf = df[df["method_a"].str.endswith("-DT") == False].copy()
    for _, row in df_rf.iterrows():
        sig = []
        if row["sig_wilcoxon_005"]:
            sig.append("Wilcoxon p<0.05")
        if row["sig_ttest_005"]:
            sig.append("t-test p<0.05")
        status = "SIGNIFICATIVO (" + ", ".join(sig) + ")" if sig else "nao significativo"
        print(f"  [{row['scenario']}] {row['method_a']} vs {row['method_b']}: "
              f"diff={row['mean_diff']:+.4f} — {status}")

    print(f"\nNota: Wilcoxon one-sided (VICReg > comparador). "
          f"Com {N_FOLDS_STAT} folds, p_min = {2/2**N_FOLDS_STAT:.4f} (two-sided).")


if __name__ == "__main__":
    run()
