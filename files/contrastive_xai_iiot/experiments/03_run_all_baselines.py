"""
03_run_all_baselines.py — Roda todos os baselines UFS com interface uniforme.

Para cada método:
  1. Aplica seleção de features (sem labels)
  2. Seleciona top-k features (k = n_features selecionadas pelo método proposto)
  3. Reporta: features selecionadas, score, tempo

Os resultados aqui são a base para o experimento de comparação (04_run_comparison.py),
que avalia cada conjunto de features com classificadores supervisionados.

Uso:
    python experiments/03_run_all_baselines.py
    python experiments/03_run_all_baselines.py --csv data/benign_samples_1sec.csv.csv
    python experiments/03_run_all_baselines.py --k 20   # forçar k fixo
"""

import sys
import os
import json
import time
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BENIGN_CSV, FULL_CSV, TABLES_DIR
from src.preprocessing import DataSensePreprocessor
from src.baselines import (
    VarianceFS, LaplacianScoreFS, SPECFS,
    MCFSFS, UDFSFS, NDFSFS, PCAFS,
    BidirectionalSilhouetteFS,
)


def run(csv_path: str, k_override: int | None = None) -> dict:
    print(f"\n{'='*60}")
    print(f"EXPERIMENTO 03 — Todos os Baselines UFS")
    print(f"{'='*60}")
    print(f"Dataset: {csv_path}\n")

    # ------------------------------------------------------------------
    # 1. Pré-processamento
    # ------------------------------------------------------------------
    pp = DataSensePreprocessor(csv_path)
    X, y = pp.load()
    feature_names = pp.get_feature_names()
    n_feat = len(feature_names)

    print(f"Dataset: {X.shape[0]:,} x {X.shape[1]}")

    # k = número de features selecionadas pelo método contrastivo
    if k_override is not None:
        k = k_override
        print(f"k fixo (override): {k}")
    else:
        contrastive_result = os.path.join(TABLES_DIR, "contrastive_result.json")
        if os.path.exists(contrastive_result):
            with open(contrastive_result, encoding="utf-8") as f:
                res = json.load(f)
            k = res["n_features_selected"]
            best_k_clusters = res["best_k"]
            print(f"k do método contrastivo: {k} features | {best_k_clusters} clusters")
        else:
            k = n_feat // 2
            best_k_clusters = 5
            print(f"Resultado contrastivo não encontrado. Usando k={k}")

    # ------------------------------------------------------------------
    # 2. Definir baselines (excluindo kmeans_silhouette que já foi rodado)
    # ------------------------------------------------------------------
    ranking_baselines = {
        "variance":        VarianceFS(),
        "laplacian_score": LaplacianScoreFS(sample_size=5000),
        "spec":            SPECFS(sample_size=5000),
        "mcfs":            MCFSFS(n_clusters=best_k_clusters, sample_size=5000),
        "udfs":            UDFSFS(n_clusters=best_k_clusters, sample_size=5000),
        "ndfs":            NDFSFS(n_clusters=best_k_clusters, sample_size=5000),
    }

    # ------------------------------------------------------------------
    # 3. Rodar cada baseline
    # ------------------------------------------------------------------
    results = {}

    for name, method in ranking_baselines.items():
        print(f"\n[{name.upper()}] Rodando...")
        t0 = time.time()
        try:
            method.fit(X)
            sel_indices = method.select_top_k(k)
            sel_names = [feature_names[i] for i in sel_indices]
            elapsed = time.time() - t0
            print(f"[{name.upper()}] Concluido em {elapsed:.1f}s")
            results[name] = {
                "status": "ok",
                "n_features_selected": k,
                "selected_features": sel_names,
                "time_s": round(elapsed, 2),
            }
        except Exception as e:
            elapsed = time.time() - t0
            print(f"[{name.upper()}] ERRO: {e}")
            results[name] = {"status": "error", "error": str(e), "time_s": round(elapsed, 2)}

    # PCA (interface diferente: transforma, não seleciona features originais)
    print(f"\n[PCA] Rodando...")
    t0 = time.time()
    try:
        pca = PCAFS()
        pca.fit(X)
        elapsed = time.time() - t0
        print(f"[PCA] {pca.n_components_} componentes explicam 95% da variância | {elapsed:.1f}s")
        results["pca"] = {
            "status": "ok",
            "n_components": pca.n_components_,
            "variance_explained": 0.95,
            "time_s": round(elapsed, 2),
            "note": "PCA transforma features; não seleciona features originais",
        }
    except Exception as e:
        results["pca"] = {"status": "error", "error": str(e)}

    # All features (baseline trivial)
    results["all_features"] = {
        "status": "ok",
        "n_features_selected": n_feat,
        "selected_features": feature_names,
        "time_s": 0.0,
    }

    # ------------------------------------------------------------------
    # 4. Relatório
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"RESUMO — Baselines UFS (k={k})")
    print(f"{'='*60}")
    print(f"{'Método':<22} {'Status':>8}  {'Features':>8}  {'Tempo(s)':>9}")
    print(f"{'-'*22} {'-'*8}  {'-'*8}  {'-'*9}")
    for name, res in results.items():
        n_sel = res.get("n_features_selected") or res.get("n_components", "—")
        status = res.get("status", "—")
        t = res.get("time_s", 0)
        print(f"  {name:<20} {status:>8}  {str(n_sel):>8}  {t:>9.1f}")

    # ------------------------------------------------------------------
    # 5. Salvar
    # ------------------------------------------------------------------
    os.makedirs(TABLES_DIR, exist_ok=True)
    out = {
        "csv": os.path.basename(csv_path),
        "n_features_total": int(n_feat),
        "k": int(k),
        "baselines": results,
    }

    def _json_default(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    out_path = os.path.join(TABLES_DIR, "baselines_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=_json_default)
    print(f"\nResultados salvos em: {out_path}")

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None)
    parser.add_argument("--k", type=int, default=None, help="Forçar k fixo de features")
    args = parser.parse_args()

    if args.csv:
        csv_path = args.csv
    elif os.path.exists(FULL_CSV):
        csv_path = FULL_CSV
    else:
        csv_path = BENIGN_CSV

    run(csv_path, k_override=args.k)
