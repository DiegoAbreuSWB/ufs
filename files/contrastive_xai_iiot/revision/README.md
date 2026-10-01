# ContrastiveXAI-FS — reproducibility package (second-round revision)

This directory contains the code that produced **every number, table and figure** of the
revised manuscript. Nothing is fitted outside the training partition of a fold: scaler,
VICReg encoder, number of clusters, feature subset, k-Means model and SHAP proxy are all
learned from the training rows of the current fold.

## Contents

| File | Purpose |
|---|---|
| `rev_common.py` | Dataset loaders, Context Groups (DataSense and WUSTL-IIoT-2021), fold construction (sample-level, execution-grouped, device-grouped), fold-wise preprocessing, group-aware augmentation, VICReg training, frozen-encoder forward pass, Silhouette objective, bidirectional search, ranking baselines, downstream evaluation. All hyperparameters are in the `HP` dictionary. |
| `run_cv.py` | Resumable experiment runner. One *job* = one (dataset, protocol, master seed, fold, variant). Each stage is cached on disk. |
| `rev_analysis.py` | Aggregates the runs into the tables (`results/revision/tables/*.csv`) and figures (`results/revision/figures/*`). |
| `rev_xai.py` | Full-data descriptive configuration: held-out proxy fidelity, TreeExplainer SHAP on held-out rows, per-cluster clarity index, t-SNE and SHAP figures. |
| `manuscript/` | LaTeX source of the revised manuscript; `build.py` produces the clean and the highlighted PDF. |
| `reports/` | Checklist of the reviewer requests, experiment report, response letter. |

## Environment

Python 3.13, PyTorch 2.12 (CPU), scikit-learn 1.9, NumPy 2.5, pandas 3.0, SciPy 1.18,
SHAP 0.52 (requires numba >= 0.68 with NumPy 2.5), matplotlib 3.10.
Hardware used: Dell XPS 9320, Intel Core i7-1260P (12 cores / 16 threads), 16 GB RAM, no GPU.

## Data

* **DataSense CIC IIoT 2025** — `data/dataset_attack_benign.csv` (227,191 one-second windows,
  71 numeric features), obtained from the Canadian Institute for Cybersecurity.
* **WUSTL-IIoT-2021** — `data/external/wustl_iiot_2021/wustl_iiot_2021.csv`, downloaded from
  <https://www.cse.wustl.edu/~jain/iiot2/index.html>. The six columns recommended for removal
  by the dataset authors are dropped; a label-free uniform 25% row sample (seed 2021) is used.

The datasets are not redistributed here; the loaders build numeric caches in `data/cache/`.

## Running

```bash
python revision/run_cv.py --plan main --workers 10   # all experiments (resumable)
python revision/rev_analysis.py                       # tables + figures
python revision/rev_xai.py datasense                  # explainability of the descriptive configuration
python revision/rev_xai.py wustl
python revision/manuscript/build.py                   # clean + highlighted PDF
```

Plans available in `run_cv.py`: `wustl`, `datasense42`, `seeds`, `grouped`, `ablation`, `main` (all).

## Seeds

Master seeds 42, 43, 44. The master seed defines the outer-fold partition; fold `r` uses the
derived seed `s + r` for the encoder-training subsample, network initialization, mini-batch
order, augmentation, the search subsample, k-Means, the graph subsamples of the spectral
baselines and the downstream classifiers.

## Output layout

```
results/revision/runs/<dataset>/<protocol>_k<folds>/seed<S>/fold<r>/<variant>/
    encoder.npz          frozen encoder weights + training-fold scaler statistics
    latent_search.json   k*, Silhouette profiles, search history, selected subset, budget subsets
    raw_search.json      same search in the standardized input space (k-Means+Sil)
    rankings.json        Variance / Laplacian / SPEC / MCFS / UDFS / NDFS rankings
    clusters.npz         k-Means centers of the selected latent representation
    extras.json          cluster stability and proxy fidelity (in-sample, internal hold-out, test fold)
    eval.json            downstream metrics per method, task and classifier
```
