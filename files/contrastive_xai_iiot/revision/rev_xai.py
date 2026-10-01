"""
rev_xai.py — Cluster explainability for the full-data descriptive configuration.

Input : results/revision/runs/<dataset>/fulldata_k1/seed42/fold0/main/
        (scaler, frozen encoder, S*, k*, k-Means centers — produced by run_cv.py)
Output: results/revision/tables/xai_*.csv, results/revision/figures/fig_shap_clusters_*.pdf,
        fig_tsne_*.pdf

Procedure
  1. every row is assigned to its latent cluster (nearest k-Means center in f(M_S*(x)));
  2. the rows are split, stratified by cluster, into proxy-training / held-out parts;
  3. a Random Forest proxy (200 trees, depth 10) is trained on a cluster-balanced
     sample of the proxy-training part, using only the selected original features;
  4. fidelity is measured on the held-out part (never seen by the proxy);
  5. TreeExplainer SHAP is computed on held-out rows only;
  6. for every cluster a quantitative clarity index is reported (no cluster is hidden).

Clarity index of cluster c:  CI_c = F_c * T_c
  F_c  held-out F1 of the proxy for cluster c   (is the explanation faithful?)
  T_c  share of the cluster's mean |SHAP| mass carried by its three top features
       (is the profile concentrated enough to be read by an analyst?)
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rev_common as rc  # noqa: E402

TAB = os.path.join(rc.REV_DIR, "tables")
FIG = os.path.join(rc.REV_DIR, "figures")
SEED = 42
EXPLAIN_PER_CLUSTER = 300


def short(n: str) -> str:
    return n.replace("network_", "").replace("_std_deviation", "_std")


def main(dataset: str = "datasense"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support

    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    d_job = os.path.join(rc.REV_DIR, "runs", dataset, "fulldata_k1", f"seed{SEED}", "fold0", "main")
    ds = rc.load_dataset(dataset)
    feats = ds["features"]
    gidx, gnames = rc.group_index_vector(feats, ds["groups"])
    z = np.load(os.path.join(d_job, "encoder.npz"))
    pp = dict(mean=z["pp_mean"], std=z["pp_std"], fmax=z["pp_fmax"])
    enc = {k: z[k] for k in ("W1", "b1", "s1", "t1", "W2", "b2", "s2", "t2")}
    with open(os.path.join(d_job, "latent_search.json"), encoding="utf-8") as f:
        ls = json.load(f)
    S = np.array(ls["selected"])
    k = int(ls["k"])
    C = np.load(os.path.join(d_job, "clusters.npz"))["centers"]
    X = rc.apply_preprocess(ds["X_raw"], pp)
    H = rc.encode(enc, X, S)
    lab_raw = ((H ** 2).sum(1, keepdims=True) - 2 * H @ C.T + (C ** 2).sum(1)).argmin(axis=1)
    # relabel clusters by decreasing size: C1 is the largest
    order = np.argsort(-np.bincount(lab_raw, minlength=k))
    remap = np.empty(k, int)
    remap[order] = np.arange(k)
    lab = remap[lab_raw]
    sizes = np.bincount(lab, minlength=k)
    n = len(X)

    rng = np.random.default_rng(SEED)
    is_train = np.zeros(n, bool)
    for c in range(k):
        idx = rng.permutation(np.where(lab == c)[0])
        is_train[idx[: int(round(0.7 * len(idx)))]] = True
    per = max(1, 5000 // k)
    fit_idx = np.concatenate([rng.permutation(np.where(is_train & (lab == c))[0])[:per] for c in range(k)])
    ho_idx = np.where(~is_train)[0]
    Xs = X[:, S]
    proxy = RandomForestClassifier(n_estimators=rc.HP["proxy_trees"], max_depth=rc.HP["proxy_depth"],
                                   random_state=SEED, n_jobs=4).fit(Xs[fit_idx], lab[fit_idx])
    pred_ho = proxy.predict(Xs[ho_idx])
    pred_in = proxy.predict(Xs[fit_idx])
    prec, rec, f1, sup = precision_recall_fscore_support(lab[ho_idx], pred_ho, labels=np.arange(k), zero_division=0)
    fidelity = dict(
        dataset=dataset, k=k, n_selected=int(len(S)), n_rows=int(n), proxy_fit_rows=int(len(fit_idx)), heldout_rows=int(len(ho_idx)),
        in_sample_accuracy=float(accuracy_score(lab[fit_idx], pred_in)),
        heldout_accuracy=float(accuracy_score(lab[ho_idx], pred_ho)),
        heldout_balanced_accuracy=float(balanced_accuracy_score(lab[ho_idx], pred_ho)),
        heldout_macro_f1=float(f1_score(lab[ho_idx], pred_ho, average="macro", zero_division=0)),
        silhouette_selected=float(ls["J"]), silhouette_full=float(ls["profile_full"][str(k)]),
    )
    pd.DataFrame([fidelity]).to_csv(os.path.join(TAB, f"xai_fidelity_{dataset}.csv"), index=False)

    # ---- SHAP on held-out rows only
    ex_idx = np.concatenate([rng.permutation(ho_idx[lab[ho_idx] == c])[:EXPLAIN_PER_CLUSTER] for c in range(k)])
    sv = shap.TreeExplainer(proxy).shap_values(Xs[ex_idx], check_additivity=False)
    sv = np.stack(sv, axis=2) if isinstance(sv, list) else np.asarray(sv)
    cls = list(proxy.classes_)
    # Eq. (23): mean |phi_{j,c}| over the held-out rows assigned to cluster c
    prof = np.zeros((len(S), k))
    for c in range(k):
        m = lab[ex_idx] == c
        if m.any() and c in cls:
            prof[:, c] = np.abs(sv[m][:, :, cls.index(c)]).mean(axis=0)
    sel_names = [feats[i] for i in S]
    sel_groups = [gnames[gidx[i]] for i in S]
    pd.DataFrame(prof, index=sel_names, columns=[f"C{c + 1}" for c in range(k)]).to_csv(
        os.path.join(TAB, f"xai_cluster_shap_matrix_{dataset}.csv"))
    glob_imp = np.abs(sv).mean(axis=(0, 2))
    pd.DataFrame(dict(feature=sel_names, group=sel_groups, mean_abs_shap=glob_imp)).sort_values(
        "mean_abs_shap", ascending=False).to_csv(os.path.join(TAB, f"xai_global_shap_{dataset}.csv"), index=False)

    meta = ds["meta"]
    y1 = meta[ds["tasks"]["binary"]].to_numpy()
    y2 = meta[ds["tasks"]["multiclass"]].to_numpy()
    benign_name = "benign" if dataset == "datasense" else "normal"
    rows = []
    for c in range(k):
        p = prof[:, c]
        tot = p.sum()
        o = np.argsort(-p)
        share = p / tot if tot > 0 else p
        top3 = float(share[o[:3]].sum())
        ent = -(share[share > 0] * np.log(share[share > 0])).sum()
        top5_groups = pd.Series([sel_groups[i] for i in o[:5]]).value_counts()
        m = lab == c
        vc = pd.Series(y2[m]).value_counts(normalize=True)
        rows.append(dict(
            cluster=f"C{c + 1}", n_rows=int(sizes[c]), share=float(sizes[c] / n),
            heldout_precision=float(prec[c]), heldout_recall=float(rec[c]), heldout_f1=float(f1[c]),
            top3_share=top3, n_eff=float(np.exp(ent)), clarity_index=float(f1[c] * top3),
            dominant_group=top5_groups.index[0], dominant_group_top5=int(top5_groups.iloc[0]),
            top1=short(sel_names[o[0]]), top2=short(sel_names[o[1]]), top3=short(sel_names[o[2]]),
            top1_share=float(share[o[0]]),
            benign_share=float((y1[m] == benign_name).mean()), dominant_label=vc.index[0], dominant_label_share=float(vc.iloc[0]),
        ))
    ctab = pd.DataFrame(rows)
    ctab["clarity_rank"] = ctab.clarity_index.rank(ascending=False, method="first").astype(int)
    ctab.to_csv(os.path.join(TAB, f"xai_cluster_clarity_{dataset}.csv"), index=False)

    # cluster x evaluation-label composition (post-hoc, descriptive only)
    pd.crosstab(pd.Series([f"C{c + 1}" for c in lab], name="cluster"), pd.Series(y2, name="label")).to_csv(
        os.path.join(TAB, f"xai_cluster_label_crosstab_{dataset}.csv"))

    # ---- Figure: per-cluster SHAP profiles (column-normalised shares), top features
    share_m = prof / np.where(prof.sum(0, keepdims=True) > 0, prof.sum(0, keepdims=True), 1)
    top_feats = np.argsort(-share_m.max(axis=1))[:22]
    top_feats = top_feats[np.argsort([sel_groups[i] for i in top_feats], kind="stable")]
    fig, ax = plt.subplots(figsize=(3.5, 4.3))
    im = ax.imshow(share_m[top_feats], aspect="auto", cmap="Blues", vmin=0, vmax=max(0.4, float(share_m.max())))
    ax.set_xticks(np.arange(k))
    ax.set_xticklabels([f"C{c + 1}" for c in range(k)], fontsize=6)
    ax.set_yticks(np.arange(len(top_feats)))
    ax.set_yticklabels([short(sel_names[i]) for i in top_feats], fontsize=5.5)
    ax.set_xlabel("Latent cluster (ordered by size)", fontsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("Share of cluster mean |SHAP|", fontsize=6.5)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_shap_clusters_{dataset}.{ext}"), dpi=300)
    plt.close(fig)

    # ---- Figure: t-SNE of the latent space (qualitative)
    from sklearn.manifold import TSNE
    ti = rng.choice(n, size=min(5000, n), replace=False)
    E = TSNE(n_components=2, random_state=SEED, perplexity=40, max_iter=1000).fit_transform(H[ti])
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))
    cmap = plt.get_cmap("tab20")
    for c in range(k):
        m = lab[ti] == c
        axes[0].scatter(E[m, 0], E[m, 1], s=2, alpha=0.6, color=cmap(c % 20), label=f"C{c + 1}", rasterized=True)
    axes[0].set_title(f"Latent k-Means clusters (k* = {k})", fontsize=8)
    axes[0].legend(markerscale=4, fontsize=5.5, ncol=2, frameon=False, loc="best")
    for i, lbl in enumerate(sorted(np.unique(y1))):
        m = y1[ti] == lbl
        axes[1].scatter(E[m, 0], E[m, 1], s=2, alpha=0.5, color=["#c0504d", "#1f4e79"][i % 2], label=str(lbl), rasterized=True)
    axes[1].set_title("Binary evaluation label (not used for training)", fontsize=8)
    axes[1].legend(markerscale=4, fontsize=6.5, frameon=False)
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_tsne_{dataset}.{ext}"), dpi=300)
    plt.close(fig)

    # ---- removed features and Context-Group coverage of the descriptive configuration
    removed = [i for i in range(len(feats)) if i not in set(S.tolist())]
    pd.DataFrame(dict(feature=[feats[i] for i in removed], group=[gnames[gidx[i]] for i in removed],
                      in_ds17=[feats[i] in rc.DATASENSE_SELECTED_17 for i in removed])).to_csv(
        os.path.join(TAB, f"xai_removed_features_{dataset}.csv"), index=False)
    cov = pd.DataFrame(dict(group=gnames, total=np.bincount(gidx, minlength=len(gnames)),
                            selected=np.bincount(gidx[S], minlength=len(gnames))))
    cov["coverage"] = cov.selected / cov.total
    cov.to_csv(os.path.join(TAB, f"xai_group_coverage_{dataset}.csv"), index=False)

    pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", lambda v: f"{v:.3f}")
    print(json.dumps(fidelity, indent=1))
    print(ctab.to_string(index=False))
    print("removed:", [feats[i] for i in removed])
    print(cov.to_string(index=False))
    print("profile_full:", {kk: round(v, 3) for kk, v in ls["profile_full"].items()})
    print("profile_selected:", {kk: round(v, 3) for kk, v in ls["profile_selected"].items()})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "datasense")
