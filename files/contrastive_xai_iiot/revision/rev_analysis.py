"""
rev_analysis.py — Aggregates the fold-wise runs into the tables and figures of the revision.

Reads   results/revision/runs/**            (written by run_cv.py)
Writes  results/revision/tables/*.csv|.tex  and  results/revision/figures/*.pdf|.png

Every table is computed from whatever jobs are finished, and each CSV records the
number of folds it is based on, so partial runs are never silently mixed up.
"""

from __future__ import annotations

import glob
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rev_common as rc  # noqa: E402

RUNS = os.path.join(rc.REV_DIR, "runs")
TAB = os.path.join(rc.REV_DIR, "tables")
FIG = os.path.join(rc.REV_DIR, "figures")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)


def _j(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def iter_jobs(dataset=None, protocol=None, variant=None, require_done=False):
    for done in sorted(glob.glob(os.path.join(RUNS, "*", "*", "seed*", "fold*", "*"))):
        parts = done.replace("\\", "/").split("/")
        ds, proto, seed, fold, var = parts[-5], parts[-4], int(parts[-3][4:]), int(parts[-2][4:]), parts[-1]
        if dataset and ds != dataset:
            continue
        if protocol and proto != protocol:
            continue
        if variant and var != variant:
            continue
        if require_done and not os.path.exists(os.path.join(done, "DONE.json")):
            continue
        yield dict(dir=done, dataset=ds, protocol=proto, seed=seed, fold=fold, variant=var)


def eval_frame() -> pd.DataFrame:
    rows = []
    for j in iter_jobs():
        p = os.path.join(j["dir"], "eval.json")
        if not os.path.exists(p):
            continue
        for method, v in _j(p).items():
            for task, clfs in v["results"].items():
                for clf, m in clfs.items():
                    rows.append(dict(dataset=j["dataset"], protocol=j["protocol"], seed=j["seed"], fold=j["fold"],
                                     variant=j["variant"], method=method, n_features=v["n_features"], task=task,
                                     clf=clf, **{k: m[k] for k in ("f1_macro", "accuracy", "precision_macro",
                                                                   "recall_macro", "mcc")}))
    return pd.DataFrame(rows)


def wilcoxon_two_sided(a, b):
    from scipy.stats import wilcoxon
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 5 or np.allclose(a, b):
        return np.nan
    return float(wilcoxon(a, b, alternative="two-sided").pvalue)


def paired(ev, dataset, protocol, variant_a, method_a, method_b, task, clf, seeds=None, variant_b="main"):
    a = ev[(ev.dataset == dataset) & (ev.protocol == protocol) & (ev.variant == variant_a) &
           (ev.method == method_a) & (ev.task == task) & (ev.clf == clf)]
    b = ev[(ev.dataset == dataset) & (ev.protocol == protocol) & (ev.variant == variant_b) &
           (ev.method == method_b) & (ev.task == task) & (ev.clf == clf)]
    if seeds is not None:
        a, b = a[a.seed.isin(seeds)], b[b.seed.isin(seeds)]
    m = a.merge(b, on=["seed", "fold"], suffixes=("_a", "_b"))
    return m


# --------------------------------------------------------------------------- #
# Natural operating points (Table 8) and significance (Table 9)
# --------------------------------------------------------------------------- #
METHOD_LABEL = {
    "proposed": "ContrastiveXAI-FS", "kmeans_sil": "k-Means+Sil", "all_features": "All Features",
    "ds17": "DS-17", "mcfs": "MCFS", "variance": "Variance", "spec": "SPEC", "laplacian": "Laplacian",
    "ndfs": "NDFS", "udfs": "UDFS",
}


def natural_points(ev, dataset, seed=42, protocol="sample_k10"):
    d = ev[(ev.dataset == dataset) & (ev.protocol == protocol) & (ev.seed == seed) & (ev.variant == "main")]
    rk = str(rc_rank_k(dataset))
    names = ["proposed", "kmeans_sil"] + [f"{m}_{rk}" for m in ("mcfs", "variance", "spec", "laplacian", "ndfs", "udfs")] \
        + ["all_features", "ds17"]
    rows = []
    for name in names:
        dm = d[d.method == name]
        if dm.empty:
            continue
        row = dict(method=name, n_folds=dm.fold.nunique(), n_features_mean=dm.n_features.mean(),
                   n_features_min=dm.n_features.min(), n_features_max=dm.n_features.max(),
                   classifiers="/".join(sorted(dm.clf.unique())))
        for task in ("binary", "multiclass"):
            dt = dm[dm.task == task]
            avg = dt.groupby("fold")[["f1_macro", "accuracy", "mcc", "recall_macro", "precision_macro"]].mean()
            row[f"{task}_f1_mean"], row[f"{task}_f1_std"] = avg.f1_macro.mean(), avg.f1_macro.std(ddof=0)
            for clf in ("rf", "dt", "knn"):
                dc = dt[dt.clf == clf]
                if not dc.empty:
                    row[f"{task}_f1_{clf}_mean"] = dc.f1_macro.mean()
                    row[f"{task}_f1_{clf}_std"] = dc.f1_macro.std(ddof=0)
            if task == "multiclass":
                row.update(acc=avg.accuracy.mean(), mcc=avg.mcc.mean(), recall=avg.recall_macro.mean(),
                           precision=avg.precision_macro.mean())
                rf = dt[dt.clf == "rf"]
                row.update(rf_recall=rf.recall_macro.mean(), rf_precision=rf.precision_macro.mean(),
                           rf_mcc=rf.mcc.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def rc_rank_k(dataset):
    return {"datasense": 25, "wustl": 14, "wustl_log": 14, "nbaiot_log": 40}[dataset]


def tost_wilcoxon(a, b, delta=0.01):
    """
    Paired equivalence test (two one-sided Wilcoxon signed-rank tests on d = a - b):
      H01: median(d) <= -delta  vs  H11: median(d) > -delta
      H02: median(d) >= +delta  vs  H12: median(d) < +delta
    Equivalence is declared when max(p1, p2) < alpha. Margin fixed in the pre-registration.
    """
    from scipy.stats import wilcoxon
    d = np.asarray(a) - np.asarray(b)
    if len(d) < 5:
        return np.nan
    p1 = wilcoxon(d + delta, alternative="greater").pvalue
    p2 = wilcoxon(d - delta, alternative="less").pvalue
    return float(max(p1, p2))


def significance(ev, dataset, seeds, protocol="sample_k10"):
    rows = []
    rk = rc_rank_k(dataset)
    for task in ("binary", "multiclass"):
        for clf in ("rf", "dt"):
            for ref in ("all_features", "ds17", "kmeans_sil", f"mcfs_{rk}"):
                m = paired(ev, dataset, protocol, "main", "proposed", ref, task, clf, seeds)
                if m.empty:
                    continue
                diff = m.f1_macro_a - m.f1_macro_b
                rows.append(dict(task=task, clf=clf, reference=ref, seeds="+".join(map(str, sorted(m.seed.unique()))),
                                 n_pairs=len(m), mean_proposed=m.f1_macro_a.mean(), mean_reference=m.f1_macro_b.mean(),
                                 mean_diff=diff.mean(), n_positive=int((diff > 0).sum()), n_negative=int((diff < 0).sum()),
                                 wilcoxon_p_two_sided=wilcoxon_two_sided(m.f1_macro_a, m.f1_macro_b),
                                 tost_p_delta001=tost_wilcoxon(m.f1_macro_a, m.f1_macro_b, 0.01),
                                 diff_min=float(diff.min()), diff_max=float(diff.max())))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Multi-seed summary
# --------------------------------------------------------------------------- #
def multiseed(ev, dataset="datasense", protocol="sample_k10"):
    d = ev[(ev.dataset == dataset) & (ev.protocol == protocol) & (ev.variant == "main") &
           (ev.method.isin(["proposed", "all_features"])) & (ev.clf.isin(["rf", "dt"]))]
    g = d.groupby(["seed", "method", "task", "clf"]).agg(f1_mean=("f1_macro", "mean"), f1_std=("f1_macro", lambda s: s.std(ddof=0)),
                                                         n_folds=("fold", "nunique"), nf_mean=("n_features", "mean")).reset_index()
    pooled = g.groupby(["method", "task", "clf"]).agg(seed_mean=("f1_mean", "mean"), seed_std=("f1_mean", lambda s: s.std(ddof=1) if len(s) > 1 else np.nan),
                                                      seed_min=("f1_mean", "min"), seed_max=("f1_mean", "max"), n_seeds=("seed", "nunique")).reset_index()
    return g, pooled


# --------------------------------------------------------------------------- #
# Feature-selection stability
# --------------------------------------------------------------------------- #
def nogueira_stability(Z: np.ndarray) -> float:
    """Nogueira, Sechidis & Brown (JMLR 2018). Z: (runs x features) binary selection matrix."""
    M, d = Z.shape
    pf = Z.mean(axis=0)
    kbar = Z.sum(axis=1).mean()
    denom = (kbar / d) * (1 - kbar / d)
    if denom == 0:
        return np.nan
    return float(1 - (M / (M - 1) * pf * (1 - pf)).mean() / denom)


def selection_stability(dataset="datasense", protocol="sample_k10", which="latent"):
    ds = rc.load_dataset(dataset)
    feats = ds["features"]
    gidx, gnames = rc.group_index_vector(feats, ds["groups"])
    recs = []
    for j in iter_jobs(dataset, protocol, "main"):
        p = os.path.join(j["dir"], "latent_search.json" if which == "latent" else "raw_search.json")
        if not os.path.exists(p):
            continue
        r = _j(p)
        recs.append(dict(seed=j["seed"], fold=j["fold"], selected=r["selected"], k=r["k"], J=r["J"],
                         J_full=r["profile_full"][str(r["k"])], n_evals=r.get("n_evals"),
                         seconds=r.get("seconds_search"), n_fwd=r.get("n_forward_additions", np.nan),
                         profile_full=r["profile_full"], profile_selected=r.get("profile_selected")))
    if not recs:
        return None
    d = len(feats)
    Z = np.zeros((len(recs), d), dtype=int)
    for i, r in enumerate(recs):
        Z[i, r["selected"]] = 1
    sizes = pd.DataFrame([dict(seed=r["seed"], fold=r["fold"], n_selected=len(r["selected"]), n_removed=d - len(r["selected"]),
                               k_star=r["k"], J_full=r["J_full"], J_selected=r["J"], n_evals=r["n_evals"],
                               seconds=r["seconds"], n_forward_additions=r["n_fwd"]) for r in recs])
    freq = pd.DataFrame(dict(feature=feats, group=[gnames[g] for g in gidx], selection_frequency=Z.mean(axis=0),
                             n_selected=Z.sum(axis=0), n_runs=len(recs)))
    for s in sorted(set(r["seed"] for r in recs)):
        m = np.array([r["seed"] == s for r in recs])
        freq[f"freq_seed{s}"] = Z[m].mean(axis=0)
    pairs = []
    for a, b in itertools.combinations(range(len(recs)), 2):
        pairs.append(dict(same_seed=recs[a]["seed"] == recs[b]["seed"], seed_a=recs[a]["seed"], seed_b=recs[b]["seed"],
                          jaccard=rc.jaccard(recs[a]["selected"], recs[b]["selected"]),
                          jaccard_removed=rc.jaccard(set(range(d)) - set(recs[a]["selected"]),
                                                     set(range(d)) - set(recs[b]["selected"]))))
    pairs = pd.DataFrame(pairs, columns=["same_seed", "seed_a", "seed_b", "jaccard", "jaccard_removed"])
    pairs["same_seed"] = pairs["same_seed"].astype(bool)
    summ = dict(dataset=dataset, protocol=protocol, space=which, n_runs=len(recs), d=d,
                size_mean=sizes.n_selected.mean(), size_std=sizes.n_selected.std(ddof=0), size_min=int(sizes.n_selected.min()),
                size_median=float(sizes.n_selected.median()), size_max=int(sizes.n_selected.max()),
                n_always_selected=int((Z.mean(axis=0) == 1).sum()), n_never_selected=int((Z.mean(axis=0) == 0).sum()),
                n_selected_ge_80pct=int((Z.mean(axis=0) >= 0.8).sum()),
                jaccard_within_seed_mean=pairs[pairs.same_seed].jaccard.mean() if pairs.same_seed.any() else np.nan,
                jaccard_within_seed_min=pairs[pairs.same_seed].jaccard.min() if pairs.same_seed.any() else np.nan,
                jaccard_across_seed_mean=pairs[~pairs.same_seed].jaccard.mean() if (~pairs.same_seed).any() else np.nan,
                jaccard_all_mean=pairs.jaccard.mean() if len(pairs) else np.nan,
                jaccard_removed_all_mean=pairs.jaccard_removed.mean() if len(pairs) else np.nan,
                nogueira=nogueira_stability(Z) if len(recs) > 1 else np.nan,
                k_star_counts=json.dumps({int(k): int(v) for k, v in sizes.k_star.value_counts().sort_index().items()}))
    prof = pd.DataFrame([{**{"seed": r["seed"], "fold": r["fold"], "which": "full"}, **{int(k): v for k, v in r["profile_full"].items()}} for r in recs] +
                        [{**{"seed": r["seed"], "fold": r["fold"], "which": "selected"}, **{int(k): v for k, v in r["profile_selected"].items()}}
                         for r in recs if r["profile_selected"]])
    return dict(sizes=sizes, freq=freq, pairs=pairs, summary=summ, profiles=prof, Z=Z, recs=recs)


# --------------------------------------------------------------------------- #
# Cluster stability / proxy fidelity (per-fold extras)
# --------------------------------------------------------------------------- #
def extras_frame(dataset="datasense", protocol=None):
    rows = []
    for j in iter_jobs(dataset, protocol, "main"):
        p = os.path.join(j["dir"], "extras.json")
        if not os.path.exists(p):
            continue
        e = _j(p)
        row = dict(protocol=j["protocol"], seed=j["seed"], fold=j["fold"], k=e["k"], silhouette=e["silhouette"],
                   ari_resample=e["stability_resample"]["ari_mean"], ami_resample=e["stability_resample"]["ami_mean"],
                   ari_resample_min=e["stability_resample"]["ari_min"],
                   min_cluster_share=min(e["cluster_sizes"]) / sum(e["cluster_sizes"]),
                   max_cluster_share=max(e["cluster_sizes"]) / sum(e["cluster_sizes"]))
        for key, tag in (("proxy_in_sample", "ins"), ("proxy_heldout_internal", "hoi"), ("proxy_heldout_test_fold", "hot")):
            if key in e:
                row.update({f"{tag}_acc": e[key]["accuracy"], f"{tag}_bacc": e[key]["balanced_accuracy"],
                            f"{tag}_f1": e[key]["macro_f1"]})
        rows.append(row)
    return pd.DataFrame(rows)


def cross_pipeline_cluster_stability(dataset="datasense", protocol="sample_k10", n_ref=5000, seed=123):
    """
    Each finished fold defines a complete pipeline (scaler, encoder, S*, k-Means centers).
    All pipelines label the same fixed reference rows; agreement is measured with ARI / AMI.
    """
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    ds = rc.load_dataset(dataset)
    ref = np.random.default_rng(seed).choice(len(ds["X_raw"]), size=n_ref, replace=False)
    Xr = ds["X_raw"][ref]
    labs = []
    for j in iter_jobs(dataset, protocol, "main"):
        pc, pe, pl = (os.path.join(j["dir"], f) for f in ("clusters.npz", "encoder.npz", "latent_search.json"))
        if not all(os.path.exists(p) for p in (pc, pe, pl)):
            continue
        z = np.load(pe)
        pp = dict(mean=z["pp_mean"], std=z["pp_std"], fmax=z["pp_fmax"])
        enc = {k: z[k] for k in ("W1", "b1", "s1", "t1", "W2", "b2", "s2", "t2")}
        S = np.array(_j(pl)["selected"])
        H = rc.encode(enc, rc.apply_preprocess(Xr, pp), S)
        C = np.load(pc)["centers"]
        lab = ((H ** 2).sum(1, keepdims=True) - 2 * H @ C.T + (C ** 2).sum(1)).argmin(axis=1)
        labs.append((j["seed"], j["fold"], len(C), lab))
    rows = []
    for (sa, fa, ka, la), (sb, fb, kb, lb) in itertools.combinations(labs, 2):
        rows.append(dict(same_seed=sa == sb, same_k=ka == kb, k_a=ka, k_b=kb,
                         ari=adjusted_rand_score(la, lb), ami=adjusted_mutual_info_score(la, lb)))
    return pd.DataFrame(rows)


def extended_silhouette_profile(dataset="datasense", protocol="sample_k10", k_max=30, seeds=(42,)):
    """
    Sensitivity analysis for the candidate range K = {2..14}: the latent Silhouette of the
    complete feature set is recomputed for k = 2..k_max on the same training-fold subsample
    and with the same seed that the search used. Uses the cached fold encoders.
    """
    ds = rc.load_dataset(dataset)
    rows = []
    for j in iter_jobs(dataset, protocol, "main"):
        if j["seed"] not in seeds:
            continue
        pe = os.path.join(j["dir"], "encoder.npz")
        if not os.path.exists(pe):
            continue
        n_splits = int(protocol.split("_k")[-1])
        tr, _ = rc.make_folds(ds, protocol.rsplit("_k", 1)[0], j["seed"], n_splits)[j["fold"]]
        z = np.load(pe)
        pp = dict(mean=z["pp_mean"], std=z["pp_std"], fmax=z["pp_fmax"])
        enc = {k: z[k] for k in ("W1", "b1", "s1", "t1", "W2", "b2", "s2", "t2")}
        seed_f = j["seed"] + j["fold"]
        sub = np.random.default_rng(seed_f).choice(len(tr), size=min(rc.HP["search_rows"], len(tr)), replace=False)
        H = rc.encode(enc, rc.apply_preprocess(ds["X_raw"][tr[sub]], pp))
        row = dict(seed=j["seed"], fold=j["fold"])
        for k in range(2, k_max + 1):
            row[k] = rc.kmeans_sil(H, k, seed_f)
        rows.append(row)
    return pd.DataFrame(rows)


def fig_extended_profile(prof, name, k_search_max=14):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = [c for c in prof.columns if isinstance(c, int)]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for _, r in prof.iterrows():
        ax.plot(ks, [r[k] for k in ks], color="#1f4e79", alpha=0.25, lw=0.7)
    ax.plot(ks, prof[ks].mean(), color="#1f4e79", lw=1.6, marker="o", ms=2.5, label="Mean over folds")
    ax.axvspan(2, k_search_max, color="#c0504d", alpha=0.08, lw=0)
    ax.axvline(k_search_max, color="#c0504d", lw=0.8, ls="--")
    ax.text(k_search_max - 0.3, ax.get_ylim()[0] + 0.01, r"$\mathcal{K}$ used by the search", fontsize=6.5, color="#c0504d",
            ha="right", va="bottom")
    ax.set_xlabel("Number of clusters $k$", fontsize=8)
    ax.set_ylabel("Latent Silhouette (all features)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fixed budgets, grouped protocols, ablation
# --------------------------------------------------------------------------- #
def budget_table(ev, dataset, seed=42, protocol="sample_k10", variant="main"):
    d = ev[(ev.dataset == dataset) & (ev.protocol == protocol) & (ev.seed == seed) & (ev.variant == variant) &
           (ev.task == "multiclass") & (ev.clf == "rf")]
    budgets = {"datasense": (10, 15, 20, 25), "wustl": (5, 10, 15, 20), "wustl_log": (5, 10, 15, 20), "nbaiot_log": (10, 20, 30, 40)}[dataset]
    rk = rc_rank_k(dataset)
    rows = []
    for m in ("proposed", "kmeans_sil", "mcfs", "variance", "spec", "laplacian"):
        row = dict(method=m)
        for b in budgets:
            name = f"{m}_b{b}" if m in ("proposed", "kmeans_sil") else f"{m}_{b}"
            x = d[d.method == name]
            row[f"k{b}_mean"], row[f"k{b}_std"], row[f"k{b}_n"] = (x.f1_macro.mean(), x.f1_macro.std(ddof=0), len(x)) if len(x) else (np.nan, np.nan, 0)
        rows.append(row)
    a = d[d.method == "all_features"]
    rows.append(dict(method="all_features", **{f"k{b}_mean": a.f1_macro.mean() for b in budgets},
                     **{f"k{b}_std": a.f1_macro.std(ddof=0) for b in budgets}, **{f"k{b}_n": len(a) for b in budgets}))
    return pd.DataFrame(rows)


def protocol_table(ev, dataset="datasense"):
    d = ev[(ev.dataset == dataset) & (ev.variant == "main") & (ev.seed == 42) &
           (ev.method.isin(["proposed", "all_features"])) & (ev.clf.isin(["rf", "dt"]))]
    g = d.groupby(["protocol", "method", "task", "clf"]).agg(f1_mean=("f1_macro", "mean"), f1_std=("f1_macro", lambda s: s.std(ddof=0)),
                                                             acc=("accuracy", "mean"), mcc=("mcc", "mean"), recall=("recall_macro", "mean"),
                                                             n_folds=("fold", "nunique"), nf=("n_features", "mean")).reset_index()
    return g


def per_class_table(dataset="datasense", method="all_features"):
    """Per-class RF F1 under each protocol (shows which categories suffer under grouped splits)."""
    ds = rc.load_dataset(dataset)
    classes = sorted(ds["meta"][ds["tasks"]["multiclass"]].unique())
    rows = []
    for j in iter_jobs(dataset, None, "main"):
        if j["seed"] != 42:
            continue
        p = os.path.join(j["dir"], "eval.json")
        if not os.path.exists(p):
            continue
        ev = _j(p)
        if method not in ev:
            continue
        pc = ev[method]["results"]["multiclass"]["rf"].get("per_class_f1")
        if pc is None or len(pc) != len(classes):
            continue
        rows.append(dict(protocol=j["protocol"], fold=j["fold"], **dict(zip(classes, pc))))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).groupby("protocol")[classes].mean().reset_index()


def duplicate_overlap(dataset="datasense"):
    """Share of test rows whose exact raw feature vector also occurs in the training partition."""
    ds = rc.load_dataset(dataset)
    X = np.ascontiguousarray(ds["X_raw"])
    h = pd.util.hash_pandas_object(pd.DataFrame(X), index=False).to_numpy()
    idle = (X == 0).all(axis=1)
    rows = []
    protos = [("sample", 10)] + ([("group_exec", 5), ("group_device", 5)] if dataset == "datasense" else [])
    for proto, k in protos:
        folds = rc.make_folds(ds, proto, 42, k)
        shares = [float(np.isin(h[te], h[tr]).mean()) for tr, te in folds]
        nonidle = [float((np.isin(h[te], h[tr]) & ~idle[te]).mean()) for tr, te in folds]
        rows.append(dict(protocol=f"{proto}_k{k}", dup_share_mean=np.mean(shares), dup_share_min=np.min(shares),
                         dup_share_max=np.max(shares), nonidle_dup_share_mean=np.mean(nonidle),
                         nonidle_dup_share_max=np.max(nonidle)))
    rows.append(dict(protocol="whole dataset: rows duplicated", dup_share_mean=float(pd.Series(h).duplicated(keep=False).mean()),
                     nonidle_dup_share_mean=float((pd.Series(h).duplicated(keep=False).to_numpy() & ~idle).mean())))
    rows.append(dict(protocol="whole dataset: idle (all-zero) rows", dup_share_mean=float(idle.mean())))
    return pd.DataFrame(rows)


def ablation_table(ev, dataset="datasense"):
    folds = sorted(ev[(ev.dataset == dataset) & (ev.variant == "aug_independent") & (ev.method == "proposed")].fold.unique())
    rows = []
    for variant, label in (("main", "Group masking (proposed)"), ("aug_independent", "Independent masking, matched rate"),
                           ("aug_none", "No masking (noise + dropout)")):
        d = ev[(ev.dataset == dataset) & (ev.protocol == "sample_k10") & (ev.seed == 42) & (ev.variant == variant) &
               (ev.fold.isin(folds))]
        if d.empty:
            continue
        row = dict(configuration=label, n_folds=d.fold.nunique())
        p = d[d.method == "proposed"]
        row["n_features"] = p.n_features.mean()
        for task in ("binary", "multiclass"):
            for clf in ("rf", "dt"):
                x = p[(p.task == task) & (p.clf == clf)]
                row[f"{task}_{clf}"] = x.f1_macro.mean()
        for b in (10, 15, 20, 25):
            x = d[(d.method == f"proposed_b{b}") & (d.task == "multiclass") & (d.clf == "rf")]
            row[f"rf8_b{b}"] = x.f1_macro.mean() if len(x) else np.nan
        rows.append(row)
    # search-direction and search-space controls (same folds, same encoder as the proposed configuration)
    d = ev[(ev.dataset == dataset) & (ev.protocol == "sample_k10") & (ev.seed == 42) & (ev.variant == "main")]
    for name, label in (("proposed_backward_only", "Backward-only search (where it differs)"),
                        ("kmeans_sil_at_nprop", "Raw-space search at |S*| of the latent search"),
                        ("kmeans_sil", "Raw-space search (natural)"), ("all_features", "All features")):
        p = d[d.method == name]
        if p.empty:
            continue
        row = dict(configuration=label, n_folds=p.fold.nunique(), n_features=p.n_features.mean())
        for task in ("binary", "multiclass"):
            x = p[(p.task == task) & (p.clf == "rf")]
            row[f"{task}_rf"] = x.f1_macro.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def ablation_selection(dataset="datasense"):
    rows = []
    for variant in ("main", "aug_independent", "aug_none"):
        recs = []
        for j in iter_jobs(dataset, "sample_k10", variant):
            if j["seed"] != 42 or j["fold"] > 4:
                continue
            p = os.path.join(j["dir"], "latent_search.json")
            if os.path.exists(p):
                recs.append(_j(p))
        if len(recs) < 2:
            continue
        d = len(rc.load_dataset(dataset)["features"])
        Z = np.zeros((len(recs), d), int)
        for i, r in enumerate(recs):
            Z[i, r["selected"]] = 1
        jac = [rc.jaccard(a["selected"], b["selected"]) for a, b in itertools.combinations(recs, 2)]
        rows.append(dict(variant=variant, n_folds=len(recs), size_mean=Z.sum(1).mean(), size_min=int(Z.sum(1).min()),
                         size_max=int(Z.sum(1).max()), jaccard_mean=float(np.mean(jac)), nogueira=nogueira_stability(Z),
                         J_full_mean=float(np.mean([r["profile_full"][str(r["k"])] for r in recs])),
                         J_selected_mean=float(np.mean([r["J"] for r in recs])),
                         k_star=json.dumps(sorted(int(r["k"]) for r in recs))))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_silhouette_profile(st, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prof = st["profiles"]
    ks = [c for c in prof.columns if isinstance(c, int)]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for which, color, label in (("full", "#1f4e79", "All features"), ("selected", "#c0504d", r"Selected subset $S^*_r$")):
        p = prof[prof.which == which][ks]
        if p.empty:
            continue
        m, s = p.mean(), p.std(ddof=0)
        ax.plot(ks, m, marker="o", ms=3, lw=1.2, color=color, label=label)
        ax.fill_between(ks, m - s, m + s, color=color, alpha=0.18, lw=0)
    ax.set_xlabel("Number of clusters $k$")
    ax.set_ylabel("Latent Silhouette")
    ax.set_xticks(ks)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=300)
    plt.close(fig)


def fig_selection_frequency(st, name, max_rows=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = st["freq"].sort_values("selection_frequency")
    f = f[f.selection_frequency < 1.0]
    if f.empty:
        return
    if max_rows:
        f = f.head(max_rows)
    fig, ax = plt.subplots(figsize=(3.5, max(1.6, 0.16 * len(f) + 0.6)))
    ax.barh(np.arange(len(f)), f.selection_frequency, color="#1f4e79", height=0.7)
    ax.set_yticks(np.arange(len(f)))
    ax.set_yticklabels([n.replace("network_", "") for n in f.feature], fontsize=5.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"Selection frequency over {int(f.n_runs.iloc[0])} folds", fontsize=7)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", alpha=0.3, lw=0.4)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ev = eval_frame()
    ev.to_csv(os.path.join(TAB, "eval_long.csv"), index=False)
    out = {}
    for dsname in ("datasense", "wustl", "wustl_log", "nbaiot_log"):
        if (ev.dataset == dsname).any():
            out[f"natural_{dsname}"] = natural_points(ev, dsname)
            seeds42 = [42]
            out[f"significance_seed42_{dsname}"] = significance(ev, dsname, seeds42)
            out[f"budgets_{dsname}"] = budget_table(ev, dsname)
        for which in ("latent", "raw"):
            st = selection_stability(dsname, "sample_k10", which)
            if st is None:
                continue
            out[f"selsizes_{which}_{dsname}"] = st["sizes"]
            out[f"selfreq_{which}_{dsname}"] = st["freq"].sort_values("selection_frequency")
            out[f"selsummary_{which}_{dsname}"] = pd.DataFrame([st["summary"]])
            out[f"silprofile_{which}_{dsname}"] = st["profiles"]
            if which == "latent":
                fig_silhouette_profile(st, f"fig_silhouette_profile_{dsname}")
                fig_selection_frequency(st, f"fig_selection_frequency_{dsname}")
        ex = extras_frame(dsname)
        if not ex.empty:
            out[f"extras_{dsname}"] = ex
            num = ex.drop(columns=["seed", "fold"]).groupby("protocol").agg(["mean", "std", "min"])
            num.columns = ["_".join(c) for c in num.columns]
            out[f"extras_summary_{dsname}"] = num.reset_index()
            cp = cross_pipeline_cluster_stability(dsname)
            if not cp.empty:
                out[f"cluster_cross_{dsname}"] = cp
                out[f"cluster_cross_summary_{dsname}"] = cp.groupby(["same_seed", "same_k"]).agg(
                    ari_mean=("ari", "mean"), ari_std=("ari", "std"), ami_mean=("ami", "mean"), ami_std=("ami", "std"),
                    n_pairs=("ari", "size")).reset_index()
                out[f"cluster_cross_overall_{dsname}"] = pd.DataFrame([dict(
                    ari_mean=cp.ari.mean(), ari_std=cp.ari.std(), ari_min=cp.ari.min(), ami_mean=cp.ami.mean(),
                    ami_std=cp.ami.std(), ami_min=cp.ami.min(), n_pairs=len(cp))])
    if (ev.dataset == "datasense").any():
        g, pooled = multiseed(ev)
        out["multiseed_by_seed"], out["multiseed_pooled"] = g, pooled
        out["significance_allseeds_datasense"] = significance(ev, "datasense", None)
        out["protocols_datasense"] = protocol_table(ev)
        out["protocols_perclass_all_datasense"] = per_class_table("datasense", "all_features")
        out["protocols_perclass_proposed_datasense"] = per_class_table("datasense", "proposed")
        out["ablation_datasense"] = ablation_table(ev)
        out["ablation_selection_datasense"] = ablation_selection()
        out["duplicates_datasense"] = duplicate_overlap("datasense")
    if (ev.dataset == "wustl").any():
        out["duplicates_wustl"] = duplicate_overlap("wustl")
    for name, df in out.items():
        if df is None or len(df) == 0:
            continue
        df.to_csv(os.path.join(TAB, f"{name}.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", lambda v: f"{v:.4f}")
    for name in sys.argv[1:] or [k for k in out if not k.startswith(("selfreq", "silprofile", "extras_d", "extras_w", "cluster_cross_d", "cluster_cross_w", "selsizes"))]:
        if name in out and out[name] is not None and len(out[name]):
            print(f"\n===== {name} =====")
            print(out[name].to_string(index=False))


if __name__ == "__main__":
    main()
