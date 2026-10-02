"""
run_cv.py — Resumable, strictly fold-internal experiment runner.

A *job* is one (dataset, protocol, n_splits, master seed, fold, variant, level).
Every stage of a job is cached on disk, so the runner can be interrupted and
relaunched at any time; finished stages are never recomputed.

Job directory:
  results/revision/runs/<dataset>/<protocol>_k<n_splits>/seed<S>/fold<r>/<variant>/

Stages
  encoder        fold-wise scaler + VICReg encoder trained on the training partition
  latent_search  k* selection, bidirectional search, Silhouette profiles, budgeted subsets
  raw_search     raw-space k-Means+Sil wrapper (same search, standardized input space)
  rankings       Variance / Laplacian / SPEC / MCFS / UDFS / NDFS rankings
  extras         cluster stability, held-out fidelity of the Random Forest proxy
  eval           downstream RF / DT / KNN on the untouched test partition

Levels
  full        all stages, all methods, RF/DT/KNN, fixed-budget regime
  seed        proposed method + All Features (RF, DT)        -> extra master seeds, grouped CV
  ablation    proposed method under a modified augmentation   (RF, DT) + budgets
  descriptive full-data reference configuration (no test partition)

Usage
  python revision/run_cv.py --plan main --workers 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rev_common as rc  # noqa: E402

RUNS = os.path.join(rc.REV_DIR, "runs")
LOGS = os.path.join(rc.REV_DIR, "_logs")

DATASET_CFG = {
    "datasense": dict(rank_k=25, budgets=(10, 15, 20, 25), knn=True, ds17=True,
                      rank_eval=("mcfs", "variance", "spec", "laplacian", "ndfs", "udfs"),
                      budget_rank=("mcfs", "variance"), eval_budgets=True),
    # Unchanged pipeline on WUSTL-IIoT-2021 (direct reproduction): reduced evaluation scope.
    "wustl": dict(rank_k=14, budgets=(5, 10, 15, 20), knn=False, ds17=False,
                  rank_eval=(), budget_rank=(), eval_budgets=False),
    "nbaiot_log": dict(rank_k=40, budgets=(10, 20, 30, 40), knn=False, ds17=False,
                       rank_eval=("mcfs", "variance"), budget_rank=("mcfs",), eval_budgets=True),
    # Log-scaled attributes (methodological adaptation): main external-dataset experiment.
    "wustl_log": dict(rank_k=14, budgets=(5, 10, 15, 20), knn=False, ds17=False,
                      rank_eval=("mcfs", "variance"), budget_rank=("mcfs",), eval_budgets=True),
}
AUG_OF_VARIANT = {"main": "group", "aug_independent": "independent", "aug_none": "none"}

_DS_CACHE: dict = {}
_FOLD_CACHE: dict = {}


def job_dir(job: dict) -> str:
    return os.path.join(RUNS, job["dataset"], f"{job['protocol']}_k{job['n_splits']}",
                        f"seed{job['seed']}", f"fold{job['fold']}", job["variant"])


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _dataset(name):
    if name not in _DS_CACHE:
        ds = rc.load_dataset(name)
        if os.environ.get("REV_SMOKE"):
            keep = np.sort(np.random.default_rng(0).choice(len(ds["X_raw"]), 20_000, replace=False))
            ds["X_raw"] = ds["X_raw"][keep]
            ds["meta"] = ds["meta"].iloc[keep].reset_index(drop=True)
        from sklearn.preprocessing import LabelEncoder
        ds["y"] = {t: LabelEncoder().fit(ds["meta"][c].to_numpy()) for t, c in ds["tasks"].items()}
        ds["y_enc"] = {t: ds["y"][t].transform(ds["meta"][c].to_numpy()) for t, c in ds["tasks"].items()}
        ds["gidx"], ds["gnames"] = rc.group_index_vector(ds["features"], ds["groups"])
        _DS_CACHE[name] = ds
    return _DS_CACHE[name]


def _folds(ds, protocol, seed, n_splits):
    key = (ds["name"], protocol, seed, n_splits)
    if key not in _FOLD_CACHE:
        _FOLD_CACHE[key] = rc.make_folds(ds, protocol, seed, n_splits)
    return _FOLD_CACHE[key]


def stepwise_budgets(obj, start, budgets):
    """
    Budget-constrained variant: forced backward elimination from `start`.
    Above 25 features, floor(0.1*|S|) features are removed per pass (those whose
    individual removal leaves the highest objective); from 25 features downwards
    exactly one feature is removed per pass. Subsets are recorded at each budget.
    """
    S = frozenset(start)
    out = {}
    for b in budgets:
        if b >= len(S):
            out[str(b)] = sorted(int(i) for i in S)
    targets = sorted([b for b in budgets if b < len(S)], reverse=True)
    while targets:
        scores = sorted(((obj(S - {j}), j) for j in sorted(S)), reverse=True)
        r = max(1, int(0.1 * len(S))) if len(S) > 25 else 1
        r = min(r, len(S) - targets[0])
        S = S - {j for _, j in scores[:r]}
        if len(S) == targets[0]:
            out[str(targets.pop(0))] = sorted(int(i) for i in S)
    return out


def run_job(job: dict) -> str:
    import warnings
    warnings.filterwarnings("ignore")
    import sklearn
    import torch
    torch.set_num_threads(1)
    sklearn.set_config(working_memory=128)

    d_job = job_dir(job)
    os.makedirs(d_job, exist_ok=True)
    tag = f"{job['dataset']}/{job['protocol']}_k{job['n_splits']}/s{job['seed']}/f{job['fold']}/{job['variant']}"
    log = rc.Logger(os.path.join(d_job, "job.log"))
    prog = os.path.join(LOGS, "progress.log")

    def mark(stage, t0):
        with open(prog, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {tag} {stage} {time.time() - t0:.0f}s\n")

    try:
        ds = _dataset(job["dataset"])
        cfg = DATASET_CFG[job["dataset"]]
        level = job["level"]
        d = len(ds["features"])
        seed_f = job["seed"] + job["fold"]
        if level == "descriptive":
            tr = np.arange(len(ds["X_raw"]))
            te = np.array([], dtype=int)
        else:
            tr, te = _folds(ds, job["protocol"], job["seed"], job["n_splits"])[job["fold"]]

        pp = rc.fit_preprocess(ds["X_raw"][tr])
        X_tr = rc.apply_preprocess(ds["X_raw"][tr], pp)
        X_te = rc.apply_preprocess(ds["X_raw"][te], pp) if len(te) else np.zeros((0, d), np.float32)

        # ---------------- encoder ----------------
        p_enc = os.path.join(d_job, "encoder.npz")
        if not os.path.exists(p_enc):
            t0 = time.time()
            model, hist = rc.train_encoder(X_tr, ds["gidx"], len(ds["gnames"]), seed_f,
                                           aug_mode=AUG_OF_VARIANT[job["variant"]], log=log)
            enc = rc.encoder_to_numpy(model)
            np.savez(p_enc + ".tmp.npz", **enc, pp_mean=pp["mean"], pp_std=pp["std"], pp_fmax=pp["fmax"],
                     loss_hist=np.array(hist), train_seconds=time.time() - t0)
            os.replace(p_enc + ".tmp.npz", p_enc)
            mark("encoder", t0)
        z = np.load(p_enc)
        enc = {k: z[k] for k in ("W1", "b1", "s1", "t1", "W2", "b2", "s2", "t2")}

        rng = np.random.default_rng(seed_f)
        sub = rng.choice(len(X_tr), size=min(rc.HP["search_rows"], len(X_tr)), replace=False)
        X_s = X_tr[sub]

        # ---------------- latent search ----------------
        p_ls = os.path.join(d_job, "latent_search.json")
        if not os.path.exists(p_ls):
            t0 = time.time()
            obj = rc.SubsetObjective(X_s, seed_f, "latent", enc)
            res = rc.bidirectional_search(obj, d, log=log)
            res["seconds_search"] = time.time() - t0
            res["profile_full"] = obj.profile_full
            res["profile_selected"] = obj.profile(res["selected"])
            S = frozenset(res["selected"])
            res["loo_J"] = {str(j): obj(S - {j}) for j in sorted(S)} if len(S) > 2 else {}
            bo = rc.backward_only_search(obj, d)
            res["backward_only"] = bo["selected"]
            res["n_forward_additions"] = sum(1 for h in res["history"] if h["action"] == "add")
            if level in ("full", "ablation"):
                t1 = time.time()
                res["budgets"] = stepwise_budgets(obj, res["selected"], cfg["budgets"])
                res["seconds_budgets"] = time.time() - t1
            res["n_evals_total"] = len(obj.cache)
            _save(p_ls, res)
            mark("latent_search", t0)
        ls = _load(p_ls)
        S_star = ls["selected"]

        # ---------------- raw-space wrapper + rankings ----------------
        raw, rk = None, None
        if level == "full":
            p_rs = os.path.join(d_job, "raw_search.json")
            if not os.path.exists(p_rs):
                t0 = time.time()
                objr = rc.SubsetObjective(X_s, seed_f, "raw")
                res = rc.bidirectional_search(objr, d, log=log)
                res["seconds_search"] = time.time() - t0
                res["profile_full"] = objr.profile_full
                # subset on the raw-space path with the same cardinality as the latent subset
                path = {d: list(range(d))}
                cur = set(range(d))
                for h in res["history"][1:]:
                    cur = cur - {h["feature"]} if h["action"] == "remove" else cur | {h["feature"]}
                    path[len(cur)] = sorted(cur)
                res["at_latent_cardinality"] = path.get(len(S_star))
                res["budgets"] = stepwise_budgets(objr, res["selected"], cfg["budgets"])
                res["n_evals_total"] = len(objr.cache)
                _save(p_rs, res)
                mark("raw_search", t0)
            raw = _load(p_rs)
            p_rk = os.path.join(d_job, "rankings.json")
            if not os.path.exists(p_rk):
                t0 = time.time()
                _save(p_rk, rc.ranking_baselines(X_tr, pp, seed_f))
                mark("rankings", t0)
            rk = _load(p_rk)

        # ---------------- extras: clusters + proxy fidelity ----------------
        p_ex = os.path.join(d_job, "extras.json")
        if not os.path.exists(p_ex) and job["variant"] == "main":
            t0 = time.time()
            _save(p_ex, cluster_and_proxy(X_tr, X_te, X_s, enc, S_star, ls["k"], seed_f, d_job))
            mark("extras", t0)

        # ---------------- downstream evaluation ----------------
        if level != "descriptive":
            p_ev = os.path.join(d_job, "eval.json")
            ev = _load(p_ev) if os.path.exists(p_ev) else {}
            y_tr = {t: ds["y_enc"][t][tr] for t in ds["tasks"]}
            y_te = {t: ds["y_enc"][t][te] for t in ds["tasks"]}
            both = list(ds["tasks"].keys())
            plan = []  # (name, indices, classifiers, tasks)
            clf_main = ["rf", "dt"] + (["knn"] if (cfg["knn"] and level == "full") else [])
            plan.append(("proposed", S_star, clf_main, both))
            if job["variant"] == "main":
                plan.append(("all_features", list(range(d)), clf_main, both))
            if level == "full":
                k = cfg["rank_k"]
                plan.append(("kmeans_sil", raw["selected"], clf_main, both))
                for name in cfg["rank_eval"]:
                    if rk.get(name):
                        plan.append((f"{name}_{k}", rk[name][:k], clf_main, both))
                if cfg["ds17"]:
                    idx17 = [ds["features"].index(f) for f in rc.DATASENSE_SELECTED_17]
                    plan.append(("ds17", idx17, clf_main, both))
                if raw.get("at_latent_cardinality") and raw["at_latent_cardinality"] != raw["selected"]:
                    plan.append(("kmeans_sil_at_nprop", raw["at_latent_cardinality"], ["rf"], both))
                if ls["backward_only"] != S_star:
                    plan.append(("proposed_backward_only", ls["backward_only"], ["rf"], both))
                for b in (cfg["budgets"] if cfg["eval_budgets"] else ()):
                    plan.append((f"proposed_b{b}", ls["budgets"][str(b)], ["rf"], ["multiclass"]))
                    plan.append((f"kmeans_sil_b{b}", raw["budgets"][str(b)], ["rf"], ["multiclass"]))
                    for name in cfg["budget_rank"]:
                        if rk.get(name) and not (b == k and name in cfg["rank_eval"]):
                            plan.append((f"{name}_{b}", rk[name][:b], ["rf"], ["multiclass"]))
            if level == "ablation":
                for b in (cfg["budgets"][0], cfg["budgets"][-1]):
                    plan.append((f"proposed_b{b}", ls["budgets"][str(b)], ["rf"], ["multiclass"]))
            for name, idx, clfs, tasks in plan:
                if name in ev:
                    continue
                t0 = time.time()
                r = rc.evaluate_subset(X_tr, X_te, {t: y_tr[t] for t in tasks}, {t: y_te[t] for t in tasks},
                                       idx, clfs, seed_f, per_class_tasks=("multiclass",))
                ev[name] = dict(n_features=len(idx), features=[int(i) for i in idx], results=r,
                                seconds=time.time() - t0)
                _save(p_ev, ev)
                log(f"    eval {name} ({len(idx)}f) {time.time() - t0:.0f}s")
            mark("eval", time.time())
        _save(os.path.join(d_job, "DONE.json"), dict(job=job, finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                                                      n_train=int(len(tr)), n_test=int(len(te))))
        return f"OK {tag}"
    except Exception:
        err = traceback.format_exc()
        with open(os.path.join(d_job, "ERROR.txt"), "w", encoding="utf-8") as f:
            f.write(err)
        return f"FAIL {tag}\n{err}"


def cluster_and_proxy(X_tr, X_te, X_s, enc, S_star, k, seed, d_job) -> dict:
    """Cluster stability inside the fold and in-sample / held-out fidelity of the RF proxy."""
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, adjusted_mutual_info_score, adjusted_rand_score,
                                 balanced_accuracy_score, f1_score, recall_score, silhouette_score)

    S = np.array(S_star)
    rng = np.random.default_rng(seed + 7)
    n = len(X_tr)
    rows = rng.choice(n, size=min(20_000, n), replace=False)
    H = rc.encode(enc, X_tr[rows], S)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(H)
    lab = km.labels_
    np.savez(os.path.join(d_job, "clusters.npz"), centers=km.cluster_centers_)
    out = dict(k=int(k), cluster_sizes=np.bincount(lab, minlength=k).tolist(),
               silhouette=rc.sampled_silhouette(H, lab, seed))

    # --- cluster stability: refit on disjoint-seeded resamples, compare on a fixed reference set
    H_ref = rc.encode(enc, X_s, S)
    ref = [km.predict(H_ref)]
    for b in range(4):
        rb = rng.choice(n, size=min(rc.HP["search_rows"], n), replace=False)
        kb = KMeans(n_clusters=k, n_init=rc.HP["kmeans_n_init"], random_state=seed + 100 + b).fit(
            rc.encode(enc, X_tr[rb], S))
        ref.append(kb.predict(H_ref))
    ari = [adjusted_rand_score(ref[i], ref[j]) for i in range(len(ref)) for j in range(i + 1, len(ref))]
    ami = [adjusted_mutual_info_score(ref[i], ref[j]) for i in range(len(ref)) for j in range(i + 1, len(ref))]
    out["stability_resample"] = dict(ari_mean=float(np.mean(ari)), ari_min=float(np.min(ari)),
                                     ami_mean=float(np.mean(ami)), ami_min=float(np.min(ami)), n_pairs=len(ari))

    # --- proxy: cluster-balanced training sample, disjoint internal hold-out, outer test fold
    # at most 5000 // k rows per cluster, and never more than 70% of a cluster, so that
    # every cluster keeps disjoint rows for the internal hold-out
    per = max(1, 5000 // k)
    tr_idx = np.concatenate([
        rng.permutation(np.where(lab == c)[0])[:max(1, min(per, int(0.7 * np.sum(lab == c))))]
        for c in range(k)])
    mask = np.ones(len(rows), bool)
    mask[tr_idx] = False
    ho_idx = np.where(mask)[0]
    Xp = X_tr[rows][:, S]
    proxy = RandomForestClassifier(n_estimators=rc.HP["proxy_trees"], max_depth=rc.HP["proxy_depth"],
                                   random_state=seed, n_jobs=1).fit(Xp[tr_idx], lab[tr_idx])

    def fid(Xq, yq):
        p = proxy.predict(Xq)
        labels = np.arange(k)
        present = np.unique(yq)
        return dict(n=int(len(yq)), accuracy=float(accuracy_score(yq, p)),
                    balanced_accuracy=float(balanced_accuracy_score(yq, p)),
                    macro_f1=float(f1_score(yq, p, average="macro", labels=present, zero_division=0)),
                    n_clusters_present=int(len(present)),
                    per_cluster_recall=[float(v) for v in recall_score(yq, p, average=None, labels=labels, zero_division=0)],
                    per_cluster_f1=[float(v) for v in f1_score(yq, p, average=None, labels=labels, zero_division=0)],
                    per_cluster_n=np.bincount(yq, minlength=k).tolist())

    out["proxy_train_rows"] = int(len(tr_idx))
    out["proxy_in_sample"] = fid(Xp[tr_idx], lab[tr_idx])
    out["proxy_heldout_internal"] = fid(Xp[ho_idx], lab[ho_idx])
    if len(X_te):
        lab_te = km.predict(rc.encode(enc, X_te, S))
        out["proxy_heldout_test_fold"] = fid(X_te[:, S], lab_te)
    return out


# --------------------------------------------------------------------------- #
# Plans
# --------------------------------------------------------------------------- #
def J(dataset, protocol, n_splits, seed, fold, level, variant="main"):
    return dict(dataset=dataset, protocol=protocol, n_splits=n_splits, seed=seed, fold=fold,
                level=level, variant=variant)


def build_plan(name: str) -> list[dict]:
    """Jobs in priority order (the queue is processed in this order)."""
    jobs = []
    if name in ("wustl_log", "main"):          # 1. external dataset (log-scaled attributes)
        jobs += [J("wustl_log", "sample", 10, 42, f, "full") for f in range(10)]
    if name in ("datasense42", "main"):        # 2. primary DataSense execution + descriptive configuration
        jobs += [J("datasense", "sample", 10, 42, f, "full") for f in range(10)]
        jobs += [J("datasense", "fulldata", 1, 42, 0, "descriptive")]
    if name in ("seeds", "main"):              # 3. independent master seeds
        for s in (43, 44):
            jobs += [J("datasense", "sample", 10, s, f, "seed") for f in range(10)]
    if name in ("grouped", "main"):            # 4. grouped protocols
        jobs += [J("datasense", "group_exec", 5, 42, f, "seed") for f in range(5)]
        jobs += [J("datasense", "group_device", 5, 42, f, "seed") for f in range(5)]
    if name in ("ablation", "main"):           # 5. controlled augmentation comparison
        jobs += [J("datasense", "sample", 10, 42, f, "ablation", "aug_independent") for f in range(5)]
    if name in ("wustl", "main"):              # 6. unchanged pipeline on WUSTL (direct reproduction)
        jobs += [J("wustl", "sample", 10, 42, f, "full") for f in range(10)]
        jobs += [J("wustl_log", "fulldata", 1, 42, 0, "descriptive")]
    if name in ("nbaiot",):                   # third dataset (pre-registered, see reports/02_*)
        jobs += [J("nbaiot_log", "sample", 10, 42, f, "full") for f in range(10)]
    if name in ("ablation_none",):             # optional: encoder trained without masking
        jobs += [J("datasense", "sample", 10, 42, f, "ablation", "aug_none") for f in range(5)]
    if name == "smoke":
        jobs += [J("datasense", "sample", 10, 42, 0, "full")]
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="main")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    jobs = [j for j in build_plan(args.plan) if not os.path.exists(os.path.join(job_dir(j), "DONE.json"))]
    print(f"[run_cv] plan={args.plan} pending jobs={len(jobs)} workers={args.workers}", flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_job, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            print(f"[{(time.time() - t0) / 60:6.1f} min] ({i}/{len(jobs)}) {f.result()}", flush=True)
    with open(os.path.join(LOGS, f"FINISHED_{args.plan}.flag"), "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
    print("[run_cv] all jobs finished", flush=True)


if __name__ == "__main__":
    main()
