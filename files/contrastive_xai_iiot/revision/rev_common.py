"""
rev_common.py — Shared building blocks for the second-round revision experiments.

Everything in this module is fold-internal by construction: every function that
learns something (scaler, encoder, k*, feature subset, clusters, proxy) receives
only the training partition of the current fold. Labels are never passed to the
selection functions; they are used only by `evaluate_subsets`.

Implemented components
  - dataset loaders with on-disk numeric caches (DataSense, WUSTL-IIoT-2021)
  - fold-wise preprocessing (NaN -> 0, +/-inf -> training max, StandardScaler)
  - vectorised group-aware augmentation and VICReg training
  - NumPy forward pass of the frozen encoder (used by the subset search)
  - bidirectional latent-space Silhouette search (natural operating point)
  - forced backward elimination to a fixed budget (matched-cardinality regime)
  - raw-space k-Means+Sil wrapper and ranking baselines
  - downstream evaluation (RF / DT / KNN) for several label columns at once
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import CONTEXT_GROUPS, DATASENSE_SELECTED_17, LIST_COLUMNS, META_COLUMNS  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "data", "cache")
REV_DIR = os.path.join(ROOT, "results", "revision")

# --------------------------------------------------------------------------- #
# Hyperparameters (identical to Table 7 of the manuscript / config.py)
# --------------------------------------------------------------------------- #
HP = dict(
    hidden_dim=256,
    latent_dim=128,
    proj_dim=512,
    enc_dropout=0.2,
    epochs=100,
    batch_size=512,
    lr=1e-3,
    weight_decay=1e-4,
    eta_min=1e-5,
    grad_clip=1.0,
    vic_lambda=25.0,
    vic_mu=25.0,
    vic_nu=1.0,
    vic_eps=1e-4,
    max_masked_groups=2,       # m ~ Uniform{0, 1, 2}
    noise_std=0.05,            # standardized units
    feat_dropout=0.10,
    encoder_max_rows=20_480,   # label-free uniform subsample of the training fold (40 steps/epoch)
    k_range=tuple(range(2, 15)),
    search_rows=5_000,         # training-fold rows used by k-Means during the search
    sil_rows=2_000,            # rows used to estimate the Silhouette score
    kmeans_n_init=5,
    max_search_iter=200,
    proxy_trees=200,
    proxy_depth=10,
    graph_rows=5_000,
)

if os.environ.get("REV_SMOKE"):  # tiny configuration used only to exercise the code path
    HP.update(epochs=2, max_search_iter=1)
    REV_DIR = os.path.join(ROOT, "results", "revision_smoke")


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
WUSTL_DROP = ["StartTime", "LastTime", "SrcAddr", "DstAddr", "sIpId", "dIpId"]

# Context Groups rebuilt for WUSTL-IIoT-2021 from the measurement semantics of the
# Argus flow attributes (no labels, class frequencies or classifier scores used).
WUSTL_CONTEXT_GROUPS = {
    "Packet/Byte Volume": ["SrcPkts", "DstPkts", "TotPkts", "SrcBytes", "DstBytes", "TotBytes"],
    "Application Payload": ["SAppBytes", "DAppBytes", "TotAppByte"],
    "Traffic Rate and Load": ["SrcLoad", "DstLoad", "Load", "SrcRate", "DstRate", "Rate"],
    "Loss and Retransmission": ["SrcLoss", "DstLoss", "Loss", "pLoss"],
    "Inter-Packet Timing and Jitter": ["SrcJitter", "DstJitter", "SIntPkt", "DIntPkt",
                                       "SrcJitAct", "DstJitAct"],
    "Flow Duration and Activity": ["Dur", "RunTime", "Mean", "Sum", "Min", "Max", "IdleTime"],
    "Connection Setup": ["TcpRtt", "SynAck"],
    "Network Multiplexing": ["Proto", "Sport", "Dport"],
    "IP Header Control": ["sTtl", "dTtl", "sTos", "sDSb"],
}


def load_datasense() -> dict:
    """Returns dict(X_raw float32 [n,71], features, meta DataFrame)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    npy = os.path.join(CACHE_DIR, "datasense_X.npy")
    meta_p = os.path.join(CACHE_DIR, "datasense_meta.pkl")
    feat_p = os.path.join(CACHE_DIR, "datasense_features.json")
    if not (os.path.exists(npy) and os.path.exists(meta_p) and os.path.exists(feat_p)):
        csv = os.path.join(ROOT, "data", "dataset_attack_benign.csv")
        head = pd.read_csv(csv, nrows=5, low_memory=False)
        drop = set(META_COLUMNS + LIST_COLUMNS)
        num_cols = [c for c in head.columns
                    if c not in drop and pd.api.types.is_numeric_dtype(head[c])]
        meta_cols = [c for c in ["device_name", "label_full", "label1", "label2", "label3",
                                 "label4", "timestamp_start"] if c in head.columns]
        df = pd.read_csv(csv, usecols=num_cols + meta_cols, low_memory=False)
        X = df[num_cols].to_numpy(dtype=np.float32)
        np.save(npy, X)
        df[meta_cols].to_pickle(meta_p)
        with open(feat_p, "w", encoding="utf-8") as f:
            json.dump(num_cols, f)
    X = np.load(npy)
    meta = pd.read_pickle(meta_p)
    with open(feat_p, encoding="utf-8") as f:
        features = json.load(f)
    return dict(name="datasense", X_raw=X, features=features, meta=meta,
                groups=CONTEXT_GROUPS, tasks={"binary": "label1", "multiclass": "label2"},
                strat_col="label2")


def load_wustl(frac: float = 0.25, seed: int = 2021) -> dict:
    """WUSTL-IIoT-2021. A label-free uniform row sample of size `frac` is drawn once."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    tag = f"wustl_f{int(frac * 100)}_s{seed}"
    npy = os.path.join(CACHE_DIR, f"{tag}_X.npy")
    meta_p = os.path.join(CACHE_DIR, f"{tag}_meta.pkl")
    feat_p = os.path.join(CACHE_DIR, f"{tag}_features.json")
    if not (os.path.exists(npy) and os.path.exists(meta_p) and os.path.exists(feat_p)):
        csv = os.path.join(ROOT, "data", "external", "wustl_iiot_2021", "wustl_iiot_2021.csv")
        df = pd.read_csv(csv, low_memory=False)
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(df), size=int(round(frac * len(df))), replace=False))
        df = df.iloc[idx].reset_index(drop=True)
        features = [c for c in df.columns if c not in WUSTL_DROP + ["Traffic", "Target"]]
        np.save(npy, df[features].to_numpy(dtype=np.float32))
        meta = pd.DataFrame({
            "label1": np.where(df["Target"].to_numpy() == 1, "attack", "normal"),
            "label2": df["Traffic"].astype(str).to_numpy(),
            "timestamp_start": df["StartTime"].astype(str).to_numpy(),
        })
        meta.to_pickle(meta_p)
        with open(feat_p, "w", encoding="utf-8") as f:
            json.dump(features, f)
    X = np.load(npy)
    meta = pd.read_pickle(meta_p)
    with open(feat_p, encoding="utf-8") as f:
        features = json.load(f)
    return dict(name="wustl", X_raw=X, features=features, meta=meta,
                groups=WUSTL_CONTEXT_GROUPS, tasks={"binary": "label1", "multiclass": "label2"},
                strat_col="label2")


def load_nbaiot_log() -> dict:
    """N-BaIoT: label-free 3.5% uniform sample (seed 2018), signed log transform, 5 stream-type groups."""
    tag = "nbaiot_f35_s2018"
    X = np.load(os.path.join(CACHE_DIR, f"{tag}_X.npy")).astype(np.float64)
    X = (np.sign(X) * np.log1p(np.abs(X))).astype(np.float32)
    meta = pd.read_pickle(os.path.join(CACHE_DIR, f"{tag}_meta.pkl"))
    with open(os.path.join(CACHE_DIR, f"{tag}_features.json"), encoding="utf-8") as f:
        features = json.load(f)
    groups = {}
    for c in features:
        prefix = c.split("_L")[0]          # MI_dir, H, HH, HH_jit, HpHp
        groups.setdefault(prefix, []).append(c)
    return dict(name="nbaiot_log", X_raw=X, features=features, meta=meta, groups=groups,
                tasks={"binary": "label1", "multiclass": "label2"}, strat_col="label2")


def load_dataset(name: str) -> dict:
    if name == "datasense":
        return load_datasense()
    if name == "wustl":
        return load_wustl()
    if name == "nbaiot_log":
        return load_nbaiot_log()
    if name == "wustl_log":
        # Methodological adaptation for heavy-tailed flow attributes: a stateless, label-free
        # signed log transform x -> sign(x) * log(1 + |x|) applied to every attribute before
        # the (fold-wise) standardization. Nothing else changes.
        ds = load_wustl()
        X = ds["X_raw"].astype(np.float64)
        ds["X_raw"] = (np.sign(X) * np.log1p(np.abs(X))).astype(np.float32)
        ds["name"] = "wustl_log"
        return ds
    raise ValueError(name)


def group_index_vector(features: list[str], groups: dict[str, list[str]]) -> tuple[np.ndarray, list[str]]:
    """Maps every feature to exactly one Context Group id. Fails if a feature is unassigned."""
    name_to_g = {}
    names = list(groups.keys())
    for gi, g in enumerate(names):
        for f in groups[g]:
            if f in name_to_g:
                raise ValueError(f"feature {f} assigned to two groups")
            name_to_g[f] = gi
    missing = [f for f in features if f not in name_to_g]
    if missing:
        raise ValueError(f"features without Context Group: {missing}")
    return np.array([name_to_g[f] for f in features], dtype=np.int64), names


# --------------------------------------------------------------------------- #
# Fold construction
# --------------------------------------------------------------------------- #
def make_folds(ds: dict, protocol: str, seed: int, n_splits: int = 10):
    """
    protocol:
      sample      — StratifiedKFold on the multiclass label (sample-level, IID).
      group_exec  — StratifiedGroupKFold; a group is one attack execution
                    (`label_full`) or, for benign traffic, one contiguous 60-s
                    capture block shared by all devices (DataSense only).
      group_device— StratifiedGroupKFold with device identity as group (DataSense only).
      group_time  — StratifiedGroupKFold with contiguous time blocks as groups (WUSTL).
    Labels are used here only to stratify the outer evaluation folds.
    """
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    meta = ds["meta"]
    y = meta[ds["strat_col"]].to_numpy()
    n = len(y)
    if protocol == "sample":
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(skf.split(np.zeros(n), y))
    if protocol == "group_exec":
        t = pd.to_datetime(meta["timestamp_start"])
        block = (t.astype("int64") // (60 * 10**9)).astype(str)
        g = np.where(meta["label1"].to_numpy() == "benign",
                     "benign_block_" + block.to_numpy(), meta["label_full"].to_numpy())
    elif protocol == "group_device":
        g = meta["device_name"].to_numpy()
    elif protocol == "group_time":
        t = pd.to_datetime(meta["timestamp_start"])
        g = (t.astype("int64") // (300 * 10**9)).to_numpy()
    else:
        raise ValueError(protocol)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(n), y, groups=g))


# --------------------------------------------------------------------------- #
# Fold-wise preprocessing
# --------------------------------------------------------------------------- #
def fit_preprocess(X_tr_raw: np.ndarray) -> dict:
    X = X_tr_raw.astype(np.float64, copy=True)
    X[np.isnan(X)] = 0.0
    finite = np.where(np.isfinite(X), X, np.nan)
    fmax = np.nanmax(finite, axis=0)
    fmax = np.where(np.isfinite(fmax), fmax, 0.0)
    bad = ~np.isfinite(X)
    if bad.any():
        X[bad] = np.broadcast_to(fmax, X.shape)[bad]
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std_safe = np.where(std > 0, std, 1.0)
    return dict(fmax=fmax, mean=mean, std=std_safe, raw_std=std,
                raw_min=X.min(axis=0), raw_max=X.max(axis=0))


def apply_preprocess(X_raw: np.ndarray, pp: dict) -> np.ndarray:
    X = X_raw.astype(np.float64, copy=True)
    X[np.isnan(X)] = 0.0
    bad = ~np.isfinite(X)
    if bad.any():
        X[bad] = np.broadcast_to(pp["fmax"], X.shape)[bad]
    return ((X - pp["mean"]) / pp["std"]).astype(np.float32)


# --------------------------------------------------------------------------- #
# VICReg encoder
# --------------------------------------------------------------------------- #
def _torch():
    import torch
    return torch


def build_encoder(input_dim: int):
    torch = _torch()
    nn = torch.nn

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, HP["hidden_dim"]), nn.BatchNorm1d(HP["hidden_dim"]), nn.ReLU(),
                nn.Dropout(HP["enc_dropout"]),
                nn.Linear(HP["hidden_dim"], HP["latent_dim"]), nn.BatchNorm1d(HP["latent_dim"]), nn.ReLU(),
            )
            self.projector = nn.Sequential(
                nn.Linear(HP["latent_dim"], HP["proj_dim"]), nn.BatchNorm1d(HP["proj_dim"]), nn.ReLU(),
                nn.Linear(HP["proj_dim"], HP["proj_dim"]),
            )

        def forward(self, x):
            return self.encoder(x)

    return Encoder()


def vicreg_loss(z, zp):
    torch = _torch()
    F = torch.nn.functional
    b, r = z.shape
    inv = F.mse_loss(z, zp)

    def var_term(x):
        return torch.mean(F.relu(1.0 - torch.sqrt(x.var(dim=0) + HP["vic_eps"])))

    def cov_term(x):
        xc = x - x.mean(dim=0)
        cov = (xc.T @ xc) / (b - 1)
        off = cov.pow(2)
        off.fill_diagonal_(0)
        return off.sum() / r

    var = 0.5 * (var_term(z) + var_term(zp))
    cov = 0.5 * (cov_term(z) + cov_term(zp))
    return HP["vic_lambda"] * inv + HP["vic_mu"] * var + HP["vic_nu"] * cov


def augment(x, group_idx, n_groups, gen, mode: str = "group"):
    """
    mode = 'group'       : Context-Group masking (m ~ U{0,1,2}) + noise + feature dropout (proposed)
    mode = 'independent' : independent per-feature masking with the same expected masked
                           fraction as 'group' + noise + feature dropout
    mode = 'none'        : noise + feature dropout only (no structured masking)
    """
    torch = _torch()
    b, d = x.shape
    noise = torch.randn(b, d, generator=gen) * HP["noise_std"]
    keep = (torch.rand(b, d, generator=gen) >= HP["feat_dropout"]).to(x.dtype)
    out = (x + noise) * keep
    if mode == "group":
        m = torch.randint(0, HP["max_masked_groups"] + 1, (b,), generator=gen)
        ranks = torch.rand(b, n_groups, generator=gen).argsort(dim=1).argsort(dim=1)
        gmask = ranks < m[:, None]
        out = out.masked_fill(gmask[:, group_idx], 0.0)
    elif mode == "independent":
        # expected masked fraction of group masking: E[m] / n_groups (groups drawn uniformly)
        p = (HP["max_masked_groups"] / 2.0) / n_groups
        out = out.masked_fill(torch.rand(b, d, generator=gen) < p, 0.0)
    elif mode != "none":
        raise ValueError(mode)
    return out


def train_encoder(X_tr: np.ndarray, group_idx: np.ndarray, n_groups: int, seed: int,
                  aug_mode: str = "group", epochs: int | None = None, log=None) -> tuple[object, list[float]]:
    torch = _torch()
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(X_tr)
    if n > HP["encoder_max_rows"]:
        rows = rng.choice(n, size=HP["encoder_max_rows"], replace=False)
        X_fit = X_tr[rows]
    else:
        X_fit = X_tr
    Xt = torch.from_numpy(np.ascontiguousarray(X_fit, dtype=np.float32))
    gidx = torch.from_numpy(group_idx)
    model = build_encoder(X_tr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    n_epochs = epochs or HP["epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=HP["eta_min"])
    bs = HP["batch_size"]
    n_fit = len(Xt)
    steps = n_fit // bs
    hist = []
    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(n_fit, generator=gen)
        tot = 0.0
        for s in range(steps):
            xb = Xt[perm[s * bs:(s + 1) * bs]]
            va = augment(xb, gidx, n_groups, gen, aug_mode)
            vb = augment(xb, gidx, n_groups, gen, aug_mode)
            za = model.projector(model.encoder(va))
            zb = model.projector(model.encoder(vb))
            loss = vicreg_loss(za, zb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), HP["grad_clip"])
            opt.step()
            tot += float(loss.detach())
        sched.step()
        hist.append(tot / max(steps, 1))
        if log and (ep == 0 or (ep + 1) % 20 == 0):
            log(f"    epoch {ep + 1}/{n_epochs} loss {hist[-1]:.4f}")
    model.eval()
    return model, hist


def encoder_to_numpy(model) -> dict:
    """Frozen eval-mode encoder as plain arrays (Linear -> BN -> ReLU -> Linear -> BN -> ReLU)."""
    enc = model.encoder
    l1, bn1, l2, bn2 = enc[0], enc[1], enc[4], enc[5]

    def bn_affine(bn):
        scale = (bn.weight / (bn.running_var + bn.eps).sqrt()).detach().numpy()
        shift = (bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps).sqrt()).detach().numpy()
        return scale.astype(np.float32), shift.astype(np.float32)

    s1, t1 = bn_affine(bn1)
    s2, t2 = bn_affine(bn2)
    return dict(W1=l1.weight.detach().numpy().T.copy(), b1=l1.bias.detach().numpy().copy(), s1=s1, t1=t1,
                W2=l2.weight.detach().numpy().T.copy(), b2=l2.bias.detach().numpy().copy(), s2=s2, t2=t2)


def encode(enc: dict, X: np.ndarray, subset: np.ndarray | None = None) -> np.ndarray:
    """h = f_theta(M_S(x)); unselected features are set to 0 (the training-fold mean)."""
    if subset is not None:
        Xm = np.zeros_like(X)
        Xm[:, subset] = X[:, subset]
    else:
        Xm = X
    h = np.maximum((Xm @ enc["W1"] + enc["b1"]) * enc["s1"] + enc["t1"], 0.0)
    h = np.maximum((h @ enc["W2"] + enc["b2"]) * enc["s2"] + enc["t2"], 0.0)
    return h.astype(np.float32)


# --------------------------------------------------------------------------- #
# Silhouette objective and searches
# --------------------------------------------------------------------------- #
def kmeans_sil(H: np.ndarray, k: int, seed: int, n_init: int | None = None) -> float:
    """Returns -1.0 for invalid configurations (fewer than two non-empty clusters)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km = KMeans(n_clusters=k, n_init=n_init or HP["kmeans_n_init"], random_state=seed)
        lab = km.fit_predict(H)
    return sampled_silhouette(H, lab, seed)


def sampled_silhouette(H: np.ndarray, lab: np.ndarray, seed: int) -> float:
    """
    Silhouette on a fixed random row sample (identical to scikit-learn's `sample_size`
    sampling). A partition whose evaluation sample contains fewer than two clusters has
    an undefined Silhouette and is rejected with the score -1.0.
    """
    from sklearn.metrics import silhouette_score
    from sklearn.utils import check_random_state

    m = min(HP["sil_rows"], len(H))
    idx = check_random_state(seed).permutation(len(H))[:m]
    if len(np.unique(lab[idx])) < 2:
        return -1.0
    return float(silhouette_score(H[idx], lab[idx]))


def silhouette_profile(H: np.ndarray, seed: int) -> dict[int, float]:
    return {int(k): kmeans_sil(H, k, seed) for k in HP["k_range"]}


class SubsetObjective:
    """J(S) for a fixed k*. `space='latent'` evaluates f_theta(M_S(X)); 'raw' evaluates X[:, S]."""

    def __init__(self, X_sub: np.ndarray, seed: int, space: str, enc: dict | None = None):
        self.X = X_sub
        self.seed = seed
        self.space = space
        self.enc = enc
        self.cache: dict[frozenset, float] = {}
        H0 = self._rep(np.arange(X_sub.shape[1]))
        self.profile_full = silhouette_profile(H0, seed)
        self.k = max(self.profile_full, key=lambda k: self.profile_full[k])
        self.cache[frozenset(range(X_sub.shape[1]))] = self.profile_full[self.k]

    def _rep(self, idx: np.ndarray) -> np.ndarray:
        if self.space == "latent":
            return encode(self.enc, self.X, idx)
        return np.ascontiguousarray(self.X[:, idx])

    def __call__(self, S: frozenset) -> float:
        if S not in self.cache:
            self.cache[S] = kmeans_sil(self._rep(np.array(sorted(S))), self.k, self.seed)
        return self.cache[S]

    def profile(self, S) -> dict[int, float]:
        return silhouette_profile(self._rep(np.array(sorted(S))), self.seed)


def bidirectional_search(obj: SubsetObjective, d: int, log=None) -> dict:
    """
    Algorithm 2: backward-removal pass followed by forward-recovery pass, repeated
    until neither pass yields a strict improvement. Starts from the complete set.
    """
    S = frozenset(range(d))
    J = obj(S)
    history = [dict(step=0, action="init", feature=None, n=len(S), J=J)]
    it = 0
    improved = True
    while improved and it < HP["max_search_iter"]:
        improved = False
        it += 1
        # backward pass
        best, bestJ = None, J
        if len(S) > 2:
            for j in sorted(S):
                v = obj(S - {j})
                if v > bestJ:
                    best, bestJ = j, v
        if best is not None:
            S, J, improved = S - {best}, bestJ, True
            history.append(dict(step=it, action="remove", feature=int(best), n=len(S), J=J))
        # forward-recovery pass
        best, bestJ = None, J
        for j in sorted(set(range(d)) - S):
            v = obj(S | {j})
            if v > bestJ:
                best, bestJ = j, v
        if best is not None:
            S, J, improved = S | {best}, bestJ, True
            history.append(dict(step=it, action="add", feature=int(best), n=len(S), J=J))
        if log:
            log(f"    search it {it}: |S|={len(S)} J={J:.4f} evals={len(obj.cache)}")
    return dict(selected=sorted(int(i) for i in S), J=float(J), k=int(obj.k), history=history,
                n_evals=len(obj.cache))


def backward_only_search(obj: SubsetObjective, d: int) -> dict:
    """Ablation variant: Algorithm 2 without the forward-recovery pass."""
    S = frozenset(range(d))
    J = obj(S)
    while len(S) > 2:
        best, bestJ = None, J
        for j in sorted(S):
            v = obj(S - {j})
            if v > bestJ:
                best, bestJ = j, v
        if best is None:
            break
        S, J = S - {best}, bestJ
    return dict(selected=sorted(int(i) for i in S), J=float(J), k=int(obj.k), n_evals=len(obj.cache))


# --------------------------------------------------------------------------- #
# Ranking baselines (same definitions as src/baselines.py; training fold only)
# --------------------------------------------------------------------------- #
def ranking_baselines(X_tr: np.ndarray, pp: dict, seed: int, which=None) -> dict[str, list[int]]:
    from src.baselines import LaplacianScoreFS, MCFSFS, NDFSFS, SPECFS, UDFSFS

    out = {}
    which = which or ["variance", "laplacian", "spec", "mcfs", "udfs", "ndfs"]
    if "variance" in which:
        # Variance is degenerate after z-scoring (all ones); it is therefore computed on
        # the training-fold min-max scaled features.
        rng_ = np.where(pp["raw_max"] > pp["raw_min"], pp["raw_max"] - pp["raw_min"], 1.0)
        var01 = (pp["raw_std"] / rng_) ** 2
        out["variance"] = [int(i) for i in np.argsort(var01)[::-1]]
    makers = {
        "laplacian": lambda: LaplacianScoreFS(sample_size=HP["graph_rows"], random_state=seed),
        "spec": lambda: SPECFS(sample_size=HP["graph_rows"], random_state=seed),
        "mcfs": lambda: MCFSFS(n_clusters=5, sample_size=HP["graph_rows"], random_state=seed),
        "udfs": lambda: UDFSFS(sample_size=HP["graph_rows"], random_state=seed),
        "ndfs": lambda: NDFSFS(sample_size=HP["graph_rows"], random_state=seed),
    }
    import warnings
    for name in which:
        if name == "variance":
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = makers[name]().fit(X_tr)
            sc = np.nan_to_num(np.asarray(m.scores_, dtype=np.float64), nan=-np.inf, posinf=np.inf, neginf=-np.inf)
            out[name] = [int(i) for i in np.argsort(sc, kind="stable")[::-1]]
        except Exception as e:  # pragma: no cover
            out[name] = None
            out[f"{name}_error"] = repr(e)
    return out


# --------------------------------------------------------------------------- #
# Downstream evaluation
# --------------------------------------------------------------------------- #
def _metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                                 precision_score, recall_score)
    return dict(
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision_macro=float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        recall_macro=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        mcc=float(matthews_corrcoef(y_true, y_pred)),
    )


def evaluate_subset(X_tr, X_te, y_tr: dict, y_te: dict, idx, classifiers, seed: int,
                    per_class_tasks=()) -> dict:
    """
    y_tr / y_te: {task: integer-encoded labels}. Returns {task: {clf: metrics}}.
    KNN neighbours are computed once per subset and reused for every task.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    from sklearn.neighbors import NearestNeighbors
    from sklearn.tree import DecisionTreeClassifier

    idx = np.asarray(idx, dtype=int)
    A = np.ascontiguousarray(X_tr[:, idx])
    B = np.ascontiguousarray(X_te[:, idx])
    res = {t: {} for t in y_tr}
    nb = None
    if "knn" in classifiers:
        nb = NearestNeighbors(n_neighbors=5, algorithm="brute", n_jobs=1).fit(A).kneighbors(B, return_distance=False)
    for t in y_tr:
        labels = np.unique(np.concatenate([y_tr[t], y_te[t]]))
        for c in classifiers:
            if c == "rf":
                pred = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1).fit(A, y_tr[t]).predict(B)
            elif c == "dt":
                pred = DecisionTreeClassifier(random_state=seed).fit(A, y_tr[t]).predict(B)
            elif c == "knn":
                votes = y_tr[t][nb]
                n_cls = int(max(y_tr[t].max(), y_te[t].max())) + 1
                counts = np.zeros((len(votes), n_cls), dtype=np.int16)
                for col in range(votes.shape[1]):
                    np.add.at(counts, (np.arange(len(votes)), votes[:, col]), 1)
                pred = counts.argmax(axis=1)
            else:
                raise ValueError(c)
            m = _metrics(y_te[t], pred)
            if t in per_class_tasks and c == "rf":
                m["per_class_f1"] = [float(v) for v in
                                     f1_score(y_te[t], pred, average=None, labels=labels, zero_division=0)]
            res[t][c] = m
    return res


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 1.0


class Logger:
    def __init__(self, path: str | None):
        self.path = path
        self.t0 = time.time()

    def __call__(self, msg: str):
        line = f"[{time.time() - self.t0:7.0f}s] {msg}"
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        else:
            print(line, flush=True)
