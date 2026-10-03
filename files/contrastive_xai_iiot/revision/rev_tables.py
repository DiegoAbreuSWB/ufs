r"""
rev_tables.py — Turns results/revision/tables/*.csv into the LaTeX tables of the manuscript
(revision/manuscript/tables/*.tex) and into numeric macros (revision/manuscript/numbers_auto.tex),
so that no number in the paper is copied by hand.
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
OUT = os.path.join(HERE, "manuscript", "tables")
os.makedirs(OUT, exist_ok=True)
MACROS: dict[str, str] = {}
PRE = {"datasense": "DS", "wustl": "WU", "wustl_log": "WL", "nbaiot_log": "NB"}


def rd(name):
    p = os.path.join(TAB, f"{name}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def f3(v, nd=3):
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def lead(v, nd=3):
    """0.936 -> .936 (compact style used in the result tables)."""
    s = f3(v, nd)
    return s[1:] if s.startswith("0.") else s


def write(name, body):
    with open(os.path.join(OUT, f"{name}.tex"), "w", encoding="utf-8") as f:
        f.write(body)
    print(f"[tables] {name}.tex")


def macro(name, value):
    MACROS[name] = str(value)


LABEL = {"proposed": r"\method", "kmeans_sil": r"$k$-Means+Sil", "all_features": "All Features",
         "ds17": r"DS-17$^{\dagger}$", "mcfs": "MCFS", "variance": "Variance", "spec": "SPEC",
         "laplacian": "Laplacian", "ndfs": "NDFS", "udfs": "UDFS"}


def base(m):
    return m.rsplit("_", 1)[0] if m.rsplit("_", 1)[-1].isdigit() else m


# --------------------------------------------------------------------------- #
def tab_natural(ds, label, caption, knn):
    d = rd(f"natural_{ds}")
    if d is None:
        return
    rows = []
    for _, r in d.iterrows():
        b = base(r.method)
        nf = f"{r.n_features_mean:.0f}" if r.n_features_min == r.n_features_max else \
            f"{r.n_features_mean:.1f} ({int(r.n_features_min)}--{int(r.n_features_max)})"
        rows.append(f"{LABEL[b]} & {nf} & {lead(r.binary_f1_mean)}\\,{{\\scriptsize$\\pm${lead(r.binary_f1_std)}}} & "
                    f"{lead(r.multiclass_f1_mean)}\\,{{\\scriptsize$\\pm${lead(r.multiclass_f1_std)}}} & "
                    f"{lead(r.acc)} & {lead(r.mcc)} & {lead(r.recall)} \\\\")
        if b in ("kmeans_sil", "udfs"):
            pass
    # separator before the two reference configurations
    idx = [i for i, r in enumerate(d.method) if r == "all_features"]
    if idx:
        rows.insert(idx[0], "\\midrule")
    clf = "RF/DT/KNN" if knn else "RF/DT"
    mc = {"datasense": "8-cl.", "nbaiot_log": "11-cl."}.get(ds, "5-cl.")
    n = int(d.n_folds.min())
    body = f"""\\begin{{table}}[t]
\\revon
\\caption{{{caption} (Mean$\\pm$Std Over {n} Folds, Master Seed 42, Average of {clf})}}
\\label{{{label}}}
\\centering
\\footnotesize
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{@{{}}llccccc@{{}}}}
\\toprule
Method & \\#F & Bin.\\ F1 & {mc} F1 & Acc. & MCC & Recall \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\multicolumn{{7}}{{@{{}}p{{0.97\\columnwidth}}@{{}}}}{{\\#F: mean number of selected features over the folds (range in parentheses when the size varies). Acc., MCC, and Recall refer to the {mc.replace('.', '')}ass task. Different \\#F values are different operating points, not a matched-budget ranking.{' $^{\\dagger}$Supervised reference subset.' if ds == 'datasense' else ''}}}
\\end{{tabular}}
\\end{{table}}
"""
    write(f"natural_{ds}", body)
    for _, r in d.iterrows():
        b = base(r.method)
        tag = {"proposed": "Prop", "all_features": "All", "kmeans_sil": "KMS", "mcfs": "MCFS", "ds17": "DSx"}.get(b)
        if tag:
            pre = PRE[ds]
            macro(f"{pre}{tag}BinAvg", f3(r.binary_f1_mean))
            macro(f"{pre}{tag}MultiAvg", f3(r.multiclass_f1_mean))
            for clf_ in ("rf", "dt", "knn"):
                for task, tt in (("binary", "Bin"), ("multiclass", "Multi")):
                    col = f"{task}_f1_{clf_}_mean"
                    if col in d.columns and not pd.isna(r[col]):
                        macro(f"{pre}{tag}{tt}{clf_.upper()}", f3(r[col]))
            macro(f"{pre}{tag}NF", f"{r.n_features_mean:.1f}")


def tab_significance():
    s42 = rd("significance_seed42_datasense")
    sall = rd("significance_allseeds_datasense")
    if s42 is None:
        return
    rows = []
    name = {"all_features": "All Features", "ds17": "DS-17", "kmeans_sil": "$k$-Means+Sil", "mcfs_25": "MCFS-25"}

    def fmt_p(p):
        if pd.isna(p):
            return "--"
        return "$<$0.001" if p < 0.001 else f"{p:.3f}"

    for task, tl in (("binary", "Binary"), ("multiclass", "8-class")):
        for clf in ("rf", "dt"):
            for ref in ("all_features", "ds17", "kmeans_sil", "mcfs_25"):
                a = s42[(s42.task == task) & (s42.clf == clf) & (s42.reference == ref)]
                if a.empty:
                    continue
                a = a.iloc[0]
                cells = [tl, clf.upper(), f"vs. {name[ref]}", f"{a.mean_diff:+.4f}", f"{int(a.n_positive)}/{int(a.n_pairs)}", fmt_p(a.wilcoxon_p_two_sided)]
                b = sall[(sall.task == task) & (sall.clf == clf) & (sall.reference == ref)] if sall is not None else None
                if b is not None and not b.empty and b.iloc[0].n_pairs > a.n_pairs:
                    b = b.iloc[0]
                    cells += [f"{b.mean_diff:+.4f}", f"{int(b.n_positive)}/{int(b.n_pairs)}", fmt_p(b.wilcoxon_p_two_sided)]
                else:
                    cells += ["--", "--", "--"]
                rows.append(" & ".join(cells) + " \\\\")
        rows.append("\\midrule")
    rows = rows[:-1]
    body = f"""\\begin{{table}}[t]
\\revon
\\caption{{Paired Comparisons of \\method{{}} With Reference Configurations (Two-Sided Wilcoxon Signed-Rank Test Over Folds)}}
\\label{{tab:wilcoxon}}
\\centering
\\footnotesize
\\setlength{{\\tabcolsep}}{{2.5pt}}
\\begin{{tabular}}{{@{{}}lllrcrrcr@{{}}}}
\\toprule
 & & & \\multicolumn{{3}}{{c}}{{Seed 42 (10 folds)}} & \\multicolumn{{3}}{{c}}{{Seeds 42--44 (30 folds)}} \\\\
\\cmidrule(lr){{4-6}}\\cmidrule(l){{7-9}}
Task & Clf. & Reference & $\\Delta$F1 & Wins & $p$ & $\\Delta$F1 & Wins & $p$ \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\multicolumn{{9}}{{@{{}}p{{0.97\\columnwidth}}@{{}}}}{{$\\Delta$F1: mean macro-F1 of \\method{{}} minus the reference. Wins: folds in which \\method{{}} is strictly higher. The $k$-Means+Sil, MCFS-25, and DS-17 configurations were evaluated only in the primary execution.}}
\\end{{tabular}}
\\end{{table}}
"""
    write("wilcoxon", body)


def tab_seeds():
    g = rd("multiseed_by_seed")
    pooled = rd("multiseed_pooled")
    sizes = rd("selsizes_latent_datasense")
    if g is None or sizes is None:
        return
    rows = []
    seeds = sorted(g.seed.unique())

    def cell(df, method, task, clf, seed=None):
        x = df[(df.method == method) & (df.task == task) & (df.clf == clf)]
        if seed is not None:
            x = x[x.seed == seed]
        return x.iloc[0] if len(x) else None

    for s in seeds:
        sz = sizes[sizes.seed == s]
        ks = sz.k_star.value_counts().sort_index()
        cells = [str(s), f"{sz.n_selected.mean():.1f} ({int(sz.n_selected.min())}--{int(sz.n_selected.max())})",
                 f"{int(sz.k_star.min())}--{int(sz.k_star.max())}"]
        for task in ("binary", "multiclass"):
            for m in ("proposed", "all_features"):
                c = cell(g, m, task, "rf", s)
                cells.append(f"{lead(c.f1_mean)}\\,{{\\scriptsize$\\pm${lead(c.f1_std)}}}" if c is not None else "--")
        c = cell(g, "proposed", "multiclass", "dt", s)
        c2 = cell(g, "all_features", "multiclass", "dt", s)
        cells.append(lead(c.f1_mean) if c is not None else "--")
        cells.append(lead(c2.f1_mean) if c2 is not None else "--")
        rows.append(" & ".join(cells) + " \\\\")
    rows.append("\\midrule")
    cells = ["Across seeds", f"{sizes.n_selected.mean():.1f} ({int(sizes.n_selected.min())}--{int(sizes.n_selected.max())})",
             f"{int(sizes.k_star.min())}--{int(sizes.k_star.max())}"]
    for task in ("binary", "multiclass"):
        for m in ("proposed", "all_features"):
            c = cell(pooled, m, task, "rf")
            cells.append(f"{lead(c.seed_mean)}\\,{{\\scriptsize$\\pm${lead(c.seed_std, 4)}}}")
    for m in ("proposed", "all_features"):
        c = cell(pooled, m, "multiclass", "dt")
        cells.append(lead(c.seed_mean))
    rows.append(" & ".join(cells) + " \\\\")
    body = f"""\\begin{{table*}}[t]
\\revon
\\caption{{Three Independent Executions of the Complete 10-Fold Experiment (Macro-F1; Per-Seed Entries: Mean$\\pm$Std Over the 10 Folds; Last Row: Mean$\\pm$Std of the Three Seed Means)}}
\\label{{tab:seeds}}
\\centering
\\footnotesize
\\begin{{tabular}}{{@{{}}lcccccccc@{{}}}}
\\toprule
 & & & \\multicolumn{{2}}{{c}}{{Binary, RF}} & \\multicolumn{{2}}{{c}}{{8-class, RF}} & \\multicolumn{{2}}{{c}}{{8-class, DT}} \\\\
\\cmidrule(lr){{4-5}}\\cmidrule(lr){{6-7}}\\cmidrule(l){{8-9}}
Master seed & $|S^{{*}}_r|$ mean (range) & $k^{{*}}_r$ range & \\method & All Features & \\method & All Features & \\method & All Features \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""
    write("seeds", body)
    for m, tag in (("proposed", "Prop"), ("all_features", "All")):
        for task, tt in (("binary", "Bin"), ("multiclass", "Multi")):
            for clf in ("rf", "dt"):
                c = cell(pooled, m, task, clf)
                if c is not None:
                    macro(f"Seeds{tag}{tt}{clf.upper()}", f3(c.seed_mean))
                    macro(f"Seeds{tag}{tt}{clf.upper()}Std", f3(c.seed_std, 4))


def tab_stability():
    rows = []
    for ds, space, lab in (("datasense", "latent", r"\method{} (DataSense)"), ("datasense", "raw", r"$k$-Means+Sil (DataSense)"),
                           ("wustl_log", "latent", r"\method{} (WUSTL-IIoT, log-scaled)"),
                           ("wustl_log", "raw", r"$k$-Means+Sil (WUSTL-IIoT, log-scaled)"),
                           ("wustl", "latent", r"\method{} (WUSTL-IIoT, unchanged preprocessing)"),
                           ("wustl", "raw", r"$k$-Means+Sil (WUSTL-IIoT, unchanged preprocessing)")):
        s = rd(f"selsummary_{space}_{ds}")
        if s is None:
            continue
        s = s.iloc[0]
        rows.append(f"{lab} & {int(s.n_runs)} & {s.size_mean:.1f} & {int(s.size_min)}--{int(s.size_max)} & {int(s.n_always_selected)} & {int(s.n_never_selected)} & "
                    f"{f3(s.jaccard_all_mean)} & {f3(s.jaccard_removed_all_mean)} & {f3(s.nogueira)} \\\\")
        pre = PRE[ds] + ("Lat" if space == "latent" else "Raw")
        macro(f"{pre}SizeMean", f"{s.size_mean:.1f}")
        macro(f"{pre}SizeMin", int(s.size_min))
        macro(f"{pre}SizeMax", int(s.size_max))
        macro(f"{pre}SizeMedian", f"{s.size_median:.0f}")
        macro(f"{pre}Jaccard", f3(s.jaccard_all_mean))
        macro(f"{pre}JaccardWithin", f3(s.jaccard_within_seed_mean))
        macro(f"{pre}JaccardAcross", f3(s.jaccard_across_seed_mean))
        macro(f"{pre}JaccardRemoved", f3(s.jaccard_removed_all_mean))
        macro(f"{pre}Nogueira", f3(s.nogueira))
        macro(f"{pre}Always", int(s.n_always_selected))
        macro(f"{pre}Never", int(s.n_never_selected))
        macro(f"{pre}Runs", int(s.n_runs))
    if not rows:
        return
    body = f"""\\begin{{table*}}[t]
\\revon
\\caption{{Stability of the Fold-Specific Feature Subsets}}
\\label{{tab:stability}}
\\centering
\\footnotesize
\\begin{{tabular}}{{@{{}}lcccccccc@{{}}}}
\\toprule
Selector (dataset) & Folds & Mean $|S^{{*}}_r|$ & Range & Always selected & Never selected & Jaccard (selected) & Jaccard (removed) & Nogueira stability \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\multicolumn{{9}}{{@{{}}p{{0.97\\textwidth}}@{{}}}}{{Jaccard values are means over all pairs of folds. ``Always''/``Never'' count the features selected in all/none of the folds. DataSense has 71 features and WUSTL-IIoT-2021 has 41. The \\method{{}} rows for DataSense pool the three master seeds; the other rows use the primary execution.}}
\\end{{tabular}}
\\end{{table*}}
"""
    write("stability", body)


def tab_budget(ds, label, caption):
    d = rd(f"budgets_{ds}")
    if d is None:
        return
    budgets = [int(c[1:-5]) for c in d.columns if c.endswith("_mean")]
    rows = []
    best = {b: d[d.method != "all_features"][f"k{b}_mean"].max() for b in budgets}
    for _, r in d.iterrows():
        if r.method == "all_features":
            rows.append("\\midrule")
            rows.append("All Features$^{\\dagger}$ & " + " & ".join(f3(r[f"k{b}_mean"]) for b in budgets) + " & -- \\\\")
            continue
        if all(pd.isna(r[f"k{b}_mean"]) for b in budgets):
            continue
        cells = []
        for b in budgets:
            v = r[f"k{b}_mean"]
            s = f3(v)
            cells.append(f"\\textbf{{{s}}}" if (not pd.isna(v) and abs(v - best[b]) < 5e-4) else s)
        delta = r[f"k{budgets[0]}_mean"] - r[f"k{budgets[-1]}_mean"]
        rows.append(f"{LABEL[r.method]} & " + " & ".join(cells) + f" & {delta:+.3f} \\\\")
        tag = {"proposed": "Prop", "kmeans_sil": "KMS", "mcfs": "MCFS"}.get(r.method)
        if tag:
            for b in budgets:
                macro(f"{PRE[ds]}Bud{tag}{'abcdefghij'[budgets.index(b)]}", f3(r[f"k{b}_mean"]))
    mc = {"datasense": "8-Class", "nbaiot_log": "11-Class"}.get(ds, "5-Class")
    body = f"""\\begin{{table}}[t]
\\revon
\\caption{{{caption}}}
\\label{{{label}}}
\\centering
\\footnotesize
\\begin{{tabular}}{{@{{}}l{'c' * len(budgets)}c@{{}}}}
\\toprule
Method & {' & '.join(f'$k{{=}}{b}$' for b in budgets)} & $\\Delta$ \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\multicolumn{{{len(budgets) + 2}}}{{@{{}}p{{0.95\\columnwidth}}@{{}}}}{{$\\Delta = \\mathrm{{F1}}(k{{=}}{budgets[0]}) - \\mathrm{{F1}}(k{{=}}{budgets[-1]})$. Best value per budget in bold. $^{{\\dagger}}$Unconstrained reference using all features.}}
\\end{{tabular}}
\\end{{table}}
"""
    write(f"budget_{ds}", body)


def tab_protocols():
    d = rd("protocols_datasense")
    dup = rd("duplicates_datasense")
    if d is None or d.protocol.nunique() < 2:
        return
    rows = []
    names = {"sample_k10": "Sample-level (10 folds)", "group_exec_k5": "Execution-grouped (5 folds)",
             "group_device_k5": "Device-grouped (5 folds)"}
    for proto in ("sample_k10", "group_exec_k5", "group_device_k5"):
        x = d[d.protocol == proto]
        if x.empty:
            continue
        cells = [names[proto]]
        nf = x[x.method == "proposed"].nf.mean()
        cells.append(f"{nf:.1f}")
        for task in ("binary", "multiclass"):
            for m in ("proposed", "all_features"):
                c = x[(x.method == m) & (x.task == task) & (x.clf == "rf")]
                cells.append(f"{lead(c.f1_mean.iloc[0])}\\,{{\\scriptsize$\\pm${lead(c.f1_std.iloc[0])}}}" if len(c) else "--")
                if len(c):
                    macro(f"Proto{''.join(w.capitalize() for w in proto.split('_')[:-1])}{'Prop' if m == 'proposed' else 'All'}{'Bin' if task == 'binary' else 'Multi'}", f3(c.f1_mean.iloc[0]))
        if dup is not None:
            q = dup[dup.protocol == proto]
            cells.append(f"{100 * q.nonidle_dup_share_mean.iloc[0]:.2f}" if len(q) else "--")
        rows.append(" & ".join(cells) + " \\\\")
    body = f"""\\begin{{table*}}[t]
\\revon
\\caption{{Effect of the Fold-Construction Protocol on DataSense (RF Macro-F1, Mean$\\pm$Std Over Folds, Master Seed 42; the Complete Pipeline Is Refitted Inside Every Training Partition)}}
\\label{{tab:protocols}}
\\centering
\\footnotesize
\\begin{{tabular}}{{@{{}}lcccccc@{{}}}}
\\toprule
 & & \\multicolumn{{2}}{{c}}{{Binary}} & \\multicolumn{{2}}{{c}}{{8-class}} & Non-idle test windows with an \\\\
\\cmidrule(lr){{3-4}}\\cmidrule(lr){{5-6}}
Protocol & Mean $|S^{{*}}_r|$ & \\method & All Features & \\method & All Features & exact duplicate in training (\\%) \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""
    write("protocols", body)
    pc = rd("protocols_perclass_all_datasense")
    if pc is not None and len(pc) > 1:
        classes = [c for c in pc.columns if c != "protocol"]
        nice = {"benign": "Benign", "bruteforce": "BruteF.", "ddos": "DDoS", "dos": "DoS", "malware": "Malware",
                "mitm": "MitM", "recon": "Recon", "web": "Web"}
        rows = []
        for proto in ("sample_k10", "group_exec_k5", "group_device_k5"):
            x = pc[pc.protocol == proto]
            if len(x):
                rows.append(names[proto].split(" (")[0] + " & " + " & ".join(lead(x[c].iloc[0], 2) for c in classes) + " \\\\")
        body = f"""\\begin{{table}}[t]
\\revon
\\caption{{Per-Class RF F1 of the All Features Reference Under Each Protocol (DataSense, Mean Over Folds)}}
\\label{{tab:perclass}}
\\centering
\\footnotesize
\\setlength{{\\tabcolsep}}{{2.5pt}}
\\begin{{tabular}}{{@{{}}l{'c' * len(classes)}@{{}}}}
\\toprule
Protocol & {' & '.join(nice.get(c, c) for c in classes)} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
        write("perclass", body)


def tab_ablation():
    d = rd("ablation_datasense")
    sel = rd("ablation_selection_datasense")
    if d is None or sel is None or len(sel) < 2:
        return
    rows = []
    vmap = {"Group masking (proposed)": "main", "Independent masking, matched rate": "aug_independent",
            "No masking (noise + dropout)": "aug_none"}
    for conf, v in vmap.items():
        r = d[d.configuration == conf]
        s = sel[sel.variant == v]
        if r.empty or s.empty:
            continue
        r, s = r.iloc[0], s.iloc[0]
        rows.append(f"{conf} & {s.size_mean:.1f} ({int(s.size_min)}--{int(s.size_max)}) & {f3(s.jaccard_mean)} & {f3(s.J_full_mean)} & "
                    f"{f3(r.binary_rf)} & {f3(r.multiclass_rf)} & {f3(r.multiclass_dt)} & {f3(r.rf8_b25)} & {f3(r.rf8_b10)} \\\\")
    n = int(sel.n_folds.min())
    body = f"""\\begin{{table*}}[t]
\\revon
\\caption{{Controlled Augmentation Comparison on DataSense ({n} Folds of the Primary Execution; Only the Augmentation Used to Train the Encoder Differs)}}
\\label{{tab:ablation_aug}}
\\centering
\\footnotesize
\\begin{{tabular}}{{@{{}}lcccccccc@{{}}}}
\\toprule
 & \\multicolumn{{3}}{{c}}{{Selection}} & \\multicolumn{{3}}{{c}}{{Natural subset (macro-F1)}} & \\multicolumn{{2}}{{c}}{{Budgeted, 8-class RF}} \\\\
\\cmidrule(lr){{2-4}}\\cmidrule(lr){{5-7}}\\cmidrule(l){{8-9}}
Encoder augmentation & $|S^{{*}}_r|$ mean (range) & Jaccard & $J([d])$ & Bin.\\ RF & 8-cl.\\ RF & 8-cl.\\ DT & $k{{=}}25$ & $k{{=}}10$ \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\multicolumn{{9}}{{@{{}}p{{0.97\\textwidth}}@{{}}}}{{$J([d])$: latent Silhouette of the complete feature set at the fold-specific $k^{{*}}$. Jaccard: mean pairwise similarity of the subsets selected in the {n} folds.}}
\\end{{tabular}}
\\end{{table*}}
"""
    write("ablation_aug", body)


def tab_extras():
    rows = []
    for ds, lab in (("datasense", "DataSense"), ("wustl_log", "WUSTL-IIoT-2021 (log-scaled)"),
                    ("wustl", "WUSTL-IIoT-2021 (unchanged)")):
        e = rd(f"extras_{ds}")
        if e is None:
            continue
        e = e[e.protocol == "sample_k10"]
        if e.empty:
            continue
        cp = rd(f"cluster_cross_overall_{ds}")
        ms = lambda c: f"{f3(e[c].mean())}\\,{{\\scriptsize$\\pm${f3(e[c].std(ddof=0))}}}"  # noqa: E731
        rows.append(f"{lab} & {len(e)} & {ms('ari_resample')} & {ms('ami_resample')} & "
                    f"{f3(cp.ari_mean.iloc[0]) if cp is not None else '--'}\\,{{\\scriptsize$\\pm${f3(cp.ari_std.iloc[0]) if cp is not None else '--'}}} & "
                    f"{f3(cp.ami_mean.iloc[0]) if cp is not None else '--'}\\,{{\\scriptsize$\\pm${f3(cp.ami_std.iloc[0]) if cp is not None else '--'}}} & "
                    f"{ms('ins_acc')} & {ms('hoi_acc')} & {ms('hot_acc')} & {ms('hot_bacc')} \\\\")
        pre = PRE[ds]
        for c, nm in (("ari_resample", "AriRes"), ("ami_resample", "AmiRes"), ("ins_acc", "ProxyIn"), ("hoi_acc", "ProxyHoI"),
                      ("hot_acc", "ProxyHoT"), ("hot_bacc", "ProxyHoTBal"), ("hot_f1", "ProxyHoTFone"), ("silhouette", "SilSel")):
            if c in e.columns:
                macro(f"{pre}{nm}", f3(e[c].mean()))
                macro(f"{pre}{nm}Min", f3(e[c].min()))
        if cp is not None:
            macro(f"{pre}AriCross", f3(cp.ari_mean.iloc[0]))
            macro(f"{pre}AmiCross", f3(cp.ami_mean.iloc[0]))
            macro(f"{pre}AriCrossMin", f3(cp.ari_min.iloc[0]))
    if not rows:
        return
    body = f"""\\begin{{table*}}[t]
\\revon
\\caption{{Cluster Stability and Fidelity of the Random Forest Proxy Across the Fold-Specific Pipelines (Mean$\\pm$Std)}}
\\label{{tab:extras}}
\\centering
\\footnotesize
\\begin{{tabular}}{{@{{}}lccccccccc@{{}}}}
\\toprule
 & & \\multicolumn{{2}}{{c}}{{Within-fold stability}} & \\multicolumn{{2}}{{c}}{{Cross-pipeline stability}} & \\multicolumn{{4}}{{c}}{{Proxy agreement with latent clusters}} \\\\
\\cmidrule(lr){{3-4}}\\cmidrule(lr){{5-6}}\\cmidrule(l){{7-10}}
Dataset & Folds & ARI & AMI & ARI & AMI & In-sample acc. & Hold-out acc. & Test-fold acc. & Test-fold bal.\\ acc. \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\multicolumn{{10}}{{@{{}}p{{0.97\\textwidth}}@{{}}}}{{Within-fold: $k$-Means refitted on five training-fold subsamples. Cross-pipeline: all pairs of fold-specific pipelines labeling the same 5,000 reference rows. Hold-out: training-fold rows not used to fit the proxy. Test fold: rows unseen by the scaler, encoder, subset search, $k$-Means, and proxy.}}
\\end{{tabular}}
\\end{{table*}}
"""
    write("extras", body)


def tab_clarity(ds="datasense"):
    c = rd(f"xai_cluster_clarity_{ds}")
    if c is None:
        return
    c = c.sort_values("clarity_index", ascending=False)
    rows = []
    for _, r in c.iterrows():
        feats = ", ".join(f"\\feat{{{x}}}" for x in (r.top1, r.top2, r.top3))
        rows.append(f"{r.cluster} & {100 * r.share:.1f} & {f3(r.heldout_f1, 2)} & {f3(r.top3_share, 2)} & {f3(r.clarity_index, 2)} & "
                    f"{r.dominant_group} ({int(r.dominant_group_top5)}/5) & {feats} & {100 * r.benign_share:.0f} \\\\")
    k = len(c)
    body = f"""\\begin{{table*}}[t]
\\revon
\\caption{{Clarity Index and SHAP Profile of All {k} Latent Clusters of the Descriptive Configuration, in Decreasing Order of Clarity}}
\\label{{tab:clarity{'' if ds == 'datasense' else '_' + ds}}}
\\centering
\\footnotesize
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{@{{}}lrcccp{{3.2cm}}p{{6.3cm}}r@{{}}}}
\\toprule
Cluster & Share (\\%) & $F_c$ & $T_c$ & $CI_c$ & Dominant Context Group among top-5 features & Three highest-ranked features & Benign (\\%) \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\multicolumn{{8}}{{@{{}}p{{0.97\\textwidth}}@{{}}}}{{Share: fraction of all windows assigned to the cluster. $F_c$: held-out proxy F1. $T_c$: share of the cluster mean $|$SHAP$|$ carried by its three top features. $CI_c = F_c \\cdot T_c$. Feature names omit the \\texttt{{network\\_}} prefix. Benign: fraction of the cluster's windows with benign evaluation label, shown only as post-hoc context.}}
\\end{{tabular}}
\\end{{table*}}
"""
    write(f"clarity_{ds}", body)


def tab_removed():
    freq = rd("selfreq_latent_datasense")
    rem = rd("xai_removed_features_datasense")
    if freq is None:
        return
    f = freq[freq.selection_frequency < 1.0].sort_values("selection_frequency")
    removed_desc = set(rem.feature) if rem is not None else set()
    rows = []
    for _, r in f.iterrows():
        rows.append(f"\\feat{{{r.feature}}} & {r.group} & {int(r.n_runs - r.n_selected)}/{int(r.n_runs)} & "
                    f"{'Yes' if r.feature in removed_desc else 'No'} & {'Yes' if r.feature in rc.DATASENSE_SELECTED_17 else 'No'} \\\\")
    macro("DSNumEverRemoved", len(f))
    body = f"""\\begin{{table}}[t]
\\revon
\\caption{{DataSense Features Removed in at Least One of the {int(freq.n_runs.iloc[0])} Fold-Specific Subsets}}
\\label{{tab:removed}}
\\centering
\\scriptsize
\\setlength{{\\tabcolsep}}{{2.5pt}}
\\begin{{tabular}}{{@{{}}llccc@{{}}}}
\\toprule
Feature & Context Group & Removed & Descr. & DS-17 \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\multicolumn{{5}}{{@{{}}p{{0.97\\columnwidth}}@{{}}}}{{Removed: number of folds in which the feature is excluded. Descr.: removed in the full-data descriptive configuration. DS-17: member of the supervised reference subset.}}
\\end{{tabular}}
\\end{{table}}
"""
    write("removed", body)


def verdict(diff, p_w, p_t, alpha=0.05):
    """Decision rules fixed in reports/02_preregistration_third_dataset.md."""
    if p_w < alpha and diff > 0:
        return "Superior" + (" (equiv.)" if p_t < alpha else "")
    if p_w < alpha and diff < 0:
        return "Inferior" + (" (equiv.)" if p_t < alpha else "")
    if p_t < alpha:
        return "Equivalent"
    return "Inconclusive"


def tab_crossdataset():
    specs = [("datasense", "DataSense", {"all_features": "All Features", "mcfs_25": "MCFS-25", "kmeans_sil": "$k$-Means+Sil"}),
             ("nbaiot_log", "N-BaIoT", {"all_features": "All Features", "mcfs_40": "MCFS-40", "kmeans_sil": "$k$-Means+Sil"}),
             ("wustl_log", "WUSTL-IIoT", {"all_features": "All Features", "mcfs_14": "MCFS-14", "kmeans_sil": "$k$-Means+Sil"})]
    rows = []
    for ds, lab, refs in specs:
        s = rd(f"significance_seed42_{ds}")
        if s is None:
            continue
        first = True
        for ref, rl in refs.items():
            cells = [lab if first else "", rl]
            first = False
            for task in ("binary", "multiclass"):
                x = s[(s.task == task) & (s.clf == "rf") & (s.reference == ref)]
                if x.empty:
                    cells += ["--", "--", "--", "--"]
                    continue
                x = x.iloc[0]
                pw = "$<$0.001" if x.wilcoxon_p_two_sided < 0.001 else f"{x.wilcoxon_p_two_sided:.3f}"
                pt = "$<$0.001" if x.tost_p_delta001 < 0.001 else f"{x.tost_p_delta001:.3f}"
                v = verdict(x.mean_diff, x.wilcoxon_p_two_sided, x.tost_p_delta001)
                cells += [f"{x.mean_diff:+.4f}", pw, pt, v]
                macro(f"X{PRE.get(ds, 'NB') if ds != 'nbaiot_log' else 'NB'}{''.join(c for c in ref.title() if c.isalpha())}{'Bin' if task == 'binary' else 'Multi'}", v)
            rows.append(" & ".join(cells) + " \\\\")
        rows.append("\\midrule")
    rows = rows[:-1]
    body = f"""\\begin{{table*}}[t]
\\revon
\\caption{{Pre-Registered Statistical Comparison of \\method{{}} Across the Three Datasets (RF, 10 Paired Folds, Master Seed 42)}}
\\label{{tab:crossdataset}}
\\centering
\\footnotesize
\\setlength{{\\tabcolsep}}{{3.5pt}}
\\begin{{tabular}}{{@{{}}llrccl rccl@{{}}}}
\\toprule
 & & \\multicolumn{{4}}{{c}}{{Binary}} & \\multicolumn{{4}}{{c}}{{Multi-class}} \\\\
\\cmidrule(lr){{3-6}}\\cmidrule(l){{7-10}}
Dataset & Reference & $\\Delta$F1 & $p_{{\\mathrm{{W}}}}$ & $p_{{\\mathrm{{TOST}}}}$ & Verdict & $\\Delta$F1 & $p_{{\\mathrm{{W}}}}$ & $p_{{\\mathrm{{TOST}}}}$ & Verdict \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\multicolumn{{10}}{{@{{}}p{{0.97\\textwidth}}@{{}}}}{{$\\Delta$F1: mean macro-F1 of \\method{{}} minus the reference. $p_{{\\mathrm{{W}}}}$: two-sided Wilcoxon signed-rank test. $p_{{\\mathrm{{TOST}}}}$: paired equivalence test (two one-sided Wilcoxon tests) with margin $\\pm$0.01 macro-F1. Verdict rules were fixed before the N-BaIoT experiment: Superior/Inferior if $p_{{\\mathrm{{W}}}} < 0.05$; Equivalent if $p_{{\\mathrm{{TOST}}}} < 0.05$; otherwise Inconclusive. ``(equiv.)'' marks a significant difference that nevertheless lies within the $\\pm$0.01 margin. The ranking budgets (25, 40, 14) are 35\\% of the number of features of each dataset.}}
\\end{{tabular}}
\\end{{table*}}
"""
    write("crossdataset", body)


def main():
    tab_crossdataset()
    tab_natural("nbaiot_log", "tab:nbaiot", "Third Dataset N-BaIoT (Log-Scaled Attributes): Results at the Natural Operating Point of Each Method", False)
    tab_budget("nbaiot_log", "tab:nbaiot_budget", "Macro-F1 vs. Feature Budget $k$ for the 11-Class Task on N-BaIoT (RF, 10 Folds)")
    tab_natural("datasense", "tab:natural", "Classification Results on DataSense at the Natural Operating Point of Each Method", True)
    tab_natural("wustl_log", "tab:wustl", "External Dataset WUSTL-IIoT-2021 (Log-Scaled Attributes): Results at the Natural Operating Point of Each Method", False)
    tab_natural("wustl", "tab:wustl_direct", "External Dataset WUSTL-IIoT-2021 (Unchanged Preprocessing): Results at the Natural Operating Point of Each Method", False)
    tab_significance()
    tab_seeds()
    tab_stability()
    tab_budget("datasense", "tab:budget", "Macro-F1 vs. Feature Budget $k$ for the 8-Class Task on DataSense (RF, 10 Folds, Master Seed 42)")
    tab_budget("wustl_log", "tab:wustl_budget", "Macro-F1 vs. Feature Budget $k$ for the 5-Class Task on WUSTL-IIoT-2021 (Log-Scaled Attributes; RF, 10 Folds)")
    tab_budget("wustl", "tab:wustl_budget_direct", "Macro-F1 vs. Feature Budget $k$ for the 5-Class Task on WUSTL-IIoT-2021 (Unchanged Preprocessing; RF, 10 Folds)")
    tab_protocols()
    tab_ablation()
    tab_extras()
    tab_clarity("datasense")
    tab_clarity("wustl_log")
    tab_removed()
    with open(os.path.join(HERE, "manuscript", "numbers_auto.tex"), "w", encoding="utf-8") as f:
        for k, v in sorted(MACROS.items()):
            f.write(f"\\newcommand{{\\N{k}}}{{{v}}}\n")
    print(f"[tables] {len(MACROS)} macros")


if __name__ == "__main__":
    main()
