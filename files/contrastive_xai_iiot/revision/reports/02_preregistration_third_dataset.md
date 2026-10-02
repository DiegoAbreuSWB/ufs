# Pre-registration — third dataset (written before any result on it was computed)

Date: 2026-10-02. Results on DataSense (fold 0 only) and WUSTL-IIoT-2021 were already known;
no result on the dataset below had been computed when this file was written.

## Dataset choice and criterion

Criterion (fixed before evaluation): a public IoT/IIoT intrusion dataset that (i) can be
downloaded without registration, (ii) consists of numeric tabular attributes, (iii) has a
documented schema from which Context Groups can be built by measurement semantics alone,
and (iv) has a multi-class attack label. The dataset is NOT chosen because of an expected
outcome, and its result will be reported in the paper whatever it is. WUSTL-IIoT-2021 stays
in the paper regardless of this experiment.

Chosen: **N-BaIoT** (Meidan et al., IEEE Pervasive Computing 2018; UCI ML Repository id 442).
115 numeric features = 23 statistics x 5 damped time windows (L5, L3, L1, L0.1, L0.01),
9 commercial IoT devices, benign traffic + Mirai and BASHLITE attack families.

Known caveat stated in advance: N-BaIoT is reported in the literature to be close to
saturated for supervised classifiers. A "similar" outcome there is weak evidence of
superiority and will be described as such.

## Protocol (identical to the other datasets unless stated)

- Same fold-internal pipeline, hyperparameters (Table of configuration) and master seed 42, 10 stratified folds.
- Label-free uniform row sample of fixed size (to be stated), drawn once with a fixed seed, before any evaluation.
- Tasks: binary (benign vs attack) and multi-class (benign + attack types as given in the files).
- Context Groups: by stream type (host MAC-IP, host IP, channel, channel jitter, socket), defined from
  the feature-name documentation only.
- Compared configurations: ContrastiveXAI-FS (natural point), All Features, k-Means+Sil,
  MCFS and Variance at the ranking budget, fixed budgets for the proposed method, k-Means+Sil and MCFS.
- No hyperparameter tuning on this dataset. Any preprocessing adaptation (e.g. log scaling) is
  decided from the schema/marginal distributions only, before evaluation, and reported.

## Decision rules (fixed in advance)

- Superiority: two-sided Wilcoxon signed-rank over the 10 paired folds, alpha = 0.05.
- "Statistically similar" is NOT inferred from a non-significant test. It is claimed only if a
  paired equivalence test (TOST on fold-wise differences, Wilcoxon-based) rejects both one-sided
  nulls with equivalence margin **delta = 0.01 macro-F1**, alpha = 0.05.
- Otherwise the result is reported as "inconclusive" or "inferior", as the data indicate.

## Addendum (2026-10-02, before any evaluation on N-BaIoT)

- Sample: uniform, label-free inclusion probability 0.035 for every row of the 89 files
  (7,062,606 rows) with seed 2018 -> 247,277 rows, 115 features, 9 devices.
  Labels: binary (benign/attack); multi-class = benign + 5 Gafgyt + 5 Mirai attacks (11 classes).
- Label-free marginal check: 19/115 features have max|z| > 50 and float32 summation overflows.
  Decision: apply the same signed log transform used for WUSTL-IIoT-2021,
  x -> sign(x) log(1 + |x|), before fold-wise standardization. The unchanged-preprocessing
  variant is not run on N-BaIoT (compute budget); this is stated in the paper.
- Context Groups (5, by stream aggregation, from the dataset documentation):
  MI_dir (source MAC-IP), H (source IP), HH (channel), HH_jit (channel jitter), HpHp (socket).
- Ranking budget: 40 features (35% of 115, the same proportion as the other datasets);
  fixed budgets k in {10, 20, 30, 40}.
