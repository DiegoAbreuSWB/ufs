MANUSCRIPT_ID = "Access-2026-39805"
TITLE = "ContrastiveXAI-FS: Explainable Unsupervised Feature Selection for IIoT Intrusion Detection"
AUTHOR = "Diego Medeiros de Abreu"

COVER = [
    "Dear Editor,",
    "Thank you for the opportunity to resubmit our manuscript and for the constructive comments of the reviewers. "
    "We are uploading (a) this point-by-point response (“Author’s Response Files”), (b) the updated manuscript "
    "with all changes highlighted in yellow (“Highlighted PDF”), and (c) a clean copy of the updated manuscript as "
    "LaTeX source and PDF (“Main Manuscript”).",
]

SUMMARY = [
    "We thank Reviewer 1 for the detailed and constructive assessment and Reviewer 2 for recommending publication. "
    "In this revision we prioritized new experimental evidence for every point raised by Reviewer 1, and we made the "
    "complete implementation publicly available. The main additions are:",
    "(1) two additional public IoT/IIoT datasets, N-BaIoT and WUSTL-IIoT-2021, evaluated with the complete fold-wise "
    "pipeline, the N-BaIoT protocol and statistical decision rules having been registered in the public repository "
    "before the experiment was run; (2) attack-execution-grouped and device-grouped cross-validation; (3) three "
    "independent master seeds; (4) selection-stability analysis across the 30 fold-specific subsets (subset sizes, "
    "selection frequencies, Jaccard similarity, and the Nogueira stability estimator); (5) controlled component "
    "comparisons (augmentation, search direction, and evaluation space at identical budgets); (6) held-out fidelity of "
    "the SHAP proxy on data unseen by every stage of the pipeline; (7) cluster stability within folds and across "
    "fold-specific pipelines and seeds; (8) the Silhouette profile over the candidate cluster numbers; (9) a "
    "quantitative clarity index for selecting the clusters to interpret; and (10) paired equivalence tests (TOST, "
    "margin ±0.01 macro-F1) in addition to two-sided Wilcoxon tests.",
    "To produce these analyses, the experimental code was re-implemented as a single resumable runner in which "
    "preprocessing, VICReg training, selection of the number of clusters, feature selection, clustering, and the "
    "SHAP proxy are all fitted inside the training partition of each fold, and all experiments were re-executed with "
    "it. All numbers in the revised manuscript are produced by this implementation and are regenerated automatically "
    "from its outputs by the released scripts. Consequently, several values were updated; for example, the natural "
    "operating point is now reported as the mean over the fold-specific subsets (59 of 71 features on average; 61 in "
    "the descriptive configuration), and the eight-class macro-F1 with Random Forest is 0.908. The table below "
    "summarizes the main revised results.",
    {"header": ["Evidence", "Result", "Location"],
     "rows": [
         ["Natural operating point (DataSense, RF)", "0.936 binary, 0.908 eight-class; equivalent to all features within ±0.01 (TOST); significantly above MCFS-25 and DS-17", "Sec. VI-B, Tables 11–12"],
         ["Three master seeds", "0.935 ± 0.001 binary, 0.902 ± 0.005 eight-class; equivalence to all features holds over 30 folds", "Sec. VI-B, Table 13"],
         ["Selection stability (30 subsets)", "Median 61 features; Jaccard 0.69 (0.40 for raw-space search); 46 features selected in ≥80% of folds", "Sec. VI-A, Table 10, Fig. 3"],
         ["Cluster stability", "Within-fold ARI 0.964; across 30 fold-specific pipelines ARI 0.944", "Sec. VI-G, Table 22"],
         ["Held-out proxy fidelity", "0.989 accuracy, 0.970 balanced accuracy on outer test folds", "Sec. VI-G, Table 22"],
         ["N-BaIoT", "Equivalent to all features in both tasks; significantly above MCFS-40", "Sec. VI-F, Tables 19–20"],
         ["Grouped CV", "Binary detection 0.931 (execution-grouped) and 0.903 (device-grouped); no significant difference to all features", "Sec. VI-E, Tables 17–18"],
         ["Controlled comparisons", "Latent-space search above raw-space search at identical budgets (significant at k = 20)", "Sec. VI-D, Tables 15–16"],
     ]},
]

R1 = [
    dict(
        concern="Strengthen the discussion of sample-level cross-validation. The manuscript now correctly acknowledges that samples from the same device, session, attack execution, or temporal neighborhood may occur across training and test folds. This limitation is important because IIoT traffic can contain strong device- and session-specific signatures. If grouped evaluation cannot be added, this limitation should remain prominent in the abstract/conclusion and not only in the experimental and discussion sections.",
        response=[
            "Thank you. We added grouped evaluation, so the effect is now measured rather than only discussed, and we also kept the limitation in the abstract and conclusion as requested. Two grouped protocols were implemented with StratifiedGroupKFold, and the complete pipeline (scaler, VICReg encoder, k*, feature selection, clustering, proxy, and classifiers) is refitted inside every training partition: (i) an execution-grouped protocol, in which each of the 936 attack executions (column label_full) and each 60-s block of the benign capture forms a group, so that no attack execution or benign time block is shared between training and test data; and (ii) a device-grouped protocol, in which the test partition contains only devices never seen in training.",
            "Under execution-grouped folds, binary detection is essentially unchanged (0.931 for ContrastiveXAI-FS and 0.933 for all features, equivalent within ±0.01) and eight-class macro-F1 is 0.793 and 0.806. Under device-grouped folds, binary macro-F1 is 0.903 and 0.913 and eight-class macro-F1 is 0.653 and 0.691; the per-class analysis shows that the multi-class reduction is concentrated in categories recorded on very few devices (web attacks target a single device). In both grouped protocols, the differences between ContrastiveXAI-FS and the complete feature set are not statistically significant. We also report the share of test windows that have an exact duplicate in the training partition under each protocol.",
        ],
        action=[
            "Section V-A now defines the sample-level, execution-grouped, and device-grouped protocols; Section VI-E (Tables 17 and 18) reports the grouped results; Section VII-E discusses them. The abstract ends with “Main results use sample-level folds and therefore measure within-dataset generalization; device- and attack-execution-grouped evaluations are reported to quantify this effect,” and the conclusion contains the corresponding statement. Section III-B additionally reports the class distribution and the structure of the evaluated dataframe (single benign capture, 936 attack executions, and windows without traffic).",
        ],
    ),
    dict(
        concern="Cross-dataset validation remains desirable. The method is evaluated only on DataSense CIC IIoT 2025. [...] At least one additional IoT/IIoT dataset would considerably strengthen the paper. If this cannot be added in the present revision, claims regarding general applicability should remain conservative.",
        response=[
            "We agree and added two public datasets with schemas that differ from DataSense, evaluated with the unchanged method, the same hyperparameters, and the complete fold-wise protocol: N-BaIoT (nine commercial IoT devices, Mirai and BASHLITE botnets, 115 damped-window statistics) and WUSTL-IIoT-2021 (industrial control testbed with SCADA/Modbus traffic, 41 flow attributes). The Context Groups were rebuilt for each schema with the construction rule of Section III-C (five stream-aggregation groups for N-BaIoT and nine measurement groups for WUSTL-IIoT-2021), which experimentally demonstrates the transferability of the grouping principle. For N-BaIoT, the protocol, the compared configurations, and the statistical decision rules (Wilcoxon for superiority; TOST with margin ±0.01 for equivalence) were registered in the public repository before the experiment was run.",
            "On N-BaIoT, ContrastiveXAI-FS retains on average 74 of 115 features, is statistically equivalent to the complete feature set in both the binary (0.9997 vs. 0.9998) and eleven-class tasks (0.873 vs. 0.881), and significantly outperforms MCFS-40 in both tasks (eleven-class 0.873 vs. 0.804, p = 0.002). On WUSTL-IIoT-2021, it reaches 0.974 binary and 0.901 five-class macro-F1 with on average 16 of 41 attributes. On both datasets, the latent clusters are highly stable and are reproduced by the proxy on the outer test folds with balanced accuracy of at least 0.96. Claims of general applicability are limited to these evaluated settings.",
        ],
        action=[
            "Section V-H describes both datasets, their Context Groups (Table 8 for WUSTL-IIoT-2021), the sampling, and the preprocessing; Section VI-F reports the pre-registered comparison across the three datasets (Table 19) and the natural operating points (Tables 20 and 21); Section VII-E discusses transferability. The pre-registration file is included in the repository.",
        ],
    ),
    dict(
        concern="Report feature-selection stability across folds. [...] it would be useful to report the distribution of selected subset sizes and selection frequency of individual features across the ten folds. [...] A simple stability table or Jaccard similarity analysis would substantially improve confidence in the feature-selection results without requiring a new dataset.",
        response=[
            "We added a complete stability analysis over the 30 fold-specific subsets obtained with three master seeds. The subsets retain a median of 61 features (mean 55.7); 46 of the 71 features are selected in at least 80% of the folds; the mean pairwise Jaccard similarity is 0.69, both within a seed (0.70) and across seeds (0.69), compared with 0.40 for the raw-space k-Means+Sil wrapper. We also report the chance-corrected stability estimator of Nogueira et al. and the Jaccard similarity of the removed sets. The features removed most often are the four inter-packet timing statistics (network_time-delta_*), removed in 67–73% of the folds, which is consistent with the descriptive configuration. The same analysis is reported for N-BaIoT and WUSTL-IIoT-2021.",
        ],
        action=[
            "Section VI-A, Table 10 (stability across datasets), Fig. 3 (selection frequency), and Table 24 (features removed in more than 20% of the folds). The per-fold subsets are released in the repository.",
        ],
    ),
    dict(
        concern="A controlled component ablation would still strengthen the novelty claim. [...] I encourage comparison of: standard VICReg vs. group-aware VICReg; backward-only vs. bidirectional search; raw-space vs. latent-space search at identical feature budgets; and group masking vs. conventional independent feature corruption.",
        response=[
            "We added controlled comparisons in which exactly one component is changed while folds, seeds, architecture, objective, and search are fixed. (a) Evaluation space: at identical budgets, subsets chosen through the VICReg latent space yield higher eight-class macro-F1 than the same search in the standardized input space at every budget from 15 to 25 (+0.053, +0.062, +0.040), significantly at k = 20 (7 of 7 folds, p = 0.016), and the latent search produces markedly more stable subsets. (b) Search direction: where the bidirectional and backward-only searches differ, the bidirectional search attains higher eight-class macro-F1 on average (+0.027). (c) Augmentation: an encoder trained with conventional independent per-feature masking at the same expected rate (standard VICReg augmentation in the sense of unstructured corruption) yields comparable performance (binary 0.937 vs. 0.936, equivalent within ±0.01; eight-class 0.913 vs. 0.908). Comparison (c) requires retraining the encoder in every fold and was run on five folds; with five pairs, the smallest attainable two-sided Wilcoxon p-value is 0.0625, which is stated in the table.",
        ],
        action=[
            "Section V-G defines the comparisons; Section VI-D reports them in Tables 15 and 16; Section VII-A discusses the role of the latent representation and of the augmentation.",
        ],
    ),
    dict(
        concern="Repeat the stochastic experiment using several independent master seeds. [...] three to five independent master seeds would provide stronger evidence of stability.",
        response=[
            "The complete 10-fold experiment was executed with three independent master seeds (42, 43, 44). Each master seed defines a different outer-fold partition and, through the fold seed s + r, all stochastic components (encoder-training subsample, initialization, mini-batch order, augmentation, search subsample, k-Means, and classifiers). Across seeds, ContrastiveXAI-FS reaches 0.935 ± 0.001 binary and 0.902 ± 0.005 eight-class Random Forest macro-F1 (mean ± standard deviation of the seed means), and over the 30 paired folds the difference to the complete feature set remains within the ±0.01 equivalence margin in both tasks (TOST p < 0.001 and p = 0.015 with RF; p < 0.001 and p = 0.001 with DT).",
        ],
        action=[
            "Section V-B (seed schedule), Section VI-B and Table 13 (per-seed and across-seed results), Table 12 (paired tests over 30 folds), and Table 9 (configuration). The stability and cluster analyses also pool the three seeds.",
        ],
    ),
    dict(
        concern="Consider held-out evaluation of the SHAP proxy. [...] A separate proxy-training/proxy-testing split would provide stronger support for the claim that SHAP profiles faithfully approximate the discovered latent partition.",
        response=[
            "We implemented a held-out protocol. The proxy is trained on a cluster-balanced sample of training-fold rows and evaluated (i) on disjoint held-out training rows and (ii) on the outer test fold, whose rows are unseen by the scaler, encoder, subset search, k-Means model, and proxy. On DataSense, the proxy reproduces the latent clusters with 0.989 accuracy and 0.970 balanced accuracy on the outer test folds, close to its in-sample agreement (0.995). In the descriptive configuration used for the SHAP profiles, held-out accuracy is 0.988 and balanced accuracy 0.980, and SHAP values are now computed on held-out rows only. Per-cluster held-out F1 is reported for all 13 clusters (minimum 0.88).",
        ],
        action=[
            "Section IV-G (proxy protocol), Algorithm 1 (lines 13–15), Section VI-G and Table 22 (fidelity across datasets), and Table 23 (per-cluster held-out F1).",
        ],
    ),
    dict(
        concern="Clarify the practical value of retaining 63 of 71 features. [...] the paper should continue to avoid implying computational savings.",
        response=[
            "We revised the explanation to state the practical value explicitly in terms of three properties and without any computational claim: (i) the refinement removes attributes that do not support the learned behavioral structure, chiefly the inter-packet timing statistics, without loss of detection performance relative to the complete input (statistical equivalence within ±0.01); (ii) the selected subset is the validated input of the cluster explanations; and (iii) the procedure is label-free. The manuscript states that runtime, memory, throughput, and energy were not measured and that no efficiency claim is made.",
        ],
        action=[
            "Section VII-C, first paragraph (“The practical value of retaining most of the 71 features lies in three properties rather than in computational savings, which were not measured”), and Section VII-D (last sentence).",
        ],
    ),
    dict(
        concern="The low-budget result deserves emphasis rather than being treated as secondary. [...] I recommend preserving this candid interpretation in the final manuscript.",
        response=[
            "We preserved this interpretation and present it as the operating region of the method. With the fold-wise implementation, MCFS remains the most stable method at k = 10 (0.881), and ContrastiveXAI-FS improves steadily from k = 10 to k = 25 (0.878). The fixed-budget procedure for the search methods is now defined explicitly (forced backward elimination from the natural subset), and only folds in which a method provides exactly k features enter the comparison.",
        ],
        action=[
            "Section IV-F (budget-constrained variant), Section VI-C and Table 14, Section VII-B (“Operating Region and Feature Budgets”), and the conclusion.",
        ],
    ),
    dict(
        concern="Clarify why 13 clusters are selected. [...] report the Silhouette profile across candidate k values, or at least provide the scores near k=13.",
        response=[
            "We report the complete latent Silhouette profile for the candidate set K = {2, ..., 14} for all 30 fold-specific pipelines (Fig. 4), together with an extended profile up to k = 30 (Fig. 5). The profile rises steeply up to k ≈ 10 and then flattens; in the descriptive configuration k* = 13 (Silhouette 0.554 versus 0.552 at k = 14), and across folds k* is 13 or 14 in 28 of 30 cases. The extended profile shows that the Silhouette continues to increase slowly beyond the candidate range. The upper bound of K therefore acts as the granularity limit of the explanation, keeping the number of cluster profiles small enough to be inspected by an analyst while operating on the plateau of the Silhouette curve. We also corrected the description of the objective: k* is selected once per training fold with all features (Eq. 11) and is held fixed during the subset search (Eq. 12), which is the procedure executed in all experiments.",
        ],
        action=[
            "Section IV-E (Eqs. 11–12), Section VI-A (“Number of clusters”), Figs. 4 and 5, and Section VII-F.",
        ],
    ),
    dict(
        concern="Consider reporting cluster stability. Since the explainability analysis depends on k-Means clusters, the stability of these clusters across seeds or folds matters.",
        response=[
            "We report two measures with the adjusted Rand index (ARI) and adjusted mutual information (AMI). Within-fold stability refits k-Means on independent training subsamples (ARI 0.964). Cross-pipeline stability lets each of the 30 fold-specific pipelines—each with its own scaler, encoder, selected subset, and k-Means model—label the same 5,000 reference windows; the mean pairwise ARI is 0.944 (AMI 0.882), with no difference between pipelines from the same or from different master seeds. The clusters explained by SHAP are therefore reproducible across folds and seeds.",
        ],
        action=[
            "Section V-E (definitions), Section VI-G and Table 22.",
        ],
    ),
    dict(
        concern="Explain why only four of thirteen clusters receive detailed interpretation. [...] a quantitative criterion for “clearest” would be preferable to subjective selection.",
        response=[
            "We introduced a quantitative clarity index CI_c = F_c · T_c (Eq. 15), where F_c is the held-out proxy F1 of cluster c and T_c is the share of the cluster’s mean |SHAP| mass carried by its three most important features. The index is reported for all 13 clusters (Table 23), together with their dominant features, Context Groups, and post-hoc label composition. The four clusters with the highest index (CI_c ≥ 0.385) are separated by a clear gap from the remaining clusters (CI_c ≤ 0.333) and are the ones interpreted in detail: a TCP-termination profile (FIN/RST counts), a TTL profile, a payload-size profile, and a TTL-variability profile. The profile descriptions were updated to the clusters selected by this criterion.",
        ],
        action=[
            "Section IV-G (Eq. 15), Section VI-G (“Selection of the clusters to interpret”), Table 23, Fig. 7, and the four profile paragraphs.",
        ],
    ),
    dict(
        concern="Provide the implementation artifacts promised in the response. [...] code availability would materially improve reproducibility.",
        response=[
            "The complete implementation is now publicly available at https://github.com/DiegoAbreuSWB/ufs. It contains the fold-wise preprocessing, the Context Group definitions for the three datasets, the group-aware augmentation and VICReg training, the bidirectional and budget-constrained searches, the baselines, the resumable cross-validation runner with its seed schedule, the per-fold outputs (encoder weights, selected subsets, k*, Silhouette profiles, cluster-stability and proxy-fidelity measurements, and downstream metrics), the N-BaIoT pre-registration, the scripts that regenerate every table and figure, and the LaTeX source of the manuscript. The datasets are not redistributed and must be obtained from their original sources.",
        ],
        action=[
            "New “Data and Code Availability” section; Section V-J (Reproducibility).",
        ],
    ),
    dict(
        concern="Reduce some repetition. The distinction between refinement and compression, limitations of the 63-feature configuration, inability to causally attribute gains to VICReg, and scope of the SHAP proxy are repeated in several sections.",
        response=[
            "We consolidated these points. The refinement/compression distinction is now introduced once in Section I and used thereafter; the scope of the SHAP proxy is stated once in Section IV-G; the statements about causal attribution were replaced by the controlled comparisons of Section VI-D; and the repeated caveats in Sections IV, V, VI, and VII were removed, which shortened those sections while adding the new analyses.",
        ],
        action=[
            "Sections I, II-C, IV-G, the former Section IV-J (merged into Sections I and VII), V, VI, and VII.",
        ],
    ),
    dict(
        concern="Perform a final formatting and language pass. I noticed several typesetting artifacts in the proof, including merged words such as “recentsupervised,” “istherefore,” “formalstatistical,” “exceedsthe” [...]",
        response=[
            "The manuscript was re-typeset from LaTeX source with the current IEEE Access template, and a full language pass was performed. Long dataframe column names, which caused the merged words in the previous proof, are now typeset with permitted line breaks, and the final PDF was checked for merged words.",
        ],
        action=[
            "Entire manuscript.",
        ],
    ),
    dict(
        concern="References [12] and [18] are given as arXiv preprints. If final peer-reviewed versions are available, the authors should cite those versions instead.",
        response=[
            "Both works were published at the International Conference on Learning Representations (ICLR) 2022, and the references now cite the published versions. Capitalization of acronyms in the bibliography was also corrected.",
        ],
        action=[
            "References [12] (VICReg) and [18] (SCARF).",
        ],
    ),
]

REVIEWERS = [
    dict(name="Reviewer #1", tag="Reviewer#1",
         intro=["We sincerely thank Reviewer 1 for the careful assessment and for recognizing the improvements of the previous revision. Each point is addressed below; all changes are highlighted in yellow in the marked manuscript."],
         concerns=R1),
    dict(name="Reviewer #2", tag="Reviewer#2",
         intro=["We thank Reviewer 2 for the positive assessment and the recommendation for publication. The additional analyses requested by Reviewer 1 further strengthen the points appreciated by Reviewer 2, and no further changes were requested."],
         concerns=[]),
]
