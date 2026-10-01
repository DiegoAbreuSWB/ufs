# Reviewer 1 — concerns as stated in the decision letter (06-Sep-2026)

Verbatim text used as the "Reviewer's concern" field of the response letter.

1. **Sample-level cross-validation.** Strengthen the discussion of sample-level cross-validation. The manuscript now correctly acknowledges that samples from the same device, session, attack execution, or temporal neighborhood may occur across training and test folds. This limitation is important because IIoT traffic can contain strong device- and session-specific signatures. If grouped evaluation cannot be added, this limitation should remain prominent in the abstract/conclusion and not only in the experimental and discussion sections.

2. **Cross-dataset validation.** Cross-dataset validation remains desirable. The method is evaluated only on DataSense CIC IIoT 2025. The authors explain that Context Groups can be reconstructed for other datasets according to measurement semantics, but this transferability is currently conceptual rather than experimentally demonstrated. At least one additional IoT/IIoT dataset would considerably strengthen the paper. If this cannot be added in the present revision, claims regarding general applicability should remain conservative.

3. **Feature-selection stability.** Report feature-selection stability across folds. Since feature selection is independently performed within each training fold, it would be useful to report the distribution of selected subset sizes and selection frequency of individual features across the ten folds. The current 63-feature configuration is descriptive, whereas the actual evaluation uses fold-specific subsets. A simple stability table or Jaccard similarity analysis would substantially improve confidence in the feature-selection results without requiring a new dataset.

4. **Controlled component ablation.** A controlled component ablation would still strengthen the novelty claim. The manuscript correctly states that Table 11 is not a strict ablation because configurations differ simultaneously in representation and feature cardinality. If computationally feasible, I encourage comparison of: standard VICReg vs. group-aware VICReg; backward-only vs. bidirectional search; raw-space vs. latent-space search at identical feature budgets; and group masking vs. conventional independent feature corruption.

5. **Independent master seeds.** Repeat the stochastic experiment using several independent master seeds. The reported standard deviations currently represent fold variation from one 10-fold execution, not independent experimental repetitions. Because VICReg training, augmentation, k-Means initialization, and Random Forest fitting contain stochastic components, three to five independent master seeds would provide stronger evidence of stability.

6. **Held-out SHAP proxy.** Consider held-out evaluation of the SHAP proxy. The Random Forest proxy reaches 94.2% agreement with the latent cluster assignments, but this is explicitly not a held-out fidelity measurement. A separate proxy-training/proxy-testing split would provide stronger support for the claim that SHAP profiles faithfully approximate the discovered latent partition.

7. **Practical value of 63/71.** Clarify the practical value of retaining 63 of 71 features. The authors have substantially improved this point by describing the result as feature refinement rather than compression. I agree with this framing. Still, because only 11.3% of the features are removed and no runtime, memory, throughput, or energy measurements are reported, the paper should continue to avoid implying computational savings. The manuscript itself correctly acknowledges this limitation.

8. **Low-budget result.** The low-budget result deserves emphasis rather than being treated as secondary. At k=10, ContrastiveXAI-FS reaches 0.806 macro-F1, compared with 0.880 for MCFS and 0.878 for k-Means+Sil. This is an informative result because it establishes the operating region of the proposed method. The current discussion handles this appropriately; I recommend preserving this candid interpretation in the final manuscript.

9. **Why 13 clusters.** Clarify why 13 clusters are selected. The manuscript explains that k*=13 emerges from the Silhouette objective and is not expected to equal the two or eight evaluation classes. It would still be useful to report the Silhouette profile across candidate k values, or at least provide the scores near k=13, to demonstrate that the selected cluster count represents a meaningful optimum rather than a marginal numerical maximum.

10. **Cluster stability.** Consider reporting cluster stability. Since the explainability analysis depends on k-Means clusters, the stability of these clusters across seeds or folds matters. A cluster-stability measure would make the subsequent SHAP interpretation more convincing.

11. **Four of thirteen clusters.** Explain why only four of thirteen clusters receive detailed interpretation. The manuscript states that these four had the clearest SHAP profiles. This is reasonable, but a quantitative criterion for "clearest" would be preferable to subjective selection. Otherwise, there is a possibility of presenting only the most semantically convenient clusters.

12. **Implementation artifacts.** Provide the implementation artifacts promised in the response. The response states that implementation and preprocessing assets are planned for public release with the final publication package. Given the complexity of fold-wise VICReg training, semantic grouping, subset search, and seed management, code availability would materially improve reproducibility.

13. **Repetition.** Reduce some repetition. The distinction between refinement and compression, limitations of the 63-feature configuration, inability to causally attribute gains to VICReg, and scope of the SHAP proxy are repeated in several sections. These points are important, but the manuscript could be shortened without losing technical information.

14. **Formatting and language.** Perform a final formatting and language pass. I noticed several typesetting artifacts in the proof, including merged words such as "recentsupervised," "istherefore," "formalstatistical," "exceedsthe [...]".

15. **References (answer to question 4).** References [12] and [18] are given as arXiv preprints. If final peer-reviewed versions are available, the authors should cite those versions instead.

Additional remarks in the structured assessment (items A–D of question 2): sample-level CV remains a meaningful weakness; only one dataset is evaluated; no true component ablation; only one master experimental seed. Question 3: the paper is still quite long and occasionally repeats limitations and methodological distinctions.
