# Relatório dos experimentos da 2ª revisão (uso interno)

Todos os números abaixo vêm de `results/revision/tables/*.csv`, gerados por `revision/rev_analysis.py`
a partir das execuções em `results/revision/runs/`. Nenhum número foi digitado à mão nas tabelas do artigo.

## Protocolo

Implementação nova (`revision/run_cv.py`), estritamente fold-interna: scaler, encoder VICReg, k*, busca,
k-Means, proxy SHAP e classificadores ajustados só no treino de cada fold. Hiperparâmetros da Tabela 9;
encoder treinado em subamostra de 20.480 linhas; objetivo estimado em 5.000/2.000 linhas.

| Execução | Folds | Status |
|---|---|---|
| DataSense, amostra, seed 42 (todos os métodos, RF/DT/KNN, orçamentos) | 10 | ok |
| DataSense, amostra, seeds 43 e 44 (proposto + All, RF/DT) | 20 | ok |
| DataSense, agrupado por execução (corrigido) | 5 | ok |
| DataSense, agrupado por dispositivo | 5 | ok |
| DataSense, ablação mascaramento independente | 5 | ok |
| DataSense, configuração descritiva (dados completos) | 1 | ok |
| N-BaIoT (log, pré-registrado, commit a80d31a) | 10 | ok |
| WUSTL-IIoT-2021 (log) | 10 | ok |
| WUSTL-IIoT-2021 (sem log, reprodução direta) | 10 | ok, não tabulado no artigo |

## Resultados principais (RF, macro-F1)

| | Binário | Multiclasse | Nº features |
|---|---|---|---|
| DataSense proposto (seed 42) | 0,936 | 0,908 | 59,2 (28–70) |
| DataSense All Features | 0,936 | 0,911 | 71 |
| DataSense 3 seeds (média ± dp das médias) | 0,935 ± 0,001 | 0,902 ± 0,005 | 55,7 |
| N-BaIoT proposto / All | 0,9997 / 0,9998 | 0,873 / 0,881 | 74,2 / 115 |
| WUSTL proposto / All | 0,974 / 0,9999 | 0,901 / 0,953 | 15,6 / 41 |

Testes pré-registrados (Tabela 19): equivalência (TOST ±0,01) a All Features no DataSense e no N-BaIoT;
superior ao MCFS no DataSense e no N-BaIoT; inconclusivo no WUSTL.

## Demais análises

- Estabilidade (30 subconjuntos): mediana 61; Jaccard 0,69 (raw: 0,40); Nogueira 0,18; 46 features em ≥ 80% dos folds; time-delta removidas em 67–73%.
- k*: 14 em 23 folds, 13 em 5, 12 e 11 em 1 cada. Silhouette cresce até k = 30 (0,567 → 0,622).
- Clusters: ARI intra-fold 0,964; entre 30 pipelines 0,944.
- Proxy: teste externo acc 0,989, bal 0,970; descritiva hold-out acc 0,988, bal 0,980.
- Agrupado por execução: binário 0,931 vs 0,933 (equivalente); 8 classes 0,793 vs 0,806.
- Agrupado por dispositivo: binário 0,903 vs 0,913; 8 classes 0,653 vs 0,691 (Web F1 = 0).
- Orçamentos (8 classes): proposto 0,536/0,802/0,862/0,878; MCFS 0,881/0,897/0,899/0,901.
- Ablação: latente > bruto em k = 15–25 (sig. só em k = 20, p = 0,016); bidirecional > backward-only (+0,027, n.s.); mascaramento independente ≈ por grupo (0,913 vs 0,908, n.s., 5 folds).
- Clareza: C12, C6, C8, C4 (CI ≥ 0,385; próximo 0,333).

## Correções feitas durante a campanha

1. Bug de timestamp no protocolo por execução (pandas 3, resolução em µs): todo o benigno caía num fold. Corrigido e reexecutado; resultados inválidos guardados em `results/revision/_invalid_group_exec_k5_timestamp_bug/`.
2. Silhouette amostrado podia receber 1 cluster na amostra → erro; agora rejeita com −1 (como especificado no artigo).
3. Comparações de orçamento usavam subconjuntos menores que k quando a busca natural parava antes; agora só folds com exatamente k features (contagem na tabela).
