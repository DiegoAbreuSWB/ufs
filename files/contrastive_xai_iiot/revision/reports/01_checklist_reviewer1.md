# Relatório inicial — Checklist dos pedidos do Reviewer 1

**Manuscrito:** Access-2026-39805 — *ContrastiveXAI-FS: Explainable Unsupervised Feature Selection for IIoT Intrusion Detection*
**Decisão:** 06/09/2026 — rejeição com direito a **uma única** ressubmissão (processo binário do IEEE Access).
**Reviewer 2:** recomenda publicação, sem pedidos. **Reviewer 1:** 14 pedidos + 1 ponto bibliográfico.

## 0. Achado crítico anterior a qualquer pedido do revisor

A leitura do código (`src/`, `experiments/`) e da auditoria interna (`audit/pipeline_trace.md`, `audit/rerun_decision_matrix.md`, de 28/07) mostra que **os números do manuscrito submetido não foram produzidos pelo protocolo que o texto descreve**:

| O manuscrito afirma | O que o código que gerou os resultados faz |
|---|---|
| Scaler, encoder, seleção e clusters ajustados dentro de cada fold de treino | `StandardScaler` global; encoder treinado em subamostra do dataset inteiro; **uma única** seleção global (63/71) reutilizada em todos os folds; só o classificador downstream é fold-wise |
| 10-fold CV nas Tabelas 8, 10 e 11 | 5-fold nas tabelas principais; 10-fold apenas no Wilcoxon |
| Wilcoxon bicaudal, p = 0,97 / 0,98 vs. All-71 | Os valores 0,97/0,98 são os p **unilaterais**; os bicaudais salvos em `wilcoxon_results.csv` são 0,084 (binário) e **0,037** (8 classes, a favor de All-71) |
| J(S) = max_k Sil (Eq. 17) | k* é fixado uma vez (com todas as features) e mantido durante a busca |
| Silhouette exato; aproximações "não avaliadas" | k-Means em subamostra de 10.000 linhas; Silhouette em 2.000 |
| Silhouette latente 0,9711 | `contrastive_result.json` registra 0,668 |
| Orçamento fixo: "melhor subconjunto sob o orçamento" | top-k por variância dentro das 63 selecionadas; a variância é calculada **após** z-score (≈1 para todas), logo a ordem é ruído numérico. O mesmo vale para o baseline "Variance" |
| Proxy RF "94,2% agreement" | acurácia de treino (mesma amostra de ajuste) |

O Reviewer 1 elogia explicitamente o protocolo fold-wise ("addresses the earlier concern..."). Manter os números atuais sob essa descrição não é defensável. **Decisão adotada nesta revisão:** reimplementar o pipeline de forma estritamente fold-interna (`revision/`), reexecutar toda a campanha e substituir as tabelas pelos resultados reexecutados. Isso também entrega, de forma natural, estabilidade entre folds, múltiplas seeds, perfil de Silhouette e estabilidade de clusters.

## 1. Pedidos do Reviewer 1, classificados

Legenda — **E**: exige novo experimento/análise · **D**: tabela/figura derivada · **T**: ajuste textual · **F**: formatação/referências.

| # | Pedido do Reviewer 1 | Tipo | Ação planejada |
|---|---|---|---|
| R1.1 | Reforçar a discussão da CV sample-level (dispositivo, sessão, execução de ataque, vizinhança temporal); se não houver avaliação agrupada, manter a limitação em destaque no abstract/conclusion | **E + T** | Avaliação agrupada é viável: `label_full` identifica 936 execuções de ataque e `device_name` os 38 dispositivos. Rodar CV agrupada por execução/bloco temporal e por dispositivo, com seleção fold-interna. Quantificar duplicatas exatas entre treino e teste. Limitação no abstract e na conclusão. |
| R1.2 | Validação em outro dataset IoT/IIoT; se não, claims conservadores | **E** | WUSTL-IIoT-2021 (testbed IIoT/SCADA, 1,19 M fluxos, 41 atributos numéricos, download público direto). Context Groups reconstruídos pela semântica das medidas. CICIoT2023 exige formulário de cadastro — não utilizado. |
| R1.3 | Estabilidade da seleção entre folds: tamanhos, frequência por feature, Jaccard | **E/D** | Derivado das execuções fold-wise (3 seeds × 10 folds). |
| R1.4 | Ablação controlada (VICReg padrão vs. group-aware; backward-only vs. bidirecional; raw vs. latente no mesmo orçamento; group masking vs. corrupção independente) | **E** | Escala reduzida (5 folds): group masking vs. mascaramento independente com mesma taxa esperada vs. sem mascaramento; backward-only vs. bidirecional; raw vs. latente nos mesmos orçamentos. |
| R1.5 | 3 a 5 master seeds independentes | **E** | 3 seeds (42, 43, 44) × 10 folds — mínimo do intervalo pedido, limitado pelo hardware (CPU, sem GPU). |
| R1.6 | Fidelidade held-out do proxy SHAP | **E** | Proxy treinado em amostra balanceada por cluster; fidelidade medida em hold-out interno disjunto e no fold de teste externo. |
| R1.7 | Valor prático de reter 63/71; não sugerir economia computacional | **T** | Reescrever sem promessa de economia; valor = refinamento + seleção guiada por representação + explicabilidade. |
| R1.8 | Dar ênfase ao resultado de baixo orçamento (k=10) | **T (+E)** | Manter interpretação franca; Tabela 10 regenerada com procedimento de orçamento explicitamente definido. |
| R1.9 | Justificar k=13: perfil de Silhouette nos k candidatos | **E/D** | Perfil k = 2..14 por fold/seed + figura. |
| R1.10 | Estabilidade dos clusters entre seeds/folds | **E** | ARI/AMI por reamostragem dentro do fold e entre pipelines de folds/seeds distintos sobre um conjunto de referência comum. |
| R1.11 | Critério quantitativo para "clearest SHAP profiles" (4 de 13) | **E** | Critério: fidelidade held-out por cluster + concentração do perfil SHAP; tabela com **todos** os clusters. |
| R1.12 | Disponibilizar os artefatos prometidos | **E/T** | Pacote de reprodutibilidade (`revision/`: scripts, seeds, configuração, pré-processamento, tabelas). A publicação em repositório público depende de ação do autor. |
| R1.13 | Reduzir repetição (refinement vs. compression; limitações das 63 features; atribuição causal; escopo do proxy) | **T** | Consolidar cada ressalva em um único lugar. |
| R1.14 | Passada final de formatação e linguagem (palavras coladas) | **F** | Revisão de `\texttt`/espaçamento; verificação automática de palavras coladas no PDF. |
| R1.15 | Refs. [12] e [18] como arXiv → versão publicada | **F** | VICReg e SCARF: ambos ICLR 2022. |

## 2. Itens adicionais identificados (não pedidos, mas necessários)

- Corrigir Eq. (9)–(13) e Eq. (17)–(18) para refletirem a implementação (normalizações do VICReg; k* fixo durante a busca).
- Declarar as subamostras usadas (encoder, k-Means, Silhouette).
- Baseline Variance: calcular em escala min-max do fold de treino (após z-score é degenerado).
- Reportar p-valores bicaudais corretos.
- Incluir a distribuição de classes do dataframe avaliado (prometida na resposta anterior e ausente do manuscrito).
- As Figuras 1 e 2 contêm rótulos ("Reconnaissance", "Tunneling", "TTL Evasion", "max_k") que contradizem o texto conservador; precisam ser editadas no arquivo-fonte das figuras.

## 3. Ordem de execução

1. Novo dataset (WUSTL-IIoT-2021).
2. DataSense fold-wise, seed 42, todos os métodos.
3. Seeds 43 e 44.
4. CV agrupada (execução/bloco temporal; dispositivo).
5. Ablação controlada.
6. Configuração descritiva + SHAP (fidelidade held-out, critério de clareza).
7. Somente então: manuscrito, versão destacada e resposta aos revisores.
