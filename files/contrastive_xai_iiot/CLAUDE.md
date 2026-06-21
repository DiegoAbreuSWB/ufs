# CLAUDE.md — Instruções para Claude Code

## O que é este projeto

Sistema de **Seleção de Features Não Supervisionada para IDS em IIoT** que combina:
1. **Contrastive Learning** (encoder neural) para aprender representações
2. **Busca bidirecional com Silhouette** para selecionar features (evolução da dissertação de mestrado do autor)
3. **SHAP** para explicar por que cada feature foi selecionada (XAI)

O dataset é o **DataSense** (CIC IIoT 2025) — tráfego de rede + dados de sensores industriais com 50 tipos de ataque.

## Arquivos de referência obrigatórios

Ler ANTES de implementar qualquer módulo:
- `ROTEIRO_CLAUDE_CODE_DataSense.md` — Roteiro completo com especificações de cada módulo
- `config.py` — Todos os hiperparâmetros e constantes
- `verify_dataset.py` — Rodar primeiro para validar o dataset

## Regras de implementação

1. **Seleção de features é NÃO SUPERVISIONADA** — nunca usar labels (label1..label4) durante o encoder, augmenter, busca bidirecional ou SHAP. Labels só aparecem na avaliação final com classificadores.

2. **Anti-leakage** — em cross-validation, treinar encoder e selecionar features APENAS no fold de treino.

3. **Eficiência** — dataset pode ter >100K linhas. Usar `SILHOUETTE_SAMPLE_SIZE` do config para subamostrar no cálculo de Silhouette. Cachear resultados da busca bidirecional.

4. **Baselines** — implementar na ordem: Variância → Laplacian Score → SPEC → MCFS → UDFS → NDFS → PCA → k-Means+Silhouette original. Tentar `pip install skfeature-chappers` para os métodos espectrais; se falhar, implementar manualmente.

5. **Colunas de lista** (network_ips_all, etc.) — IGNORAR. Usar apenas as colunas `_count` correspondentes que já existem no dataset.

6. **Context Groups** — definidos em `config.py`. Usar para agrupar features nas visualizações e no relatório XAI.

## Ordem de execução

```
1. python verify_dataset.py data/raw/benign_samples_1sec.csv
2. Implementar src/preprocessing.py
3. Implementar src/contrastive_fs.py (encoder + busca)
4. Implementar src/xai_explainer.py (SHAP)
5. Implementar src/baselines.py (9 baselines)
6. Implementar src/evaluation.py (classificadores + métricas)
7. Rodar experiments/01..06 em sequência
```

## Dependências

```bash
pip install torch scikit-learn numpy pandas shap matplotlib seaborn xgboost tqdm scipy
pip install skfeature-chappers  # opcional, para baselines UFS
```

## Estrutura do dataset DataSense

- 94 colunas total: 10 meta + 13 listas + 71 numéricas (68 com variância > 0)
- Labels: label1 (binário), label2 (8 classes), label4 (50 classes)
- 38 dispositivos IIoT/IoT (sensores, câmeras, smart plugs, infraestrutura)
- Janelas temporais de 1 segundo
