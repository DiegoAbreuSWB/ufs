# ROTEIRO DE IMPLEMENTAÇÃO — ContrastiveXAI-FS para IIoT
## Dataset: DataSense (CIC IIoT 2025) | Técnica: Contrastive Clustering + SHAP

> **Instruções para o Claude Code:** Este é o roteiro completo. Leia-o inteiro antes de escrever qualquer código. Ele contém a análise do dataset, a arquitetura da proposta, os baselines a comparar, e a ordem exata de implementação.

---

## 1. ANÁLISE DO DATASET DATASENSE

### 1.1 Visão geral

O DataSense é um dataset de segurança IIoT criado pelo Canadian Institute for Cybersecurity (CIC/UNB) em 2025. Contém dados sincronizados de sensores industriais e tráfego de rede, com 50 tipos de ataque em 7 categorias.

**Arquivos disponíveis:**
- Arquivo benign: `/mnt/user-data/uploads/benign_samples_1sec.csv` (136.800 linhas)
- Arquivo completo (ataque + benigno): disponível localmente em `C:\Users\diego\Desktop\all_attack_benign_samples.csv`
  - O Claude Code deve solicitar ao usuário que coloque este arquivo no diretório `data/raw/`

### 1.2 Estrutura do CSV

O CSV tem **94 colunas** organizadas assim:

**Colunas de metadados (10) — NÃO usar como features:**
```
device_name, device_mac, label_full, label1, label2, label3, label4,
timestamp, timestamp_start, timestamp_end
```

**Colunas de lista/string (13) — Precisam de vetorização:**
```
log_data-types
network_ips_all, network_ips_dst, network_ips_src
network_macs_all, network_macs_dst, network_macs_src
network_ports_all, network_ports_dst, network_ports_src
network_protocols_all, network_protocols_dst, network_protocols_src
```
Estas contêm listas Python como strings (ex: `"['tcp', 'json', 'arp']"`).
Para usar como features numéricas, o paper DataSense usou `_count` (que já existe).
**Decisão: IGNORAR as colunas de lista. Usar apenas as colunas `_count` correspondentes.**

**Colunas numéricas (71) — Features utilizáveis:**
68 com variância > 0, 3 com variância zero (remover).

### 1.3 Features com variância zero (REMOVER)
```
network_fragmentation-score   (sempre 0 no benign)
network_fragmented-packets    (sempre 0 no benign)
network_tcp-flags-urg_count   (sempre 0 no benign)
```
**ATENÇÃO:** Essas features podem ter variância > 0 no dataset completo (com ataques). Verificar após carregar o dataset completo e remover apenas se variância == 0.

### 1.4 Mapeamento de features para Context Groups (do paper DataSense)

O paper DataSense definiu 9 grupos semânticos. Use este mapeamento para Context Grouping:

```python
CONTEXT_GROUPS = {
    "Log Data Rate": [
        "log_interval-messages",
        "log_messages_count",
    ],
    "Log Data Stats": [
        "log_data-ranges_avg",
        "log_data-ranges_max",
        "log_data-ranges_min",
        "log_data-ranges_std_deviation",
        "log_data-types_count",
    ],
    "Packet Traffic Rate": [
        "network_packets_all_count",
        "network_packets_dst_count",
        "network_packets_src_count",
        "network_interval-packets",
    ],
    "Fragmentation": [
        "network_fragmentation-score",
        "network_fragmented-packets",
    ],
    "Address Diversity": [
        "network_ips_all_count",
        "network_ips_dst_count",
        "network_ips_src_count",
        "network_macs_all_count",
        "network_macs_dst_count",
        "network_macs_src_count",
    ],
    "Header Flags": [
        "network_tcp-flags-ack_count",
        "network_tcp-flags-fin_count",
        "network_tcp-flags-psh_count",
        "network_tcp-flags-rst_count",
        "network_tcp-flags-syn_count",
        "network_tcp-flags-urg_count",
        "network_tcp-flags_avg",
        "network_tcp-flags_max",
        "network_tcp-flags_min",
        "network_tcp-flags_std_deviation",
        "network_ip-flags_avg",
        "network_ip-flags_max",
        "network_ip-flags_min",
        "network_ip-flags_std_deviation",
    ],
    "Timing Control": [
        "network_time-delta_avg",
        "network_time-delta_max",
        "network_time-delta_min",
        "network_time-delta_std_deviation",
        "network_ttl_avg",
        "network_ttl_max",
        "network_ttl_min",
        "network_ttl_std_deviation",
        "network_window-size_avg",
        "network_window-size_max",
        "network_window-size_min",
        "network_window-size_std_deviation",
    ],
    "Size Length": [
        "network_packet-size_avg",
        "network_packet-size_max",
        "network_packet-size_min",
        "network_packet-size_std_deviation",
        "network_header-length_avg",
        "network_header-length_max",
        "network_header-length_min",
        "network_header-length_std_deviation",
        "network_ip-length_avg",
        "network_ip-length_max",
        "network_ip-length_min",
        "network_ip-length_std_deviation",
        "network_mss_avg",
        "network_mss_max",
        "network_mss_min",
        "network_mss_std_deviation",
        "network_payload-length_avg",
        "network_payload-length_max",
        "network_payload-length_min",
        "network_payload-length_std_deviation",
    ],
    "Network Multiplexing": [
        "network_ports_all_count",
        "network_ports_dst_count",
        "network_ports_src_count",
        "network_protocols_all_count",
        "network_protocols_dst_count",
        "network_protocols_src_count",
    ],
}
```

### 1.5 Labels para avaliação

O dataset tem 4 níveis de granularidade de labels:

| Coluna | Descrição | Valores exemplo |
|--------|-----------|----------------|
| `label1` | Binário | benign, attack |
| `label2` | Categoria (8 classes) | benign, ddos, dos, recon, mitm, web, bruteforce, malware |
| `label3` | Subcategoria | benign, tcp_syn_flood, arp_spoofing, ... |
| `label4` | Tipo fino (50 classes) | benign, ddos_tcp_syn_flood, dos_slowloris, ... |

**IMPORTANTE:** Labels são usadas APENAS na avaliação final. NUNCA durante a seleção de features não supervisionada.

### 1.6 Escala do dataset

- Benign: 136.800 amostras (38 dispositivos × 3.600 janelas de 1 segundo)
- Dataset completo (com ataques): significativamente maior
- **Estratégia para eficiência:** Se o dataset completo for muito grande (>500K linhas), usar amostragem estratificada de ~100K linhas para treinar o encoder contrastivo. Usar o dataset completo para avaliação final.

---

## 2. ARQUITETURA DO SISTEMA

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline Completo                             │
│                                                                 │
│  ┌──────────┐   ┌───────────┐   ┌──────────┐   ┌────────────┐ │
│  │  Módulo 1 │──>│  Módulo 2  │──>│ Módulo 3 │──>│  Módulo 4  │ │
│  │  Prepro-  │   │ Contrastive│   │   XAI    │   │ Avaliação  │ │
│  │cessamento │   │ Feature    │   │ Explainer│   │ Comparativa│ │
│  │           │   │ Selection  │   │          │   │            │ │
│  └──────────┘   └───────────┘   └──────────┘   └────────────┘ │
│                                                                 │
│  Baselines: Laplacian Score, SPEC, MCFS, UDFS, NDFS, Variância,│
│             k-Means+Silhouette (dissertação), PCA, DataSense FS│
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. ESPECIFICAÇÃO MÓDULO A MÓDULO

### 3.1 Módulo 1 — Pré-processamento (`src/preprocessing.py`)

```python
class DataSensePreprocessor:
    """Carrega e prepara o dataset DataSense."""
    
    def __init__(self, filepath: str):
        """filepath: caminho do CSV (benign ou completo)."""
    
    def load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Retorna: (X_numeric, y_labels)
        
        Passos:
        1. Ler CSV com pandas
        2. Separar metadados: device_name, device_mac, timestamps, labels
        3. Remover colunas de lista/string (as 13 colunas de listas)
        4. Manter apenas colunas numéricas (float64 + int64)
        5. Remover features com variância zero
        6. Tratar NaN: substituir por 0 (NaN neste dataset indica ausência de tráfego)
        7. Tratar Inf: substituir por max finito da coluna
        8. Normalizar com StandardScaler
        
        X_numeric: shape (n_samples, ~68 features), numpy array normalizado
        y_labels: DataFrame com colunas label1..label4 (para avaliação)
        """
    
    def get_feature_names(self) -> list[str]:
        """Retorna nomes das features após limpeza."""
    
    def get_context_groups(self) -> dict[str, list[int]]:
        """
        Retorna mapeamento: nome_grupo -> lista de índices de features.
        Baseado no CONTEXT_GROUPS definido acima.
        Filtra apenas features que sobreviveram ao pré-processamento.
        """
    
    def subsample(self, X, y, n_samples: int = 100000, 
                  stratify_col: str = 'label2') -> tuple:
        """
        Amostragem estratificada para datasets grandes.
        Mantém proporção das classes em label2.
        """
```

### 3.2 Módulo 2 — Contrastive Feature Selection (`src/contrastive_fs.py`)

#### 3.2.1 Data Augmentation para IIoT

```python
class IIoTAugmenter:
    """
    Augmentações para dados tabulares de sensores IIoT.
    
    Estratégias (aplicar 2 das 3 aleatoriamente para cada view):
    1. Gaussian noise: ruído N(0, σ²) com σ = 0.05 * std da feature
       - Aplicar APENAS em features contínuas (avg, std, max, min)
       - NÃO aplicar em features de contagem (_count)
    2. Feature masking: zerar 15% das features aleatoriamente
       - Simula perda de dados de sensores (realista em IIoT)
    3. Scaling jitter: multiplicar por fator em [0.9, 1.1]
       - Aplicar apenas em features de volume/tamanho
    
    Detectar features de contagem: nome contém '_count' ou valor sempre inteiro
    Detectar features binárias: todos valores são 0 ou 1
    """
```

#### 3.2.2 Encoder Contrastivo

```python
class ContrastiveEncoder(nn.Module):
    """
    Arquitetura MLP para aprender representações de tráfego IIoT.
    
    Input: 68 features (varia conforme pré-processamento)
    ├── Linear(68, 256) → BatchNorm → ReLU → Dropout(0.2)
    ├── Linear(256, 128) → BatchNorm → ReLU → Dropout(0.2)  [backbone output]
    └── Linear(128, 64)  [projection head, apenas para training]
    
    forward() retorna (backbone_repr_128d, projection_64d)
    """
```

#### 3.2.3 Busca Bidirecional no Espaço Latente

```python
class ContrastiveBidirectionalFS:
    """
    Mesma lógica da dissertação, mas operando sobre representações aprendidas.
    
    Pipeline:
    1. Recebe X (features originais normalizadas)
    2. Para avaliar um subconjunto S de features:
       a. Retreinar encoder apenas com features em S (ou usar masked input)
       b. Obter representações latentes
       c. Aplicar k-Means nas representações
       d. Calcular Silhouette Score → retornar como fitness
    3. Busca bidirecional (SFS + SBS) para maximizar Silhouette
    
    OTIMIZAÇÃO IMPORTANTE — Para evitar retreinar o encoder a cada subconjunto:
    Abordagem alternativa: treinar encoder UMA VEZ com todas features.
    Depois, para cada subconjunto S, zerar as features fora de S no input
    e calcular o forward pass. Isso é ~100x mais rápido.
    
    Determinação automática de k:
    - Testar k ∈ {2, 3, ..., min(15, n_classes_conhecidas + 3)}
    - Escolher k que maximiza Silhouette médio
    - Cachear o k ótimo e reutilizar durante a busca
    """
```

### 3.3 Módulo 3 — XAI Explainer (`src/xai_explainer.py`)

```python
class ClusterSHAPExplainer:
    """
    Aplica SHAP no contexto não supervisionado.
    
    Pipeline:
    1. Recebe: X (features originais), cluster_labels (do k-Means latente)
    2. Treina Random Forest: X → cluster_labels (modelo proxy)
       - IMPORTANTE: usar max_depth=10 e n_estimators=200 para o proxy
       - Verificar acurácia do proxy (deve ser >85%, senão explicações são fracas)
    3. Aplica shap.TreeExplainer no Random Forest
    4. Gera SHAP values: shape (n_samples, n_features, n_clusters)
    
    Outputs:
    - Importância global (média |SHAP| por feature)
    - Importância por cluster (média |SHAP| por feature × cluster)
    - Comparação: features selecionadas vs não selecionadas
    """
    
    def plot_shap_summary(self, output_dir: str):
        """Beeswarm plot global."""
    
    def plot_cluster_waterfall(self, cluster_id: int, output_dir: str):
        """Waterfall plot para um cluster específico."""
    
    def plot_feature_heatmap(self, output_dir: str):
        """Heatmap: features × clusters, cor = SHAP médio."""
    
    def plot_tsne_latent(self, representations, cluster_labels, 
                          true_labels=None, output_dir: str = None):
        """
        t-SNE side-by-side: clusters aprendidos vs labels reais.
        Essencial para validar se o encoder capturou estrutura semântica.
        """
    
    def generate_text_report(self, selected_features, output_dir: str):
        """
        Gera relatório textual com:
        - Top 10 features mais importantes por SHAP global
        - Para cada cluster: top 5 features que o definem
        - Comparação: features selecionadas vs não-selecionadas
        - Coerência semântica: features de Header Flags importantes para clusters
          que correspondem a ataques de flood? Features de Address Diversity
          importantes para clusters de reconnaissance?
        """
```

### 3.4 Módulo 4 — Avaliação Comparativa (`src/evaluation.py`)

#### 3.4.1 Baselines a implementar

Baseado na survey de Solorio-Fernández et al. (2020) e no paper DataSense, implementar:

```python
BASELINES = {
    # ---- Filter Univariate ----
    "variance": {
        "description": "Ranking por variância de cada feature",
        "tipo": "filter-univariate",
        "referencia": "Baseline simples",
        "implementacao": "sklearn: VarianceThreshold ou manual"
    },
    "laplacian_score": {
        "description": "Laplacian Score (He et al., 2005)",
        "tipo": "filter-univariate-spectral",
        "referencia": "He, Cai, Niyogi. NIPS 2005",
        "implementacao": "skfeature.function.similarity_based.lap_score"
        # Instalar: pip install skfeature-chappers
        # Ou implementar: LS(f) = (f̃ᵀ L f̃) / (f̃ᵀ D f̃)
        # onde L = Laplacian, D = degree matrix, f̃ = feature centrada
    },
    "spec": {
        "description": "SPEC - Spectral Feature Selection (Zhao & Liu, 2007)",
        "tipo": "filter-univariate-spectral",
        "referencia": "Zhao, Liu. ICML 2007",
        "implementacao": "skfeature.function.similarity_based.SPEC"
    },
    
    # ---- Filter Multivariate ----
    "mcfs": {
        "description": "Multi-Cluster Feature Selection (Cai et al., 2010)",
        "tipo": "filter-multivariate-spectral-sparse",
        "referencia": "Cai, Zhang, He. KDD 2010",
        "implementacao": "skfeature.function.sparse_learning_based.MCFS"
    },
    "udfs": {
        "description": "Unsupervised Discriminative FS (Yang et al., 2011)",
        "tipo": "filter-multivariate-spectral-sparse",
        "referencia": "Yang et al. IJCAI 2011",
        "implementacao": "skfeature.function.sparse_learning_based.UDFS"
    },
    "ndfs": {
        "description": "Nonneg. Discriminative FS (Li et al., 2012)",
        "tipo": "filter-multivariate-spectral-sparse",
        "referencia": "Li et al. AAAI 2012",
        "implementacao": "skfeature.function.sparse_learning_based.NDFS"
    },
    
    # ---- Wrapper ----
    "kmeans_silhouette_original": {
        "description": "k-Means + Silhouette bidirecional (dissertação 2022)",
        "tipo": "wrapper-sequential",
        "referencia": "Abreu, D.M. Dissertação UFPA, 2022",
        "implementacao": "Implementar manualmente (é a proposta original)"
        # Diferença da proposta nova: usa k-Means DIRETAMENTE sobre features
        # brutas, sem encoder contrastivo
    },
    
    # ---- Redução de dimensionalidade ----
    "pca": {
        "description": "PCA mantendo 95% da variância",
        "tipo": "redução de dimensionalidade",
        "referencia": "Baseline clássico",
        "implementacao": "sklearn.decomposition.PCA(n_components=0.95)"
    },
    
    # ---- Sem seleção ----
    "all_features": {
        "description": "Todas as features sem seleção",
        "tipo": "baseline",
        "implementacao": "Usar X completo"
    },
}
```

#### 3.4.2 Classificadores para avaliação

```python
CLASSIFIERS = {
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "xgboost": XGBClassifier(n_estimators=100, random_state=42, 
                              use_label_encoder=False, eval_metric='mlogloss'),
}
```

#### 3.4.3 Cenários de avaliação (como no paper DataSense)

```python
SCENARIOS = {
    "binary": {
        "label_col": "label1",  # benign vs attack
        "n_classes": 2
    },
    "multiclass_8": {
        "label_col": "label2",  # 8 categorias
        "n_classes": 8
    },
    "multiclass_50": {
        "label_col": "label4",  # 50 tipos finos
        "n_classes": 50
    },
}
```

#### 3.4.4 Métricas

```python
METRICS = {
    # Métricas de classificação (avaliação externa)
    "f1_macro": f1_score(y_true, y_pred, average='macro'),
    "accuracy": accuracy_score(y_true, y_pred),
    "precision_macro": precision_score(y_true, y_pred, average='macro'),
    "recall_macro": recall_score(y_true, y_pred, average='macro'),
    "mcc": matthews_corrcoef(y_true, y_pred),
    
    # Métricas de clustering (avaliação interna)
    "silhouette": silhouette_score(X_selected, cluster_labels),
    "nmi": normalized_mutual_info_score(y_true, cluster_labels),  # usa label
    "acc_cluster": cluster_accuracy(y_true, cluster_labels),       # Hungarian matching
    
    # Métricas de eficiência
    "n_features_selected": len(selected_features),
    "reduction_ratio": 1 - len(selected) / len(all_features),
    "selection_time_seconds": elapsed,
}
```

---

## 4. PROTOCOLO EXPERIMENTAL

### 4.1 Experimento 1 — Reproduzir baseline (dissertação original)

```
Para cada cenário (binary, 8-class):
  1. Pré-processar DataSense
  2. Executar k-Means + Silhouette bidirecional sobre features brutas (SEM labels)
  3. Avaliar features selecionadas com 4 classificadores (COM labels)
  4. Reportar: F1-macro, accuracy, n_features, tempo de seleção
```

### 4.2 Experimento 2 — Método proposto completo

```
Para cada cenário (binary, 8-class):
  1. Pré-processar DataSense
  2. Treinar encoder contrastivo (SEM labels, 100 epochs)
  3. Obter representações latentes
  4. Executar busca bidirecional no espaço latente (SEM labels)
  5. Calcular SHAP no modelo proxy (SEM labels na seleção)
  6. Avaliar features selecionadas com 4 classificadores (COM labels)
  7. Gerar visualizações XAI
  8. Reportar todas as métricas
```

### 4.3 Experimento 3 — Comparação com baselines

```
Para cada baseline (9 métodos):
  Para cada cenário (binary, 8-class):
    1. Aplicar método de seleção (SEM labels)
    2. Selecionar top-k features (onde k = número selecionado pelo método proposto)
    3. Avaliar com 4 classificadores
    4. Reportar métricas
```

### 4.4 Experimento 4 — Ablation study

```
Config A: k-Means puro + Silhouette bidirecional (dissertação)
Config B: Encoder contrastivo + Silhouette bidirecional (sem XAI)
Config C: Encoder contrastivo + Silhouette bidirecional + SHAP (proposta completa)
Config D: k-Means puro + SHAP (sem contrastive)

Comparar F1-macro de cada configuração para isolar contribuições.
```

### 4.5 Cross-validation

```
Usar 5-Fold Stratified Cross-Validation para todos os experimentos.

CRÍTICO — Protocolo anti-leakage:
1. Dividir dados em 5 folds estratificados por label
2. Para CADA fold:
   a. Treinar encoder contrastivo APENAS no fold de treino (sem labels)
   b. Selecionar features APENAS no fold de treino (sem labels)
   c. Treinar classificador no fold de treino (com labels)
   d. Avaliar no fold de teste
3. Reportar média ± desvio padrão das 5 folds
```

---

## 5. ESTRUTURA DE DIRETÓRIOS

```
contrastive_xai_iiot/
├── README.md
├── requirements.txt
├── config.py                    # Hiperparâmetros centralizados
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py         # Módulo 1
│   ├── contrastive_fs.py        # Módulo 2 (encoder + busca bidirecional)
│   ├── xai_explainer.py         # Módulo 3 (SHAP + visualizações)
│   ├── evaluation.py            # Módulo 4 (classificadores + métricas)
│   └── baselines.py             # Todos os baselines UFS
│
├── experiments/
│   ├── 01_run_baseline_original.py   # Reproduz dissertação no DataSense
│   ├── 02_run_contrastive.py         # Método proposto completo
│   ├── 03_run_all_baselines.py       # Roda todos os baselines UFS
│   ├── 04_run_comparison.py          # Gera tabela comparativa final
│   ├── 05_run_ablation.py            # Ablation study
│   └── 06_run_xai_analysis.py        # Análise XAI detalhada
│
├── data/
│   └── raw/                     # Colocar CSVs aqui
│       └── all_attack_benign_samples.csv
│
├── results/
│   ├── figures/
│   │   ├── shap_summary.png
│   │   ├── tsne_clusters_vs_labels.png
│   │   ├── cluster_heatmap.png
│   │   ├── bidirectional_search_progress.png
│   │   └── comparison_barplot.png
│   ├── tables/
│   │   ├── comparison_binary.csv
│   │   ├── comparison_8class.csv
│   │   ├── ablation_results.csv
│   │   └── selected_features.csv
│   └── models/
│       └── encoder_datasense.pt
│
└── tests/
    ├── test_preprocessing.py
    ├── test_contrastive.py
    └── test_evaluation.py
```

---

## 6. DEPENDÊNCIAS

```
# requirements.txt
torch>=2.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
shap>=0.43.0
matplotlib>=3.7.0
seaborn>=0.12.0
xgboost>=2.0.0
tqdm>=4.65.0
scipy>=1.11.0
scikit-feature-ufs>=1.0.0   # ou: pip install skfeature-chappers
```

Se `skfeature` não estiver disponível, implementar Laplacian Score, SPEC, MCFS, UDFS e NDFS manualmente a partir das fórmulas dos papers originais. As implementações são relativamente simples (10-50 linhas cada para as versões básicas).

---

## 7. HIPERPARÂMETROS (`config.py`)

```python
# Encoder Contrastivo
ENCODER_HIDDEN_DIM = 256
ENCODER_LATENT_DIM = 128
ENCODER_PROJECTION_DIM = 64
ENCODER_DROPOUT = 0.2
ENCODER_EPOCHS = 100
ENCODER_BATCH_SIZE = 512
ENCODER_LR = 1e-3
CONTRASTIVE_TEMPERATURE = 0.5

# Data Augmentation
AUGMENT_NOISE_STD = 0.05
AUGMENT_MASK_RATIO = 0.15
AUGMENT_SCALE_RANGE = (0.9, 1.1)

# Busca Bidirecional
K_RANGE = range(2, 15)
MAX_SEARCH_ITERATIONS = 200  # Limite para evitar loop infinito

# SHAP
SHAP_PROXY_ESTIMATORS = 200
SHAP_PROXY_MAX_DEPTH = 10
SHAP_MAX_SAMPLES = 5000  # Para acelerar cálculo SHAP em datasets grandes

# Avaliação
CV_FOLDS = 5
RANDOM_STATE = 42
```

---

## 8. ORDEM DE IMPLEMENTAÇÃO (CHECKLIST)

Implementar nesta ordem exata. Cada fase depende da anterior.

### Fase 1 — Setup e pré-processamento
- [ ] Criar estrutura de diretórios
- [ ] Criar `requirements.txt` e `config.py`
- [ ] Implementar `DataSensePreprocessor`
- [ ] Testar com `benign_samples_1sec.csv`
- [ ] Verificar: 68 features numéricas, sem NaN, normalização correta

### Fase 2 — Baseline original (dissertação)
- [ ] Implementar busca bidirecional com k-Means + Silhouette sobre features brutas
- [ ] Implementar cache de subconjuntos avaliados
- [ ] Rodar `01_run_baseline_original.py` nos dados benign (validação do pipeline)

### Fase 3 — Encoder contrastivo
- [ ] Implementar `IIoTAugmenter` com 3 augmentações
- [ ] Implementar `ContrastiveEncoder` (MLP)
- [ ] Implementar `nt_xent_loss`
- [ ] Implementar loop de treinamento contrastivo
- [ ] Treinar no benign e verificar: loss decresce, representações não colapsam

### Fase 4 — Busca bidirecional no espaço latente
- [ ] Adaptar busca bidirecional para operar com encoder
- [ ] Testar no benign: features selecionadas devem ser < 50% do total

### Fase 5 — XAI/SHAP
- [ ] Implementar `ClusterSHAPExplainer` com modelo proxy
- [ ] Gerar SHAP summary plot
- [ ] Gerar t-SNE side-by-side
- [ ] Gerar heatmap features × clusters
- [ ] Gerar relatório textual

### Fase 6 — Baselines UFS
- [ ] Implementar ou integrar: Variância, Laplacian Score, SPEC
- [ ] Implementar ou integrar: MCFS, UDFS, NDFS
- [ ] Implementar: PCA baseline
- [ ] Wrapper em `baselines.py` com interface uniforme

### Fase 7 — Avaliação comparativa
- [ ] Implementar `EvaluationPipeline` com 5-fold CV
- [ ] Rodar todos os métodos × classificadores × cenários
- [ ] Gerar tabela comparativa (similar à Table 8 do paper DataSense)
- [ ] Gerar gráficos de barras comparativos

### Fase 8 — Com dataset completo
- [ ] Solicitar ao usuário o arquivo completo
- [ ] Rodar pipeline completo com dados de ataque
- [ ] Ablation study
- [ ] Resultados finais

---

## 9. SAÍDAS ESPERADAS

### 9.1 Tabela comparativa principal (formato)

```
| Método               | Tipo        | #Features | Silhouette | F1-binary | F1-8class | Tempo(s) |
|----------------------|-------------|-----------|------------|-----------|-----------|----------|
| All features         | baseline    | 68        | —          | X.XXX     | X.XXX     | —        |
| Variance ranking     | filter-univ | K         | —          | X.XXX     | X.XXX     | X.X      |
| Laplacian Score      | filter-univ | K         | —          | X.XXX     | X.XXX     | X.X      |
| SPEC                 | filter-univ | K         | —          | X.XXX     | X.XXX     | X.X      |
| MCFS                 | filter-mult | K         | —          | X.XXX     | X.XXX     | X.X      |
| UDFS                 | filter-mult | K         | —          | X.XXX     | X.XXX     | X.X      |
| NDFS                 | filter-mult | K         | —          | X.XXX     | X.XXX     | X.X      |
| PCA (95%)            | dim-reduct  | K         | —          | X.XXX     | X.XXX     | X.X      |
| k-Means+Sil (2022)   | wrapper     | K         | X.XXX      | X.XXX     | X.XXX     | X.X      |
| **Proposta (C+XAI)** | **wrapper** | **K**     | **X.XXX**  | **X.XXX** | **X.XXX** | **X.X**  |
```

### 9.2 Análise XAI esperada

O relatório XAI deve responder:
1. "Quais features o método selecionou?" → Lista ordenada por SHAP
2. "Por que essas features?" → SHAP values mostram contribuição por cluster
3. "Clusters correspondem a tipos de ataque?" → t-SNE colorido por label real
4. "Features fazem sentido semântico?" → Ex: `network_tcp-flags-syn_count` importante para cluster que contém TCP SYN Flood

---

## 10. REFERÊNCIAS PARA O CLAUDE CODE

- **Dissertação original:** Abreu, D.M. (2022) — k-Means + Silhouette bidirecional
- **DataSense paper:** Firouzi et al. (2025) — Dataset, Context Groups, RRS-guided GA
- **Survey UFS:** Solorio-Fernández et al. (2020) — Taxonomia completa de métodos UFS
- **SimCLR:** Chen et al. (2020) — NT-Xent loss, contrastive framework
- **SHAP:** Lundberg & Lee (2017) — TreeExplainer
- **C-SHAP:** Wójcik et al. (2025) — SHAP + k-Means clustering
- **Laplacian Score:** He, Cai, Niyogi (NIPS 2005)
- **SPEC:** Zhao, Liu (ICML 2007)
- **MCFS:** Cai, Zhang, He (KDD 2010)
- **UDFS:** Yang et al. (IJCAI 2011)
- **NDFS:** Li et al. (AAAI 2012)
