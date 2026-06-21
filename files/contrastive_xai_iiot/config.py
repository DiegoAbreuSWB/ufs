"""
config.py — Hiperparâmetros centralizados do projeto ContrastiveXAI-FS
Dataset: DataSense (CIC IIoT 2025)
"""

import os

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")

# Dataset files (arquivos estão em data/ diretamente)
BENIGN_CSV = os.path.join(DATA_DIR, "benign_samples_1sec.csv.csv")
FULL_CSV = os.path.join(DATA_DIR, "dataset_attack_benign.csv")
FULL_CSV_CLEAN = os.path.join(DATA_DIR, "dataset_attack_benign_numeric_clean.csv")

# ============================================================
# DATASET — DataSense specifics
# ============================================================
META_COLUMNS = [
    "device_name", "device_mac", "label_full",
    "label1", "label2", "label3", "label4",
    "timestamp", "timestamp_start", "timestamp_end",
    "label",   # coluna numérica extra presente no dataset completo
]

# Colunas de lista (string) — ignorar na seleção de features
LIST_COLUMNS = [
    "log_data-types",
    "network_ips_all", "network_ips_dst", "network_ips_src",
    "network_macs_all", "network_macs_dst", "network_macs_src",
    "network_ports_all", "network_ports_dst", "network_ports_src",
    "network_protocols_all", "network_protocols_dst", "network_protocols_src",
]

# Context Groups do paper DataSense (Firouzi et al., 2025)
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

# Features selecionadas pelo DataSense paper (Table 7) — para comparação
DATASENSE_SELECTED_17 = [
    "log_messages_count",
    "log_data-ranges_avg",
    "log_data-types_count",
    "network_fragmented-packets",
    "network_interval-packets",
    "network_packets_all_count",
    "network_ips_dst_count",
    "network_ips_all_count",
    "network_macs_src_count",
    "network_packet-size_std_deviation",
    "network_ports_all_count",
    "network_protocols_all_count",
    "network_time-delta_avg",
    "network_ttl_avg",
    "network_window-size_avg",
    "network_ip-flags_max",
    "network_tcp-flags-psh_count",
]

# ============================================================
# ENCODER CONTRASTIVO
# ============================================================
ENCODER_HIDDEN_DIM = 256
ENCODER_LATENT_DIM = 128
ENCODER_PROJECTION_DIM = 512   # VICReg precisa de projector maior que NT-Xent
ENCODER_DROPOUT = 0.2
ENCODER_EPOCHS = 100
ENCODER_BATCH_SIZE = 512
ENCODER_LR = 1e-3
CONTRASTIVE_TEMPERATURE = 0.5

# ============================================================
# VICREG — substitui NT-Xent (Bardes et al., ICLR 2022)
# ============================================================
VICREG_LAMBDA = 25.0   # peso da invariância (MSE entre visões)
VICREG_MU = 25.0       # peso da variância (evita colapso dimensional)
VICREG_NU = 1.0        # peso da covariância (descorrelação das dimensões)
VICREG_EPS = 1e-4      # epsilon para estabilidade numérica no termo de variância

# Augmentação por grupos semânticos (GroupAwareAugmenter)
GROUP_MASK_PROB = 0.3          # fração de grupos a mascarar por visão
FEAT_DROPOUT_PROB = 0.1        # probabilidade de dropout por feature individual
AUGMENT_NOISE_SCALE = 0.05     # escala do ruído relativo ao std da feature

# ============================================================
# DATA AUGMENTATION (legado — IIoTAugmenter)
# ============================================================
AUGMENT_NOISE_STD = 0.05       # Fração do std da feature
AUGMENT_MASK_RATIO = 0.15      # 15% das features zeradas
AUGMENT_SCALE_RANGE = (0.9, 1.1)

# ============================================================
# BUSCA BIDIRECIONAL
# ============================================================
K_RANGE = range(2, 15)         # Range de k para k-Means
MAX_SEARCH_ITERATIONS = 200    # Limite para evitar loop infinito
SILHOUETTE_SAMPLE_SIZE = 10000 # Amostra para calcular Silhouette (eficiência)

# ============================================================
# XAI / SHAP
# ============================================================
SHAP_PROXY_ESTIMATORS = 200
SHAP_PROXY_MAX_DEPTH = 10
SHAP_MAX_SAMPLES = 5000        # Para acelerar TreeExplainer

# ============================================================
# AVALIAÇÃO
# ============================================================
CV_FOLDS = 5
RANDOM_STATE = 42
MAX_SUBSAMPLE = 100000         # Amostragem máxima para treino do encoder

# Cenários de classificação
SCENARIOS = {
    "binary": {"label_col": "label1", "description": "benign vs attack"},
    "multiclass_8": {"label_col": "label2", "description": "8 categorias"},
    "multiclass_50": {"label_col": "label4", "description": "50 tipos finos"},
}
