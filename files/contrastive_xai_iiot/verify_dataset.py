"""
verify_dataset.py — Script de verificação do dataset DataSense
Rodar ANTES de qualquer implementação para validar o ambiente e os dados.

Uso: python verify_dataset.py <caminho_do_csv>
"""

import sys
import pandas as pd
import numpy as np
from collections import Counter

# Importar config (ajustar path se necessário)
sys.path.insert(0, '.')
try:
    from config import (META_COLUMNS, LIST_COLUMNS, CONTEXT_GROUPS, 
                         DATASENSE_SELECTED_17)
except ImportError:
    print("AVISO: config.py não encontrado. Usando valores padrão.")
    META_COLUMNS = [
        "device_name", "device_mac", "label_full",
        "label1", "label2", "label3", "label4",
        "timestamp", "timestamp_start", "timestamp_end",
    ]
    LIST_COLUMNS = []


def verify_dataset(filepath: str):
    print(f"{'='*60}")
    print(f"VERIFICAÇÃO DO DATASET DATASENSE")
    print(f"{'='*60}")
    print(f"Arquivo: {filepath}\n")
    
    # 1. Carregar
    print("[1/8] Carregando CSV...")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"  Shape: {df.shape} ({df.shape[0]:,} linhas × {df.shape[1]} colunas)")
    print(f"  Memória: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    
    # 2. Colunas
    print(f"\n[2/8] Verificando colunas...")
    meta_found = [c for c in META_COLUMNS if c in df.columns]
    meta_missing = [c for c in META_COLUMNS if c not in df.columns]
    print(f"  Meta colunas encontradas: {len(meta_found)}/{len(META_COLUMNS)}")
    if meta_missing:
        print(f"  FALTANDO: {meta_missing}")
    
    # 3. Tipos de dados
    print(f"\n[3/8] Tipos de dados...")
    str_cols = [c for c in df.columns if df[c].dtype == 'object' or str(df[c].dtype) == 'string']
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in num_cols if c not in META_COLUMNS]
    print(f"  Colunas string/object: {len(str_cols)}")
    print(f"  Colunas numéricas total: {len(num_cols)}")
    print(f"  Features numéricas (excluindo meta): {len(feature_cols)}")
    
    # 4. Labels
    print(f"\n[4/8] Distribuição de labels...")
    for label_col in ['label1', 'label2', 'label3', 'label4']:
        if label_col in df.columns:
            vc = df[label_col].value_counts()
            print(f"  {label_col}: {len(vc)} classes")
            for cls, cnt in vc.head(10).items():
                pct = cnt / len(df) * 100
                print(f"    {cls}: {cnt:,} ({pct:.1f}%)")
            if len(vc) > 10:
                print(f"    ... e mais {len(vc) - 10} classes")
    
    # 5. Dispositivos
    print(f"\n[5/8] Dispositivos...")
    if 'device_name' in df.columns:
        devices = df['device_name'].nunique()
        print(f"  Dispositivos únicos: {devices}")
    
    # 6. NaN e Inf
    print(f"\n[6/8] Qualidade dos dados numéricos...")
    nan_total = df[feature_cols].isnull().sum().sum()
    inf_count = np.isinf(df[feature_cols].select_dtypes(include=[np.number])).sum().sum()
    print(f"  NaN total: {nan_total:,}")
    print(f"  Inf total: {inf_count:,}")
    if nan_total > 0:
        nan_cols = df[feature_cols].isnull().sum()
        nan_cols = nan_cols[nan_cols > 0].sort_values(ascending=False)
        print(f"  Colunas com NaN:")
        for col, cnt in nan_cols.head(5).items():
            print(f"    {col}: {cnt:,}")
    
    # 7. Variância zero
    print(f"\n[7/8] Features com variância zero...")
    zero_var = []
    for c in feature_cols:
        if df[c].std() == 0:
            zero_var.append(c)
    print(f"  Features variância zero: {len(zero_var)}")
    if zero_var:
        for c in zero_var:
            print(f"    {c} (valor constante: {df[c].iloc[0]})")
    
    usable = len(feature_cols) - len(zero_var)
    print(f"\n  FEATURES UTILIZÁVEIS: {usable}")
    
    # 8. Context Groups
    print(f"\n[8/8] Verificando Context Groups...")
    mapped = 0
    unmapped = []
    for c in feature_cols:
        if c in zero_var:
            continue
        found = False
        for group_name, group_cols in CONTEXT_GROUPS.items():
            if c in group_cols:
                found = True
                mapped += 1
                break
        if not found:
            unmapped.append(c)
    print(f"  Features mapeadas a grupos: {mapped}/{usable}")
    if unmapped:
        print(f"  Sem grupo: {unmapped}")
    
    for group_name, group_cols in CONTEXT_GROUPS.items():
        present = [c for c in group_cols if c in feature_cols and c not in zero_var]
        print(f"  {group_name}: {len(present)} features")
    
    # Resumo
    print(f"\n{'='*60}")
    print(f"RESUMO")
    print(f"{'='*60}")
    print(f"Total de linhas:          {df.shape[0]:,}")
    print(f"Features utilizáveis:     {usable}")
    print(f"Classes (label1):         {df['label1'].nunique() if 'label1' in df.columns else 'N/A'}")
    print(f"Classes (label2):         {df['label2'].nunique() if 'label2' in df.columns else 'N/A'}")
    print(f"Classes (label4):         {df['label4'].nunique() if 'label4' in df.columns else 'N/A'}")
    print(f"Dispositivos:             {df['device_name'].nunique() if 'device_name' in df.columns else 'N/A'}")
    print(f"NaN:                      {nan_total:,}")
    print(f"Inf:                      {inf_count:,}")
    print(f"Variância zero:           {len(zero_var)}")
    
    has_attacks = df['label1'].nunique() > 1 if 'label1' in df.columns else False
    print(f"Contém ataques:           {'SIM' if has_attacks else 'NÃO (apenas benign)'}")
    
    if not has_attacks:
        print(f"\nATENÇÃO: Este CSV contém apenas dados benignos.")
        print(f"Para rodar os experimentos completos, coloque o arquivo")
        print(f"'all_attack_benign_samples.csv' no diretório data/raw/")
    
    print(f"\nVerificação concluída com sucesso!")
    return df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Tentar caminhos padrão
        import os
        candidates = [
            "data/raw/all_attack_benign_samples.csv",
            "data/raw/benign_samples_1sec.csv",
        ]
        filepath = None
        for c in candidates:
            if os.path.exists(c):
                filepath = c
                break
        if filepath is None:
            print("Uso: python verify_dataset.py <caminho_do_csv>")
            print("Nenhum dataset encontrado nos caminhos padrão.")
            sys.exit(1)
    else:
        filepath = sys.argv[1]
    
    verify_dataset(filepath)
