"""Generate semantic group coverage table for paper."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import CONTEXT_GROUPS

with open(os.path.join(os.path.dirname(__file__), 'contrastive_result.json')) as f:
    contra = json.load(f)
with open(os.path.join(os.path.dirname(__file__), 'baseline_original_result.json')) as f:
    raw_bd = json.load(f)
with open(os.path.join(os.path.dirname(__file__), 'baselines_results.json')) as f:
    baselines = json.load(f)

methods = {
    'Contrastive*': set(contra['selected_features']),
    'kMeansSil-raw': set(raw_bd['selected_features']),
    'Variance': set(baselines['baselines']['variance']['selected_features']),
    'LaplScore': set(baselines['baselines']['laplacian_score']['selected_features']),
    'SPEC': set(baselines['baselines']['spec']['selected_features']),
    'MCFS': set(baselines['baselines']['mcfs']['selected_features']),
    'NDFS': set(baselines['baselines']['ndfs']['selected_features']),
}

all_feats = set(baselines['baselines']['all_features']['selected_features'])
k_values = [25, 46, 25, 25, 25, 25, 25]

print()
print('TABELA DE COBERTURA SEMANTICA POR CONTEXT GROUP')
print('='*95)
hdr = '{:<26}'.format('Grupo (total feats)')
for m, k in zip(methods, k_values):
    hdr += '{:>13}'.format(m + '({})'.format(k))
print(hdr)
print('-'*95)

group_rows = []
for group, feats_in_group in CONTEXT_GROUPS.items():
    avail = [f for f in feats_in_group if f in all_feats]
    total = len(avail)
    row = '{:<26}'.format(group + ' ({})'.format(total))
    cells = []
    for method_name, sel in methods.items():
        n = len([f for f in avail if f in sel])
        cells.append('{}/{}'.format(n, total))
        row += '{:>13}'.format('{}/{}'.format(n, total))
    print(row)
    group_rows.append((group, cells))

print('-'*95)
row = '{:<26}'.format('TOTAL')
for method_name, sel in methods.items():
    row += '{:>13}'.format(str(len(sel)) + '/' + str(len(all_feats)))
print(row)

print()
print('Silhouette:  {:<26}  k-Means Sil raw: 0.7360'.format('Contrastivo: 0.9711'))
print('Reducao:     {:<26}  k-Means Sil raw: 35.2%'.format('Contrastivo: 64.8%'))
print()
print('* Metodo proposto: encoder contrastivo + busca bidirecional no espaco latente')
