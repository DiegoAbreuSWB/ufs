"""Identify features with consensus across methods (majority voting)."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

with open(os.path.join(os.path.dirname(__file__), 'contrastive_result.json')) as f:
    contra = json.load(f)
with open(os.path.join(os.path.dirname(__file__), 'baseline_original_result.json')) as f:
    raw_bd = json.load(f)
with open(os.path.join(os.path.dirname(__file__), 'baselines_results.json')) as f:
    baselines = json.load(f)

methods_static = {
    'Variance': set(baselines['baselines']['variance']['selected_features']),
    'LaplScore': set(baselines['baselines']['laplacian_score']['selected_features']),
    'SPEC': set(baselines['baselines']['spec']['selected_features']),
    'MCFS': set(baselines['baselines']['mcfs']['selected_features']),
    'NDFS': set(baselines['baselines']['ndfs']['selected_features']),
}
contra_sel = set(contra['selected_features'])
raw_sel = set(raw_bd['selected_features'])
all_feats = set(baselines['baselines']['all_features']['selected_features'])

vote_count = {f: 0 for f in all_feats}
for sel in methods_static.values():
    for f in sel:
        vote_count[f] += 1

print('\nFEATURES SELECIONADAS PELA MAIORIA DOS BASELINES ESTATICOS (>= 3/5)')
print('=' * 75)
majority = [f for f, v in sorted(vote_count.items(), key=lambda x: -x[1]) if v >= 3]
for f in majority:
    in_contra = '[CONTRA]' if f in contra_sel else '        '
    in_raw = '[RAW-BD]' if f in raw_sel else '        '
    print(f'  {vote_count[f]:>2}/5  {in_contra}  {in_raw}  {f}')

print(f'\nTotal com maioria: {len(majority)}')

# Features UNICAS do contrastivo (nao selecionadas por nenhum baseline estatico)
only_contra = [f for f in contra_sel if vote_count.get(f, 0) == 0]
print(f'\nFEATURES EXCLUSIVAS DO METODO CONTRASTIVO (0/5 nos baselines):')
print('=' * 75)
for f in only_contra:
    print(f'  {f}')

# Features em TODOS
all_5 = [f for f, v in vote_count.items() if v == 5]
print(f'\nFEATURES SELECIONADAS POR TODOS OS 5 BASELINES ESTATICOS:')
print('=' * 75)
for f in all_5:
    in_contra = '[CONTRA]' if f in contra_sel else '        '
    in_raw = '[RAW-BD]' if f in raw_sel else '        '
    print(f'  {in_contra}  {in_raw}  {f}')
print(f'Total: {len(all_5)}')
