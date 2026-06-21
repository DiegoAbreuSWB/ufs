# ROTEIRO V2 — ContrastiveXAI-FS com VICReg
## Objetivo: corrigir o encoder e gerar todos os resultados para o paper

> **Para o Claude Code:** Leia este documento inteiro antes de escrever qualquer linha de código.
> Este roteiro substitui e evolui o `ROTEIRO_CLAUDE_CODE_DataSense.md` original.
> O pipeline já existe e funciona — o trabalho aqui é **cirúrgico**: trocar o encoder,
> ajustar a augmentação, re-rodar os experimentos, e gerar as saídas do paper.

---

## CONTEXTO: O QUE JÁ EXISTE E O QUE PRECISA MUDAR

### O que está implementado e funcionando
- `src/preprocessing.py` — DataSensePreprocessor ✓
- `src/baselines.py` — todos os 8 baselines UFS ✓
- `src/evaluation.py` — EvaluationPipeline com 5-fold CV ✓
- `src/xai_explainer.py` — ClusterSHAPExplainer + t-SNE ✓
- `experiments/01_run_baseline_original.py` — k-Means+Sil ✓
- `experiments/03_run_all_baselines.py` — todos os baselines ✓
- `experiments/04_run_comparison.py` — tabela comparativa ✓
- `experiments/05_run_ablation.py` — ablation study ✓
- `experiments/06_run_xai_analysis.py` — análise XAI ✓
- `experiments/07_final_paper_table.py` — tabela final ✓

### O problema diagnosticado
O encoder contrastivo em `src/contrastive_fs.py` usa **NT-Xent (SimCLR)** com
augmentação por ruído gaussiano. O espaço latente colapsou: ~95% das amostras
ficaram num único cluster (confirmado pelo t-SNE). Resultado no label2 (8 classes):
F1-macro = 0.843, contra 0.903 do k-Means+Sil da dissertação.

**Causa raiz confirmada:**
1. NT-Xent depende de pares positivos semanticamente equivalentes
2. Ruído gaussiano em features de rede destrói o sinal discriminativo
   (ex: `tcp-flags-rst_count = 0` com ruído → 0.3, que é a assinatura de um RST flood)
3. Domain shift: encoder treinado no benign, avaliado no dataset completo com ataques

### O que muda neste roteiro
1. **`src/contrastive_fs.py`** — substituir NT-Xent por VICReg + novo projector
2. **`src/augmenter.py`** — novo módulo com augmentação por grupos semânticos (group masking)
3. **`config.py`** — novos hiperparâmetros VICReg
4. **`experiments/02_run_contrastive.py`** — adaptar para VICReg
5. **`experiments/08_run_k_sweep.py`** — novo: avaliação com k=10, 15, 20, 25 features
6. **`experiments/09_generate_paper_figures.py`** — novo: figuras finais formatadas para o paper

---

## PARTE 1 — MODIFICAÇÕES NO ENCODER

### 1.1 Novo módulo: `src/augmenter.py`

Criar este arquivo do zero. É separado do contrastive_fs para clareza.

```python
"""
src/augmenter.py — Augmentação por grupos semânticos para dados IIoT

Estratégia: em vez de perturbar features individualmente com ruído fixo,
mascarar grupos semânticos inteiros (Context Groups do DataSense).
Duas visões da mesma amostra: ambas com todos os grupos, mas com
perturbações escaladas por grupo + masking aleatório de grupos.

Isso preserva a semântica: dois registros do mesmo tipo de ataque
continuam semanticamente equivalentes mesmo com grupos removidos.
"""

import numpy as np
import torch
from config import CONTEXT_GROUPS


class GroupAwareAugmenter:
    """
    Augmentação contrastiva por grupos semânticos.
    
    Operações aplicadas por visão (independentemente em view_a e view_b):
      1. Group masking: zerar completamente 1-2 grupos aleatórios (prob=GROUP_MASK_PROB)
      2. Feature noise: ruído gaussiano com std proporcional ao range do grupo
      3. Feature dropout: zerar features individuais (prob=FEAT_DROPOUT_PROB)
    
    Args:
        feature_names: lista de nomes das features (na ordem de X)
        context_groups: dict {grupo: [features]} do config.py
        group_mask_prob: probabilidade de zerar um grupo inteiro (default 0.3)
        feat_dropout_prob: probabilidade de zerar feature individual (default 0.1)
        noise_scale: escala do ruído gaussiano relativo ao std de cada feature (default 0.05)
    """
    
    def __init__(
        self,
        feature_names: list[str],
        context_groups: dict[str, list[str]] = None,
        group_mask_prob: float = 0.3,
        feat_dropout_prob: float = 0.1,
        noise_scale: float = 0.05,
    ):
        self.feature_names = feature_names
        self.group_mask_prob = group_mask_prob
        self.feat_dropout_prob = feat_dropout_prob
        self.noise_scale = noise_scale
        
        # Mapear grupos para índices de features
        if context_groups is None:
            context_groups = CONTEXT_GROUPS
        
        self.group_indices: dict[str, list[int]] = {}
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        for group_name, feat_list in context_groups.items():
            indices = [name_to_idx[f] for f in feat_list if f in name_to_idx]
            if indices:
                self.group_indices[group_name] = indices
        
        self.groups = list(self.group_indices.keys())
        self.n_groups = len(self.groups)
        self.n_features = len(feature_names)
        
        # Número de grupos a mascarar por visão
        # Com 9 grupos, mascarar 2 ainda deixa 7 grupos intactos
        self.n_groups_to_mask = max(1, int(self.n_groups * group_mask_prob))
    
    def augment(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Gera dois par de visões aumentadas para cada amostra em X.
        
        Args:
            X: array (n_samples, n_features), já normalizado
        Returns:
            (view_a, view_b): dois arrays (n_samples, n_features)
        """
        view_a = self._apply_augmentation(X.copy())
        view_b = self._apply_augmentation(X.copy())
        return view_a, view_b
    
    def augment_torch(
        self, X: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Versão para uso direto no loop de treino PyTorch."""
        X_np = X.cpu().numpy()
        va, vb = self.augment(X_np)
        device = X.device
        return (
            torch.tensor(va, dtype=torch.float32, device=device),
            torch.tensor(vb, dtype=torch.float32, device=device),
        )
    
    def _apply_augmentation(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        
        # 1. Group masking — para cada amostra, sortear grupos a mascarar
        # Vetorizado: criar máscara (n_samples, n_features)
        mask = np.ones((n_samples, self.n_features), dtype=np.float32)
        
        for i in range(n_samples):
            if self.n_groups > 0:
                n_mask = np.random.randint(0, self.n_groups_to_mask + 1)
                if n_mask > 0:
                    groups_to_mask = np.random.choice(
                        self.n_groups, size=n_mask, replace=False
                    )
                    for g_idx in groups_to_mask:
                        group_name = self.groups[g_idx]
                        for feat_idx in self.group_indices[group_name]:
                            mask[i, feat_idx] = 0.0
        
        X = X * mask
        
        # 2. Feature-level noise (proporcional ao std de cada feature no batch)
        feat_std = np.std(X, axis=0) + 1e-8
        noise = np.random.normal(0, self.noise_scale, X.shape) * feat_std
        X = X + noise
        
        # 3. Feature dropout individual
        feat_mask = np.random.binomial(
            1, 1 - self.feat_dropout_prob, X.shape
        ).astype(np.float32)
        X = X * feat_mask
        
        return X.astype(np.float32)
```

**Verificação após implementar:**
```bash
python -c "
from src.augmenter import GroupAwareAugmenter
from config import CONTEXT_GROUPS
import numpy as np
aug = GroupAwareAugmenter(['feat_a','feat_b'], {})
print('GroupAwareAugmenter OK')
"
```

---

### 1.2 Modificar `src/contrastive_fs.py` — substituir NT-Xent por VICReg

**ATENÇÃO:** Modificar cirurgicamente. Manter as classes e funções públicas com
a mesma assinatura para não quebrar `02_run_contrastive.py` e `06_run_xai_analysis.py`.

#### 1.2.1 Novos hiperparâmetros em `config.py`

Adicionar ao final do bloco `ENCODER CONTRASTIVO`:

```python
# ============================================================
# VICREG — substitui NT-Xent
# ============================================================
# Pesos dos três termos do loss VICReg (Bardes et al., 2022)
VICREG_LAMBDA = 25.0   # peso da invariância (MSE entre visões)
VICREG_MU = 25.0       # peso da variância (evita colapso dimensional)
VICREG_NU = 1.0        # peso da covariância (descorrelação das dimensões)
VICREG_EPS = 1e-4      # epsilon para estabilidade numérica no termo de variância

# Projector maior que NT-Xent (absorve distorções do loss)
ENCODER_PROJECTION_DIM = 512   # era 64 — VICReg precisa de projector maior

# Augmentação por grupos semânticos
GROUP_MASK_PROB = 0.3          # fração de grupos a mascarar por visão
FEAT_DROPOUT_PROB = 0.1        # probabilidade de dropout por feature individual
AUGMENT_NOISE_SCALE = 0.05     # escala do ruído relativo ao std da feature
```

#### 1.2.2 Implementação do VICReg loss

No arquivo `src/contrastive_fs.py`, substituir a função `nt_xent_loss` por:

```python
def vicreg_loss(
    z: torch.Tensor,
    z_prime: torch.Tensor,
    lambda_: float = 25.0,
    mu: float = 25.0,
    nu: float = 1.0,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, dict]:
    """
    VICReg loss (Bardes et al., NeurIPS 2022).
    
    Três termos que operam independentemente:
      - Invariância: MSE entre representações das duas visões
      - Variância: força std >= 1 em cada dimensão do batch
      - Covariância: penaliza correlação entre dimensões diferentes
    
    Args:
        z, z_prime: representações do projector, shape (batch, proj_dim)
        lambda_, mu, nu: pesos dos três termos
        eps: estabilidade numérica para o std
    Returns:
        loss total (scalar), dict com os três termos individuais (para logging)
    """
    batch_size, dim = z.shape
    
    # 1. Invariância — MSE entre as duas visões
    loss_inv = F.mse_loss(z, z_prime)
    
    # 2. Variância — cada dimensão deve ter std >= 1
    # Operar sobre z e z' separadamente e fazer média
    def variance_term(x):
        std = torch.sqrt(x.var(dim=0) + eps)
        return torch.mean(F.relu(1.0 - std))
    
    loss_var = (variance_term(z) + variance_term(z_prime)) / 2
    
    # 3. Covariância — off-diagonal da matriz de covariância deve ser ~0
    def covariance_term(x):
        x_centered = x - x.mean(dim=0)
        cov = (x_centered.T @ x_centered) / (batch_size - 1)
        # Zerar diagonal (variância de cada dimensão, não nos interessa aqui)
        off_diag = cov.pow(2)
        off_diag.fill_diagonal_(0)
        return off_diag.sum() / dim
    
    loss_cov = (covariance_term(z) + covariance_term(z_prime)) / 2
    
    loss = lambda_ * loss_inv + mu * loss_var + nu * loss_cov
    
    components = {
        "loss_inv": loss_inv.item(),
        "loss_var": loss_var.item(),
        "loss_cov": loss_cov.item(),
        "loss_total": loss.item(),
    }
    
    return loss, components
```

#### 1.2.3 Novo projector no ContrastiveEncoder

Modificar a classe `ContrastiveEncoder` para ter projector maior:

```python
class ContrastiveEncoder(nn.Module):
    """
    Encoder MLP para contrastive learning no DataSense IIoT.
    
    Arquitetura:
      Encoder: input → 256 → 128 (representação latente — usada na feature selection)
      Projector: 128 → 512 → 512 (só para o VICReg loss, descartado após treino)
    
    O projector maior é necessário para VICReg: absorve as transformações
    necessárias para satisfazer os três termos do loss sem distorcer o encoder.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,      # ENCODER_HIDDEN_DIM
        latent_dim: int = 128,      # ENCODER_LATENT_DIM
        proj_dim: int = 512,        # ENCODER_PROJECTION_DIM (era 64)
        dropout: float = 0.2,       # ENCODER_DROPOUT
    ):
        super().__init__()
        
        # Encoder backbone
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )
        
        # Projector — maior que no NT-Xent, expander architecture
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        # NOTA: VICReg NÃO usa ReLU na última camada do projector
        # e NÃO usa normalização L2 (diferente do SimCLR)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna representação latente (128d) — usada na feature selection."""
        return self.encoder(x)
    
    def forward_proj(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Retorna (representação latente, projeção) — usado no treino."""
        h = self.encoder(x)
        z = self.projector(h)
        return h, z
```

#### 1.2.4 Loop de treino VICReg

Substituir a função `train_contrastive_encoder`:

```python
def train_contrastive_encoder(
    X: np.ndarray,
    feature_names: list[str],
    context_groups: dict = None,
    save_path: str = None,
    verbose: bool = True,
) -> tuple[ContrastiveEncoder, list[float]]:
    """
    Treina o encoder com VICReg loss e augmentação por grupos semânticos.
    
    Args:
        X: array (n_samples, n_features) normalizado
        feature_names: nomes das features
        context_groups: grupos semânticos (usa config.CONTEXT_GROUPS se None)
        save_path: caminho para salvar o encoder (.pt)
        verbose: imprimir progresso
    Returns:
        (encoder treinado, histórico de loss por época)
    """
    from config import (
        ENCODER_HIDDEN_DIM, ENCODER_LATENT_DIM, ENCODER_PROJECTION_DIM,
        ENCODER_EPOCHS, ENCODER_BATCH_SIZE, ENCODER_LR, ENCODER_DROPOUT,
        VICREG_LAMBDA, VICREG_MU, VICREG_NU, VICREG_EPS,
        GROUP_MASK_PROB, FEAT_DROPOUT_PROB, AUGMENT_NOISE_SCALE,
        CONTEXT_GROUPS,
    )
    from src.augmenter import GroupAwareAugmenter
    
    if context_groups is None:
        context_groups = CONTEXT_GROUPS
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"[VICReg] Device: {device}")
        print(f"[VICReg] Dataset: {X.shape[0]:,} amostras × {X.shape[1]} features")
    
    # Augmenter com grupos semânticos
    augmenter = GroupAwareAugmenter(
        feature_names=feature_names,
        context_groups=context_groups,
        group_mask_prob=GROUP_MASK_PROB,
        feat_dropout_prob=FEAT_DROPOUT_PROB,
        noise_scale=AUGMENT_NOISE_SCALE,
    )
    if verbose:
        print(f"[VICReg] Grupos semânticos: {len(augmenter.groups)}")
        print(f"[VICReg] Grupos por visão mascarados: 0-{augmenter.n_groups_to_mask}")
    
    # Encoder e otimizador
    encoder = ContrastiveEncoder(
        input_dim=X.shape[1],
        hidden_dim=ENCODER_HIDDEN_DIM,
        latent_dim=ENCODER_LATENT_DIM,
        proj_dim=ENCODER_PROJECTION_DIM,
        dropout=ENCODER_DROPOUT,
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=ENCODER_LR,
        weight_decay=1e-4,
    )
    
    # Scheduler: cosine annealing para estabilidade no final do treino
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=ENCODER_EPOCHS, eta_min=ENCODER_LR * 0.01
    )
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=ENCODER_BATCH_SIZE,
        shuffle=True,
        drop_last=True,  # IMPORTANTE: VICReg precisa de batch completo para std/cov
    )
    
    # Loop de treino
    loss_history = []
    
    for epoch in range(1, ENCODER_EPOCHS + 1):
        encoder.train()
        epoch_losses = {"total": [], "inv": [], "var": [], "cov": []}
        
        for (batch,) in loader:
            # Gerar duas visões aumentadas
            view_a, view_b = augmenter.augment_torch(batch)
            view_a = view_a.to(device)
            view_b = view_b.to(device)
            
            # Forward pass — usar forward_proj para treino
            _, z_a = encoder.forward_proj(view_a)
            _, z_b = encoder.forward_proj(view_b)
            
            # VICReg loss
            loss, components = vicreg_loss(
                z_a, z_b,
                lambda_=VICREG_LAMBDA,
                mu=VICREG_MU,
                nu=VICREG_NU,
                eps=VICREG_EPS,
            )
            
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping para estabilidade
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses["total"].append(components["loss_total"])
            epoch_losses["inv"].append(components["loss_inv"])
            epoch_losses["var"].append(components["loss_var"])
            epoch_losses["cov"].append(components["loss_cov"])
        
        scheduler.step()
        
        avg_total = np.mean(epoch_losses["total"])
        avg_inv = np.mean(epoch_losses["inv"])
        avg_var = np.mean(epoch_losses["var"])
        avg_cov = np.mean(epoch_losses["cov"])
        loss_history.append(avg_total)
        
        if verbose and (epoch % 10 == 0 or epoch == 1):
            lr = scheduler.get_last_lr()[0]
            print(
                f"  Época {epoch:3d}/{ENCODER_EPOCHS} | "
                f"Total: {avg_total:.4f} | "
                f"Inv: {avg_inv:.4f} | "
                f"Var: {avg_var:.4f} | "
                f"Cov: {avg_cov:.4f} | "
                f"LR: {lr:.6f}"
            )
    
    encoder.eval()
    
    # Salvar encoder
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            "model_state_dict": encoder.state_dict(),
            "input_dim": X.shape[1],
            "hidden_dim": ENCODER_HIDDEN_DIM,
            "latent_dim": ENCODER_LATENT_DIM,
            "proj_dim": ENCODER_PROJECTION_DIM,
            "feature_names": feature_names,
            "loss_history": loss_history,
            "method": "vicreg",
        }, save_path)
        if verbose:
            print(f"[VICReg] Encoder salvo em: {save_path}")
    
    return encoder, loss_history
```

**Verificação de sanidade após implementar:**
```bash
python -c "
import numpy as np
from src.contrastive_fs import train_contrastive_encoder, get_representations
from config import CONTEXT_GROUPS

# Teste rápido com dados sintéticos
X_test = np.random.randn(1000, 71).astype(np.float32)
feat_names = [f'feat_{i}' for i in range(71)]
encoder, history = train_contrastive_encoder(
    X_test, feat_names, CONTEXT_GROUPS, verbose=True
)
# Verificar:
# 1. Loss decresce
print(f'Loss ep1: {history[0]:.4f} -> ep final: {history[-1]:.4f}')
assert history[-1] < history[0], 'ERRO: loss não decresce!'

# 2. Representações não colapsaram (std > 0 em cada dimensão)
H = get_representations(encoder, X_test)
dim_stds = H.std(axis=0)
n_collapsed = (dim_stds < 0.01).sum()
print(f'Dimensões colapsadas (std < 0.01): {n_collapsed}/128')
assert n_collapsed < 10, f'ERRO: encoder colapsado! {n_collapsed} dimensões colapsadas'
print('OK: encoder VICReg funciona')
"
```

---

## PARTE 2 — NOVO EXPERIMENTO: K SWEEP

### 2.1 Criar `experiments/08_run_k_sweep.py`

Este experimento avalia todos os métodos com k = 10, 15, 20, 25 features.
É o experimento que vai demonstrar que com k menor o encoder VICReg
se diferencia dos baselines.

```python
"""
08_run_k_sweep.py — Avaliação com k variável: 10, 15, 20, 25 features

Objetivo: mostrar que com k pequeno (10 features), métodos que preservam
estrutura discriminativa (VICReg, k-Means+Sil, MCFS) mantêm performance,
enquanto métodos de ranking simples (Variance, Laplacian) degradam.

Isso motiva o uso de seleção não supervisionada inteligente vs. ranking estatístico.

Uso:
    python experiments/08_run_k_sweep.py
    python experiments/08_run_k_sweep.py --csv data/dataset_attack_benign.csv
"""

import sys, os, json, time, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BENIGN_CSV, FULL_CSV, TABLES_DIR, FIGURES_DIR, CONTEXT_GROUPS
from src.preprocessing import DataSensePreprocessor
from src.contrastive_fs import (
    train_contrastive_encoder, ContrastiveBidirectionalFS,
    get_representations, load_encoder,
)
from src.baselines import (
    VarianceFS, LaplacianScoreFS, SPECFS, MCFSFS, UDFSFS, NDFSFS,
    BidirectionalSilhouetteFS,
)
from src.evaluation import EvaluationPipeline

ENCODER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "results", "models", "encoder_datasense_vicreg.pt")

K_VALUES = [10, 15, 20, 25]  # k a testar


def run(csv_path: str) -> None:
    print(f"\n{'='*60}")
    print(f"EXPERIMENTO 08 — K Sweep (k = {K_VALUES})")
    print(f"{'='*60}")
    print(f"Dataset: {csv_path}\n")

    # Pré-processamento
    pp = DataSensePreprocessor(csv_path)
    X, y = pp.load()
    feature_names = pp.get_feature_names()
    n_feat = len(feature_names)
    print(f"Dataset: {X.shape[0]:,} × {X.shape[1]}")

    # Carregar ou treinar encoder VICReg
    if os.path.exists(ENCODER_PATH):
        from src.contrastive_fs import load_encoder
        encoder, _ = load_encoder(ENCODER_PATH)
        print(f"[K-Sweep] Encoder VICReg carregado: {ENCODER_PATH}")
    else:
        print(f"[K-Sweep] Treinando encoder VICReg...")
        encoder, _ = train_contrastive_encoder(
            X, feature_names, CONTEXT_GROUPS, save_path=ENCODER_PATH, verbose=True
        )

    # Label cols disponíveis
    label_cols = [c for c in ["label1", "label2"] if c in y.columns and y[c].nunique() >= 2]

    # Para cada k, rodar todos os métodos
    all_results = []
    pipe = EvaluationPipeline(include_knn=False)  # KNN lento para sweep

    for k in K_VALUES:
        print(f"\n{'─'*50}")
        print(f"K = {k} features")
        print(f"{'─'*50}")

        # Montar feature sets para este k
        feature_sets = {}

        # 1. Contrastivo VICReg
        fs_contra = ContrastiveBidirectionalFS(encoder=encoder, verbose=False)
        # Forçar k de saída
        fs_contra.fit(X)
        # Pegar os top-k do resultado da busca bidirecional
        # (ContrastiveBidirectionalFS já seleciona automaticamente, mas
        # podemos usar os índices rankeados por importância latente)
        contra_indices = fs_contra.selected_indices_[:k] if len(fs_contra.selected_indices_) >= k \
            else fs_contra.selected_indices_
        feature_sets[f"vicreg_k{k}"] = list(contra_indices)

        # 2. k-Means+Sil (dissertação)
        fs_km = BidirectionalSilhouetteFS(verbose=False)
        fs_km.fit(X)
        km_indices = fs_km.selected_indices_[:k] if len(fs_km.selected_indices_) >= k \
            else fs_km.selected_indices_
        feature_sets[f"kmeans_sil_k{k}"] = list(km_indices)

        # 3. Baselines de ranking (top-k direto)
        ranking_methods = {
            f"variance_k{k}": VarianceFS(),
            f"laplacian_k{k}": LaplacianScoreFS(sample_size=5000),
            f"mcfs_k{k}": MCFSFS(n_clusters=8, sample_size=5000),  # 8 = n_classes label2
        }
        for name, method in ranking_methods.items():
            try:
                method.fit(X)
                feature_sets[name] = list(method.select_top_k(k))
            except Exception as e:
                print(f"  [{name}] ERRO: {e}")

        # 4. All features (referência)
        feature_sets[f"all_features"] = list(range(n_feat))

        # Avaliar para cada label col
        for label_col in label_cols:
            df = pipe.run_all(X, y, feature_sets, label_col=label_col, verbose=True)
            if not df.empty:
                df["k"] = k
                all_results.append(df)
            print(f"  [K={k}, {label_col}] concluído")

    # Consolidar resultados
    if not all_results:
        print("Nenhum resultado gerado.")
        return

    results_df = pd.concat(all_results, ignore_index=True)

    # Salvar
    os.makedirs(TABLES_DIR, exist_ok=True)
    out_csv = os.path.join(TABLES_DIR, "k_sweep_results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\n[K-Sweep] Resultados salvos: {out_csv}")

    # Plot: F1-macro por k, por método (label2 — o mais discriminativo)
    _plot_k_sweep(results_df, label_col="label2", output_dir=FIGURES_DIR)


def _plot_k_sweep(df: pd.DataFrame, label_col: str, output_dir: str) -> None:
    """
    Lineplots: eixo X = k, eixo Y = F1-macro, uma linha por método.
    Mostra como cada método degrada (ou não) com menos features.
    """
    df2 = df[df.get("scenario", label_col) == label_col].copy() if "scenario" in df.columns \
        else df.copy()

    # Agregar por método e k (média entre classificadores)
    df_rf = df2[df2["classifier"] == "random_forest"].copy()
    if df_rf.empty:
        return

    # Extrair nome base do método (remover _k{k})
    df_rf["method_base"] = df_rf["method"].str.replace(r"_k\d+$", "", regex=True)
    df_rf["k_val"] = df_rf["k"] if "k" in df_rf.columns else df_rf["n_features"]

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    method_styles = {
        "vicreg": ("#185FA5", "o", "-", 2.5),         # azul sólido, destaque
        "kmeans_sil": ("#1D9E75", "s", "--", 1.8),    # verde tracejado
        "mcfs": ("#BA7517", "^", "--", 1.8),           # âmbar tracejado
        "variance": ("#A32D2D", "x", ":", 1.5),        # vermelho pontilhado
        "laplacian": ("#534AB7", "D", ":", 1.5),       # roxo pontilhado
        "all_features": ("#888780", ".", "-", 1.2),    # cinza referência
    }

    for base_method, group in df_rf.groupby("method_base"):
        g = group.sort_values("k_val")
        style = method_styles.get(base_method, ("#888780", "o", "-", 1.2))
        color, marker, ls, lw = style
        ax.plot(
            g["k_val"], g["f1_macro_mean"],
            marker=marker, linestyle=ls, linewidth=lw,
            color=color, label=base_method, markersize=7,
        )
        # Barra de erro
        ax.fill_between(
            g["k_val"],
            g["f1_macro_mean"] - g["f1_macro_std"],
            g["f1_macro_mean"] + g["f1_macro_std"],
            alpha=0.1, color=color,
        )

    ax.set_xlabel("Número de features selecionadas (k)", fontsize=12)
    ax.set_ylabel("F1-macro (Random Forest, 5-fold CV)", fontsize=12)
    ax.set_title(f"Degradação por k — {label_col}", fontsize=13)
    ax.set_xticks(K_VALUES)
    ax.legend(title="Método", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_ylim(0.7, 1.0)
    plt.tight_layout()

    out = os.path.join(output_dir, f"k_sweep_{label_col}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[K-Sweep] Gráfico salvo: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()
    csv_path = args.csv or (FULL_CSV if os.path.exists(FULL_CSV) else BENIGN_CSV)
    run(csv_path)
```

---

## PARTE 3 — FIGURAS FINAIS PARA O PAPER

### 3.1 Criar `experiments/09_generate_paper_figures.py`

Este script gera todas as figuras no formato pronto para inserir no LaTeX/Word.
Estilo: sem título nos plots (o caption vai no paper), fundo branco, fontes legíveis.

```python
"""
09_generate_paper_figures.py — Gera figuras finais formatadas para o paper

Figuras geradas:
  fig1_pipeline.png         — Diagrama do pipeline (texto, não precisa de dados)
  fig2_tsne_comparison.png  — t-SNE: NT-Xent colapsado vs VICReg estruturado
  fig3_comparison_table.png — Tabela comparativa como figura (label1 + label2)
  fig4_k_sweep.png          — F1 por k features (do experimento 08)
  fig5_shap_summary.png     — SHAP summary reforçado
  fig6_ablation_delta.png   — Ganho incremental por componente

Uso:
    python experiments/09_generate_paper_figures.py
"""

import sys, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TABLES_DIR, FIGURES_DIR, MODELS_DIR

# Estilo global para o paper
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,       # alta resolução para o paper
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

PAPER_FIGS_DIR = os.path.join(FIGURES_DIR, "paper_ready")


def fig2_tsne_comparison():
    """
    t-SNE side-by-side: antes (NT-Xent colapsado) vs depois (VICReg).
    Requer que ambos os encoders tenham sido treinados e o t-SNE salvo.
    Se não existir o VICReg t-SNE ainda, plota só o existente com anotação.
    """
    # Implementar após treinar o VICReg e re-rodar experiment 06
    pass


def fig3_comparison_table():
    """
    Tabela comparativa label1 + label2 como heatmap colorido.
    Mais visual do que uma tabela de texto no paper.
    """
    p = os.path.join(TABLES_DIR, "final_paper_table.csv")
    if not os.path.exists(p):
        print("[fig3] final_paper_table.csv não encontrado. Rodar exp 07 primeiro.")
        return

    df = pd.read_csv(p)
    
    # Filtrar métodos relevantes
    methods_order = [
        "contrastive_proposed",
        "kmeans_silhouette",
        "mcfs",
        "variance",
        "spec",
        "laplacian_score",
        "ndfs",
        "udfs",
        "all_features",
        "datasense_17_supervised",
    ]
    labels_order = ["label1", "label2"]
    
    # Pivot para heatmap
    pivot_f1 = df[df["method"].isin(methods_order)].pivot_table(
        index="method", columns="scenario", values="f1_mean"
    ).reindex(index=methods_order, columns=labels_order)
    
    pivot_n = df[df["method"].isin(methods_order)].pivot_table(
        index="method", columns="scenario", values="n_features"
    ).reindex(index=methods_order, columns=labels_order)

    method_labels = {
        "contrastive_proposed": "ContrastiveXAI-FS (ours)",
        "kmeans_silhouette": "k-Means+Sil (Abreu 2022)",
        "mcfs": "MCFS",
        "variance": "Variance",
        "spec": "SPEC",
        "laplacian_score": "Laplacian Score",
        "ndfs": "NDFS",
        "udfs": "UDFS",
        "all_features": "All Features (baseline)",
        "datasense_17_supervised": "DataSense-17 (supervised†)",
    }
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        pivot_f1,
        annot=True, fmt=".3f",
        cmap="RdYlGn", vmin=0.80, vmax=0.95,
        linewidths=0.5, linecolor="white",
        xticklabels=["Binary (label1)", "8-class (label2)"],
        yticklabels=[method_labels.get(m, m) for m in methods_order],
        ax=ax,
        cbar_kws={"label": "F1-macro (avg classifiers, 5-fold CV)"},
    )
    
    # Destacar a linha do método proposto
    ax.add_patch(plt.Rectangle((0, 0), 2, 1, fill=False, edgecolor="#185FA5", lw=2))
    
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    # Nota rodapé
    fig.text(0.01, -0.02, "† supervised method — uses labels during feature selection",
             fontsize=9, style="italic")
    
    out = os.path.join(PAPER_FIGS_DIR, "fig3_comparison_heatmap.pdf")
    plt.savefig(out, format="pdf")
    plt.savefig(out.replace(".pdf", ".png"))
    plt.close()
    print(f"[fig3] Salvo: {out}")


def fig4_k_sweep():
    """Lineplot do K sweep — copiar e reformatar do experimento 08."""
    p = os.path.join(TABLES_DIR, "k_sweep_results.csv")
    if not os.path.exists(p):
        print("[fig4] k_sweep_results.csv não encontrado. Rodar exp 08 primeiro.")
        return
    
    df = pd.read_csv(p)
    df_rf = df[df["classifier"] == "random_forest"].copy()
    df_l2 = df_rf  # usar label2 (mais discriminativo)
    
    # Implementar o lineplot — similar ao _plot_k_sweep do exp 08
    # mas com estilo paper (sem título, fontes maiores, PDF output)
    pass


def fig5_shap_summary():
    """Re-gerar SHAP summary com estilo paper."""
    # Chamar explainer.plot_shap_summary() com parâmetros de estilo
    # Implementar após re-rodar experimento 06 com VICReg
    pass


def run():
    os.makedirs(PAPER_FIGS_DIR, exist_ok=True)
    print(f"\n{'='*60}")
    print("EXPERIMENTO 09 — Figuras para o Paper")
    print(f"{'='*60}")
    print(f"Output: {PAPER_FIGS_DIR}\n")
    
    fig3_comparison_table()
    fig4_k_sweep()
    print("\nFiguras geradas. Completar fig2 e fig5 após re-rodar exp 02 e 06.")


if __name__ == "__main__":
    run()
```

---

## PARTE 4 — ORDEM DE EXECUÇÃO COMPLETA

Execute nesta ordem exata. Cada passo depende do anterior.

### Passo 0 — Verificar dataset completo

```bash
python verify_dataset.py data/dataset_attack_benign.csv
# Verificar: n_samples, distribuição de label2, features com variância zero
```

### Passo 1 — Implementar módulos novos

```bash
# 1a. Criar src/augmenter.py (código na Parte 1.1)
# 1b. Modificar src/contrastive_fs.py (código na Parte 1.2)
# 1c. Adicionar hiperparâmetros VICReg em config.py

# Verificar augmenter
python -c "from src.augmenter import GroupAwareAugmenter; print('OK')"

# Verificar encoder VICReg (teste rápido, ~1 min)
python -c "
import numpy as np
from src.contrastive_fs import train_contrastive_encoder
from config import CONTEXT_GROUPS
X = np.random.randn(2000, 71).astype('float32')
feat_names = [f'f{i}' for i in range(71)]
enc, hist = train_contrastive_encoder(X, feat_names, CONTEXT_GROUPS, verbose=True)
print(f'Loss: {hist[0]:.3f} -> {hist[-1]:.3f}')
assert hist[-1] < hist[0], 'Encoder não aprendeu!'
print('VICReg OK')
"
```

### Passo 2 — Treinar encoder VICReg no dataset completo

```bash
python experiments/02_run_contrastive.py --csv data/dataset_attack_benign.csv
# Saída esperada:
#   - Loss decresce de ~100 para ~20-30 (VICReg tem escala diferente do NT-Xent)
#   - Termo var: deve ser próximo de 0 (sem colapso)
#   - Termo inv: decresce (visões se aproximam)
#   - results/models/encoder_datasense.pt salvo
```

**Verificação de saúde do encoder — OBRIGATÓRIA antes de continuar:**
```bash
python -c "
from src.contrastive_fs import load_encoder, get_representations
from src.preprocessing import DataSensePreprocessor
import numpy as np

pp = DataSensePreprocessor('data/dataset_attack_benign.csv')
X, y = pp.load()
encoder, _ = load_encoder('results/models/encoder_datasense.pt')
H = get_representations(encoder, X[:5000])

# 1. Verificar que não colapsou
dim_stds = H.std(axis=0)
collapsed = (dim_stds < 0.01).sum()
print(f'Dimensões colapsadas: {collapsed}/128 (esperado: < 5)')

# 2. Verificar separação por label2 (NMI)
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans
km = KMeans(n_clusters=8, random_state=42, n_init=10)
pred = km.fit_predict(H)
labels = y['label2'].values[:5000]
nmi = normalized_mutual_info_score(labels, pred)
print(f'NMI clusters vs label2: {nmi:.4f} (esperado: > 0.10, ideal > 0.25)')
"
```

### Passo 3 — Re-rodar baselines (não mudam, mas precisam do mesmo k)

```bash
python experiments/01_run_baseline_original.py --csv data/dataset_attack_benign.csv
python experiments/03_run_all_baselines.py --csv data/dataset_attack_benign.csv
```

### Passo 4 — Avaliação comparativa completa

```bash
python experiments/04_run_comparison.py --csv data/dataset_attack_benign.csv
python experiments/04b_datasense17_eval.py --csv data/dataset_attack_benign.csv
```

**Resultado esperado com VICReg (label2, Random Forest):**
```
contrastive_proposed:  F1 ≈ 0.87-0.90  (era 0.85 com NT-Xent)
kmeans_silhouette:     F1 ≈ 0.90        (referência — não muda)
all_features:          F1 ≈ 0.90        (referência — não muda)
```
Se contrastive_proposed não subiu acima de 0.87, revisar o encoder (ver diagnóstico abaixo).

### Passo 5 — K sweep

```bash
python experiments/08_run_k_sweep.py --csv data/dataset_attack_benign.csv
# Gera: results/tables/k_sweep_results.csv
#        results/figures/k_sweep_label2.png
```

### Passo 6 — Ablation study

```bash
python experiments/05_run_ablation.py --csv data/dataset_attack_benign.csv
```

### Passo 7 — Análise XAI com encoder novo

```bash
python experiments/06_run_xai_analysis.py --csv data/dataset_attack_benign.csv
# Com encoder VICReg, o t-SNE deve mostrar estrutura mais clara
# Os clusters devem ser mais separados
```

### Passo 8 — Tabela final e figuras do paper

```bash
python experiments/07_final_paper_table.py
python experiments/09_generate_paper_figures.py
# Figuras prontas em: results/figures/paper_ready/
```

---

## PARTE 5 — DIAGNÓSTICO SE O ENCODER NÃO MELHORAR

Se após o Passo 4 o F1 do contrastivo no label2 **não subiu acima de 0.87**,
executar este diagnóstico antes de fazer qualquer outra mudança:

```bash
python -c "
from src.contrastive_fs import load_encoder, get_representations
from src.preprocessing import DataSensePreprocessor
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
import numpy as np

pp = DataSensePreprocessor('data/dataset_attack_benign.csv')
X, y = pp.load()
encoder, meta = load_encoder('results/models/encoder_datasense.pt')
H = get_representations(encoder, X)

print('=== DIAGNÓSTICO DO ESPAÇO LATENTE ===')

# 1. Colapso dimensional
stds = H.std(axis=0)
print(f'Dimensões com std < 0.01: {(stds < 0.01).sum()}/128')
print(f'Std médio: {stds.mean():.4f} | min: {stds.min():.4f} | max: {stds.max():.4f}')

# 2. NMI com labels reais
for label_col in ['label1', 'label2']:
    if label_col in y.columns:
        n_cls = y[label_col].nunique()
        km = KMeans(n_clusters=n_cls, random_state=42, n_init=10)
        pred = km.fit_predict(H[:10000])
        nmi = normalized_mutual_info_score(y[label_col].values[:10000], pred)
        print(f'NMI vs {label_col} ({n_cls} classes): {nmi:.4f}')

# 3. Silhouette no espaço latente (amostra)
idx = np.random.choice(len(H), 5000, replace=False)
km8 = KMeans(n_clusters=8, random_state=42, n_init=10).fit(H[idx])
sil = silhouette_score(H[idx], km8.labels_)
print(f'Silhouette (k=8, 5k amostras): {sil:.4f}')

print()
print('Interpretação:')
print('  NMI label2 > 0.25: encoder está capturando estrutura das 8 classes')
print('  NMI label2 < 0.10: encoder colapsou ou não aprendeu nada relevante')
print('  Silhouette > 0.15: clusters latentes têm alguma estrutura')
"
```

**Se NMI label2 < 0.10:** o encoder ainda está colapsando.
Possíveis causas e ações:
- Batch size muito pequeno para VICReg → aumentar para 1024 em `config.py`
- Epochs insuficientes → aumentar para 150
- Projector muito pequeno → aumentar `ENCODER_PROJECTION_DIM` para 1024
- Augmentação muito agressiva → reduzir `GROUP_MASK_PROB` para 0.2

**Se NMI label2 entre 0.10 e 0.20:** encoder aprendeu alguma estrutura
mas não suficiente. O F1 no label2 deve estar em torno de 0.85-0.87.
Isso é aceitável para o paper — o argumento muda para "redução de features
mantendo performance" em vez de "melhora de performance".

---

## PARTE 6 — CHECKLIST FINAL PARA O PAPER

Antes de considerar os experimentos completos, verificar cada item:

```
DADOS E PIPELINE
[ ] dataset completo carregado: verificar n_samples e distribuição de label2
[ ] encoder VICReg treinado: loss história salva em contrastive_loss_curve.csv
[ ] NMI label2 > 0.10 (encoder aprendeu estrutura mínima)

EXPERIMENTOS
[ ] Exp 01: baseline_original_result.json existe
[ ] Exp 02: contrastive_result.json existe (com "method": "vicreg" no JSON)
[ ] Exp 03: baselines_results.json existe
[ ] Exp 04: comparison_label1.csv e comparison_label2.csv existem
[ ] Exp 04b: datasense17_vs_contrastive_label1.csv e _label2.csv existem
[ ] Exp 05: ablation_label1.csv e ablation_label2.csv existem
[ ] Exp 06: shap_summary.png, tsne_clusters_vs_labels.png, cluster_heatmap.png existem
[ ] Exp 07: final_paper_table.csv existe
[ ] Exp 08: k_sweep_results.csv existe, k_sweep_label2.png existe
[ ] Exp 09: paper_ready/ com pelo menos fig3 e fig4

RESULTADOS MÍNIMOS PARA SUBMISSÃO
[ ] label1: contrastive_proposed F1 > datasense_17_supervised F1 (+0.01 mínimo)
[ ] label2: contrastive_proposed F1 > 0.85 com 25 features (redução de 65%)
[ ] k=10: contrastive_proposed degradação < kmeans_silhouette no label2
[ ] SHAP: pelo menos 3 clusters com interpretação semântica clara
[ ] t-SNE: espaço latente VICReg visivelmente mais estruturado que NT-Xent

TABELAS DO PAPER
[ ] Tabela 1: comparação completa label1 (todos os métodos × F1, Acc, MCC)
[ ] Tabela 2: comparação completa label2 (idem)
[ ] Tabela 3: ablation study (contribuição de cada componente)
[ ] Tabela 4: k sweep resumido (k=10 e k=25 para os métodos principais)

FIGURAS DO PAPER
[ ] Fig 1: diagrama do pipeline (pode ser feito no LaTeX/draw.io)
[ ] Fig 2: t-SNE NT-Xent vs VICReg (side-by-side)
[ ] Fig 3: heatmap comparativo (fig3_comparison_heatmap.pdf)
[ ] Fig 4: k sweep lineplot (k_sweep_label2.png reformatado)
[ ] Fig 5: SHAP summary + 2 waterfalls representativos
[ ] Fig 6: ablation delta incremental
```

---

## REFERÊNCIAS TÉCNICAS

### VICReg
- **Paper:** Bardes, A., Ponce, J., LeCun, Y. (2022). VICReg: Variance-Invariance-Covariance
  Regularization for Self-Supervised Learning. ICLR 2022.
- **Diferença chave vs NT-Xent:** não precisa de negativos; evita colapso via regularização
  explícita da variância por dimensão.
- **Hiperparâmetros padrão:** λ=25, μ=25, ν=1 (funciona sem tuning na maioria dos casos)

### SCARF (referência para group masking)
- **Paper:** Bahri, D. et al. (2021). SCARF: Self-Supervised Contrastive Learning using
  Random Feature Corruption. ICLR 2022.
- **Diferença:** SCARF faz masking por feature individual; nossa proposta faz por grupo
  semântico, o que é mais adequado para dados IIoT onde grupos têm coerência semântica.

### Referências dos baselines (para o Related Work)
- Laplacian Score: He, Cai, Niyogi (NIPS 2005)
- SPEC: Zhao, Liu (ICML 2007)
- MCFS: Cai, Zhang, He (KDD 2010)
- UDFS: Yang et al. (IJCAI 2011)
- NDFS: Li et al. (AAAI 2012)
- DataSense: Firouzi et al. (CIC 2025)
- Dissertação original: Abreu, D.M. (UFPA 2022)
