# Status final da revisão e limitações remanescentes (uso interno — não vai para o artigo)

## Entregáveis

| # | Entregável | Arquivo |
|---|---|---|
| 1 | Checklist dos pedidos do Reviewer 1 | `reports/01_checklist_reviewer1.md` |
| 2 | Relatório dos experimentos | `reports/03_relatorio_experimentos.md` |
| 3 | Tabelas e figuras | `manuscript/tables/*.tex`, `results/revision/figures/` |
| 4 | Manuscrito limpo (PDF + LaTeX) | `deliverables/ContrastiveXAI_FS_R2_clean.pdf`, `deliverables/ContrastiveXAI_FS_R2_LaTeX_source.zip` |
| 5 | Versão destacada (amarelo) | `deliverables/ContrastiveXAI_FS_R2_highlighted.pdf` |
| 6 | Resposta aos revisores | `deliverables/Response_to_Reviewers_ContrastiveXAI_FS_R2.docx` |
| 7 | Limitações remanescentes | este arquivo |

## Atendimento ao Reviewer 1

Os 15 pontos têm experimento, tabela ou alteração textual correspondente; nenhum foi respondido apenas com texto quando havia teste viável.

## O que um revisor ainda pode questionar

1. **Os números mudaram em relação à versão anterior** (63 → 59 features em média; 0,906 → 0,908). A carta explica que todos os experimentos foram reexecutados com a implementação fold-interna publicada. Se o revisor comparar com a versão anterior, essa é a explicação.
2. **Mascaramento por grupo vs. independente:** desempenho comparável (independente numericamente melhor em 4 de 5 folds). O texto apresenta isso de forma neutra e defende o desenho pela coerência semântica, mas o benefício empírico da contribuição C1 não foi demonstrado.
3. **k\* no limite do intervalo:** o Silhouette continua subindo até k = 30; o texto justifica o teto 14 como limite de granularidade interpretável. Um revisor pode argumentar que não é um ótimo natural.
4. **Orçamento baixo:** em k = 10 o método fica bem abaixo do MCFS (0,536 vs 0,881), pior que o 0,806 da versão anterior.
5. **Generalização para dispositivos novos:** 8 classes cai para cerca de 0,65–0,69 para todos os métodos; o binário se mantém acima de 0,90.
6. **WUSTL:** sem equivalência nem superioridade; subconjuntos muito variáveis (3–35 features). A reprodução sem a transformação log degenera e não foi tabulada (está no repositório).
7. **Seeds:** a seed 42 foi a melhor; nas seeds 43/44 alguns folds selecionam poucas features e caem. Agregado: proposto ligeiramente abaixo de All Features (p < 0,001), dentro da margem de equivalência.
8. **Ablação do mascaramento com 5 folds:** o Wilcoxon não atinge p < 0,05 com 5 pares (declarado na tabela).
9. **N-BaIoT saturado:** o Variance-40 atinge o mesmo F1 com 40 features (o proposto usa 74).
10. **Recursos computacionais** (tempo, memória, energia) não foram medidos.

## Avaliação de risco

A versão atende substancialmente aos pedidos do Reviewer 1, com evidência experimental nova para todos os pontos e código público. O principal risco de rejeição é o revisor interpretar os itens 2 e 3 como enfraquecimento das contribuições C1 e da justificativa de k\*, já que ambos estão visíveis nas tabelas. O risco é moderado: o próprio revisor valorizou a transparência na rodada anterior.
