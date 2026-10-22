# Diagnóstico de metadados e prévia N2

Estas rotinas operam sobre o **corpus completo e imutável**. Não selecionam
linhas, não criam splits, não treinam classificadores e não inferem autoria.
Rótulo, área, fonte e gerador são usados apenas em tabelas posteriores às
decisões textuais. O corpus original jamais é reescrito.

## 1. Diagnóstico de origem

`normalization_diagnostics` compara o `word_count` declarado com
`len(content.split())` (convenção canônica da view derivada) e uma contagem
lexical auxiliar. A diferença pode resultar de limpeza anterior sem recálculo;
não deve ser chamada de erro de texto sem investigar a procedência. Exporta
histograma das diferenças e taxas por fonte.

Para `content_hash`, compara o SHA-256 do UTF-8 exato e testa, **somente como
hipóteses**, NFC, padronização de quebras de linha, `strip` e uma assinatura
NFKC/casefold/espaços. Nenhuma coincidência substitui o hash declarado.
Exporta todos os IDs discrepantes, distribuição por fonte e arquivo de origem,
além de exemplos com texto para inspeção.

As duas classes acadêmicas são inspecionadas no corpus inteiro quanto a títulos de
fontes/referências (inclusive Markdown e `Referências (selecionadas)`) e candidatos estritos; a amostra cega de 120 textos sintéticos é
distribuída entre geradores, com até metade reservada a ocorrências de títulos
de fontes. **A amostra é enriquecida, não estima prevalência**. A prevalência
observada vem das tabelas de censo.

## 2. Prévia N2

`preview_n2` usa a configuração separada `corpus_normalization_n2_preview.yaml`,
que habilita **somente para a prévia** propostas de remoção bibliográfica. A
configuração N2 principal mantém essa remoção desligada até a decisão do
pesquisador. `preview_n2` exige um N1 verificado da mesma fonte e revalida o texto N1 linha
a linha. Exporta spans propostos, perda em palavras, casos retidos pela guarda,
tabelas por classe/área/fonte e exemplos cegos antes/depois. Não escreve um
corpus N2. As decisões são independentes dos metadados de classe, fonte e split.

Os candidatos de recorte são:

- `Abstract` em inglês com evidência lexical e limites explícitos entre
  `Resumo` e `Introdução`, ou antes de `Resumo` seguido de `Introdução`;
- bibliografia terminal com título explícito, após 60% do texto, corpo anterior
  com pelo menos 200 palavras, sufixo com pelo menos 30 palavras e ao menos
  três linhas datadas;
- rodapé editorial terminal estrito.

Esses limites são **propostas conservadoras para validar**, não estimativas de
precisão. Se uma proposta conjunta remover pelo menos 20 palavras, deixar menos
de 90% do N1 e o N1 tiver ao menos 50 palavras, o N2 retém integralmente o N1
e assinala `N2_GUARD_RETAINED`. A guarda pode impedir recortes corretos; os
casos precisam ser examinados antes de qualquer decisão de relaxá-la.

Não há remoção geral de terminologia inglesa, citações internas, HTML/CSS de
exemplo ou seções de fontes não delimitadas. Comprimento e balanceamento ficam
para a seleção posterior. A inspeção pelo assistente é exploratória, não
equivale a anotação humana independente para publicação.

## Ordem de execução no cluster

Executar na raiz do repositório. A saída usa o ID Slurm, sem sobrescrever runs.

```bash
python -m pip install -e .
python -m pytest -q tests/test_normalization.py tests/test_normalization_diagnostics.py
mkdir -p results/logs
sbatch --export=ALL,HORUS_DIAGNOSTIC_MODE=source scripts/slurm_normalization_diagnostics.sh
sbatch --export=ALL,HORUS_DIAGNOSTIC_MODE=preview,HORUS_N1_RUN_DIR=results/normalization/full-n1-115189 scripts/slurm_normalization_diagnostics.sh
```

Depois de ambos os jobs terminarem, compartilhar `summary.json`, as tabelas
CSV e os exemplos JSONL dos dois diretórios. **Não executar ainda**
`slurm_normalize_corpus.sh` com `HORUS_NORMALIZE_STAGE=n2`: primeiro validar
os recortes e decidir se os limiares propostos permanecem.
