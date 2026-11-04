# N3: ensaio histórico de neutralização simétrica de apresentação

Esta documentação preserva a N3 simétrica anterior. Para a nova execução que
restringe Markdown e placeholders aos sintéticos, use
`scripts/SELECTION_DIAGNOSTICS_N2_N3.md` e `HORUS_NORMALIZE_STAGE=n3s`.

## Hipótese e escopo

N3 deriva do mesmo corpus imutável que N2, reaplicando as regras N1/N2 e
acrescentando somente neutralização de sintaxe Markdown e remoção de
placeholders explícitos. Nenhuma regra usa classe, fonte, gerador, tema ou split.
Não há seleção, correção estilística, remoção de bibliografia ou alteração de
critérios de comprimento. A view N2 congelada permanece a referência.

São neutralizados títulos `#`, marcadores não numerados `-`, `*`, `+`, ênfase,
links Markdown (o texto âncora permanece), delimitadores de código inline e
cercas de blocos de código. O conteúdo dentro do código permanece. A regra
conserva listas numeradas, que podem ser seções legítimas em textos jurídicos
ou acadêmicos. Se um bloco cercado não fecha, o bloco é retido e sinalizado.
Registros com `<pre>` ou `<code>` permanecem intactos nessa etapa. Placeholders
seguem a expressão explícita já existente em `normalization.py`; placeholders
em código protegido permanecem.

Linhas de diálogo literário iniciadas por `-` podem se parecer com listas.
Por isso os exemplos cegos de textos literários humanos alterados devem ser
inspecionados antes de aceitar a N3 como view padrão. Esta é uma hipótese de
erro conhecida do ensaio, não motivo para apagar previamente esses textos.

Uma guarda impede esvaziar texto e veta perdas substanciais de palavras
lexicais (texto de pelo menos 50 palavras, perda de pelo menos 20 e retenção
abaixo de 90%). Sinais `N3_CHANGED`, `N3_MARKDOWN_CHANGED` e
`N3_PLACEHOLDER_CHANGED` contam documentos alterados; `operation:*` conta
ocorrências. Revisões não excluem documentos automaticamente.

## Execução no cluster

Os comandos pressupõem que as alterações locais já chegaram ao repositório
do cluster por Git. Executar a partir da raiz de `Benchmark-Horus`:

```bash
cd /home/victor.roquetto/Benchmark-Horus
git pull
PY="$HOME/miniconda3/envs/horus_theme/bin/python"
"$PY" -m pip install -e .
"$PY" -m pytest -q tests/test_normalization.py tests/test_middle_segments.py
mkdir -p results/logs
N3_JOB=$(sbatch --parsable --export=ALL,HORUS_NORMALIZE_STAGE=n3 scripts/slurm_normalize_corpus.sh)
N3_JOB="${N3_JOB%%;*}"
echo "N3_JOB=$N3_JOB"
```

Após `sacct -j "$N3_JOB"` informar `COMPLETED 0:0`:

```bash
N3_RUN="results/normalization/full-n3-${N3_JOB}"
cat "$N3_RUN/verification.json"
cat "$N3_RUN/summary.json"
PROBE_JOB=$(sbatch --parsable --export="ALL,HORUS_N3_RUN=$N3_RUN" scripts/slurm_compare_normalization_views.sh)
PROBE_JOB="${PROBE_JOB%%;*}"
echo "PROBE_JOB=$PROBE_JOB"
```

Após a segunda conclusão:

```bash
PROBE_RUN="results/audits/normalization-view-probes-n3-${PROBE_JOB}"
cat "$PROBE_RUN/provenance.json"
ls -lh "$PROBE_RUN/probe_metrics.csv" "$PROBE_RUN/paired_delta_intervals.csv" \
  "$PROBE_RUN/n2_n3_full_corpus_surfaces.csv" "$PROBE_RUN/probe_trajectory.png"
```

## Interpretação

`paired_surface_summary.csv` e `n2_n3_full_corpus_surfaces.csv` mostram a
mudança em todo o corpus. `review_examples_blind.jsonl` contém exemplos
originais versus N3, com rótulos e proveniência separados no manifest.
`probe_metrics.csv` e `paired_delta_intervals.csv` comparam raw, N1, N2 e N3
nos mesmos IDs e modelos; treino de vocabulário/scaler ocorre apenas no split
de treino e thresholds apenas na validação. Para parágrafos e sentenças, os
segmentos centrais precisam alinhar em todas as views, e a cobertura deve ser
examinada antes de interpretar métricas. Os splits herdados ainda não são
certificados contra leakage documental, e este experimento não é OOD.

Critério de avaliação: verificar perdas de conteúdo em exemplos cegos,
redução do gap de marcadores e mudanças pareadas nas sondas superficiais e
lexicais. Uma queda de acurácia não aprova automaticamente a N3; a decisão
depende também da preservação de conteúdo legítimo e da cobertura.
