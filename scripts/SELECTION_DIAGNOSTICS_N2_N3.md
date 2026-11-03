# Diagnóstico de seleção a partir de N2 e N3

Este experimento é somente diagnóstico. Ele lê os Parquets verificados da N2
e N3, mede suporte comum e sondas superficiais, e simula três políticas de
seleção. Não altera nem publica um benchmark selecionado.

As views são `n2`, `n3` (limpeza simétrica já executada) e
`n3_synthetic_only` (simulação virtual: humano N2 + sintético N3). Esta última
permite medir a hipótese de neutralizar a apresentação apenas no sintético,
sem modificar novamente o corpus nem confundir a N3 existente com outra view.

Os cenários são:

1. `common_support`: retém todas as faixas amplas de comprimento que possuem
   ambas as classes dentro da mesma área. Não força quantidades iguais.
2. `balanced_coarse`: sorteio determinístico de quantidades iguais por classe,
   área e faixa ampla.
3. `balanced_fine`: mesmo princípio em faixas de largura 0,25 em log2(1+palavras).

`full` é a referência sem seleção. Bordas e largura são parâmetros
exploratórios em `configs/experiments/selection_diagnostics_n2_n3.yaml`;
não são critérios aprovados. A seleção usa apenas área e comprimento.
Fonte, gerador, pontuação, apresentação, tema, gênero, split antigo e status
de revisão são medidos depois, não usados para escolher linhas.

As sondas usam Logistic Regression para comprimento, apresentação, pontuação
e comprimento+apresentação. Seus splits exploratórios são por
`split_group_id` quando presente, com ajuste apenas no treino e limiar na
validação. Near-duplicates e lineage documental ainda podem atravessar esses
splits; logo, suas métricas não são estimativas finais de desempenho ou OOD.
As linhas `n2_fixed_*` aplicam a mesma seleção de IDs N2 em N3, isolando o
efeito da normalização; as linhas sem esse prefixo mostram a seleção própria
de cada view.

## Cluster

Após enviar os novos arquivos ao repositório remoto:

```bash
cd /home/victor.roquetto/Benchmark-Horus
git pull
PY="$HOME/miniconda3/envs/horus_theme/bin/python"
"$PY" -m pip install -e '.[audit,dev]'
"$PY" -m pytest -q tests/test_selection_diagnostics.py
test -s results/normalization/full-n2-116035/normalized_corpus.parquet
test -s results/normalization/full-n3-125850/normalized_corpus.parquet
mkdir -p results/logs
JOB_ID=$(sbatch --parsable scripts/slurm_selection_diagnostics.sh)
JOB_ID="${JOB_ID%%;*}"
echo "JOB_ID=$JOB_ID"
```

Depois que o job finalizar:

```bash
sacct -j "$JOB_ID" --format=JobID,JobName%24,State,ExitCode,Elapsed
RUN="results/selection/diagnostics-${JOB_ID}"
tail -n 30 "results/logs/horus-select-diag-${JOB_ID}.out"
tail -n 50 "results/logs/horus-select-diag-${JOB_ID}.err"
cat "$RUN/summary.json"
ls -lh "$RUN"
```

Para conferir arquivos necessários antes de compartilhar resultados:

```bash
for FILE in common_support.csv provenance_by_length_bin.csv \
  balance_smd_ks.csv composition_retention.csv provenance_retention.csv \
  surface_probes.csv candidate_overlap.csv n3_changes_by_class_area.csv \
  scenario_tradeoff.png candidate_membership.parquet; do
  test -s "$RUN/$FILE" && echo "OK: $FILE" || echo "FALTOU: $FILE"
done
```

`candidate_membership.parquet` contém somente IDs das simulações e está
ignorado pelo Git, como os outros Parquets do projeto. As tabelas, figuras,
`summary.json` e `README.md` podem ser versionados para a análise posterior.
