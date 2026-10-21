# Classificação temática do corpus completo

Este protocolo atribui `theme` aos **97.659 textos humanos** de
`dataset_final_completo`, com duas passagens independentes de Gemma 4 31B IT e
GPT-OSS 120B. Os 5.000 textos humanos da fonte `Corpus Carolina - Wikipédia`
recebem a categoria administrativa `Wikipédia` pela regra de origem. Os demais
92.659 recebem os mesmos prompts por área e a mesma porção de texto nos dois
modelos. Os temas sintéticos são preservados. Nenhum arquivo original é editado.

A primeira passagem consolida um tema da taxonomia apenas se os modelos
concordarem. Divergências e respostas `incerto` seguem para uma segunda
passagem com as duas hipóteses iniciais apresentadas sem identificar o modelo.
Concordância nessa passagem pode sofrer ancoragem. Respostas concordantes
`fora_taxonomia` mantêm esse estado; demais casos sem consenso permanecem
`incerto`. Textos truncados por limite de contexto também ficam `incerto` e
são sinalizados. A concordância das LLMs **não mede acurácia temática**.
O leitor aceita diferenças não ambíguas de maiúsculas e espaços em um nome
da taxonomia. Se a resposta não contiver um tema válido, registra
`parse_failure`, usa `incerto` e envia o caso à segunda passagem. Os JSONL
brutos permanecem para auditoria; nenhum nome fora da lista é remapeado por
semelhança semântica.

Os pedidos são divididos em arquivos de 1.000 textos. Cada tarefa Slurm guarda
progresso em JSONL e pode ser retomada com o mesmo índice. Os índices e
metadados registram hashes do corpus, taxonomia, prompts, solicitações e
revisões dos modelos. Execute os comandos **no login do Abaporu**, na ordem
abaixo. Não inicie a segunda passagem até que as duas primeiras terminem.

## 1. Preparar o ambiente

```bash
cd /mnt/recod/homes/victor.roquetto/Benchmark-Horus
git pull --ff-only
mkdir -p results/logs
export HORUS_ABAPORU_HOME=/mnt/recod/homes/victor.roquetto
export HORUS_BERT_PYTHON="$HORUS_ABAPORU_HOME/venvs/horus_bert_theme/bin/python"
export HORUS_HF_HUB="$HORUS_ABAPORU_HOME/.cache/huggingface/hub"
export HORUS_GEMMA_REVISION=842da3794eaa0b77d5f08bae87a17459d91ff475
export HORUS_GPTOSS_REVISION=b5c939de8f754692c1647ca79fbf85e8c1e70f8a
export HORUS_FULL_CONFIG=experiments/01_quality_audit/theme_agreement_600/configs/theme_llm_full.yaml
"$HORUS_BERT_PYTHON" -m pytest -q tests/test_theme_llm_600.py tests/test_theme_llm_second_pass.py tests/test_theme_llm_full.py
"$HORUS_BERT_PYTHON" -m horus_benchmark.cli --config configs/datasets/horus_full.yaml validate
```

## 2. Preparar a primeira passagem

Este preparo lê o corpus e tokeniza os 92.659 textos. Execute em um nó de
cômputo, não no login. A partição `h200` foi a partição disponível confirmada
no projeto; o preparo usa CPU, embora a alocação abaixo reserve uma GPU.

```bash
export HORUS_FIRST_INDEX="$(srun --partition=h200 --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --mem=128G --time=12:00:00 \
  "$HORUS_BERT_PYTHON" -m horus_benchmark.audit.theme_llm_full_cli \
  --config "$HORUS_FULL_CONFIG" prepare-first)"
printf '%s\n' "$HORUS_FIRST_INDEX" | tee results/logs/theme_full.first_index.path
test -s "$HORUS_FIRST_INDEX"
"$HORUS_BERT_PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); print("Pedidos:",d["rows"],"shards:",len(d["shards"]),"truncados:",d["truncated_rows"]); assert d["rows"]==92659' "$HORUS_FIRST_INDEX"
```

## 3. Executar Gemma e GPT-OSS na primeira passagem

Faça um smoke de 12 prompts no shard com a maior entrada antes do array.
Ele grava no destino definitivo e será retomado pelo array. Aguarde os dois
jobs e confira que terminaram com `COMPLETED` antes de remover o limite.

```bash
export HORUS_FULL_INDEX="$HORUS_FIRST_INDEX"
export HORUS_LONGEST_SHARD="$("$HORUS_BERT_PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(max(d["shards"],key=lambda s:max(s["max_input_tokens_by_model"].values()))["shard_id"])' "$HORUS_FULL_INDEX")"
export HORUS_SMOKE_LIMIT=12 HORUS_SMOKE_STRATEGY=longest
gemma_smoke="$(sbatch --parsable --array="$HORUS_LONGEST_SHARD" --export=ALL scripts/slurm_theme_llm_gemma_abaporu.sh)"
gpt_smoke="$(sbatch --parsable --array="$HORUS_LONGEST_SHARD" --export=ALL scripts/slurm_theme_llm_gptoss_abaporu.sh)"
printf 'Smoke Gemma: %s; GPT-OSS: %s\n' "$gemma_smoke" "$gpt_smoke"
```

Após ambos terminarem:

```bash
sacct -j "$gemma_smoke,$gpt_smoke" --format=JobID,State,ExitCode,Elapsed,MaxRSS
unset HORUS_SMOKE_LIMIT HORUS_SMOKE_STRATEGY
last_shard="$("$HORUS_BERT_PYTHON" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["shards"])-1)' "$HORUS_FULL_INDEX")"
gemma_first="$(sbatch --parsable --array="0-${last_shard}%1" --export=ALL scripts/slurm_theme_llm_gemma_abaporu.sh)"
gpt_first="$(sbatch --parsable --array="0-${last_shard}%1" --export=ALL scripts/slurm_theme_llm_gptoss_abaporu.sh)"
printf 'Primeira passagem Gemma: %s; GPT-OSS: %s\n' "$gemma_first" "$gpt_first"
```

`%1` limita cada modelo a um shard simultâneo. O processamento de 92.659
textos por modelo pode levar muitas horas ou dias; monitore com `squeue -u
"$USER"` e `sacct -j "$gemma_first,$gpt_first" --format=JobID,State,ExitCode,Elapsed`.
Uma tarefa interrompida pode ser reenviada com o mesmo índice e comando.
Se um shard antigo falhou ao ler JSONL com separadores Unicode dentro do
texto, atualize o código com `git pull` e reenvie somente os shards sem saída.
Os pedidos preparados e seus hashes continuam válidos; não é preciso gerar
um novo índice. As saídas novas usam `%A_%a` nos nomes dos logs do Slurm para
identificar diretamente o array e o shard.
Antes de preparar a segunda passagem, confira os arquivos de todos os shards.
`COMPLETED` no Slurm indica apenas que o processo terminou; não garante que
cada shard tenha produzido as respostas esperadas. O próprio comando
`prepare-second` rejeita shards ausentes ou incompletos.

## 4. Preparar e executar a segunda passagem

Em uma sessão nova do shell, repita as exportações da etapa 1 e recupere o
primeiro índice com `export HORUS_FIRST_INDEX="$(cat
results/logs/theme_full.first_index.path)"`. Apenas depois de confirmar os
dois arrays da etapa 3, execute:

```bash
export HORUS_SECOND_INDEX="$(srun --partition=h200 --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --mem=128G --time=12:00:00 \
  "$HORUS_BERT_PYTHON" -m horus_benchmark.audit.theme_llm_full_cli \
  --config "$HORUS_FULL_CONFIG" prepare-second --first-index "$HORUS_FIRST_INDEX")"
printf '%s\n' "$HORUS_SECOND_INDEX" | tee results/logs/theme_full.second_index.path
test -s "$HORUS_SECOND_INDEX"
"$HORUS_BERT_PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); print("Segunda passagem:",d["rows"],"shards:",len(d["shards"]),"truncados:",d["truncated_rows"])' "$HORUS_SECOND_INDEX"
```

Se `shards` for zero, pule os arrays e vá à etapa 5. Caso contrário:

```bash
export HORUS_FULL_INDEX="$HORUS_SECOND_INDEX"
export HORUS_LONGEST_SHARD="$("$HORUS_BERT_PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(max(d["shards"],key=lambda s:max(s["max_input_tokens_by_model"].values()))["shard_id"])' "$HORUS_FULL_INDEX")"
export HORUS_SMOKE_LIMIT=12 HORUS_SMOKE_STRATEGY=longest
gemma_smoke2="$(sbatch --parsable --array="$HORUS_LONGEST_SHARD" --export=ALL scripts/slurm_theme_llm_gemma_abaporu.sh)"
gpt_smoke2="$(sbatch --parsable --array="$HORUS_LONGEST_SHARD" --export=ALL scripts/slurm_theme_llm_gptoss_abaporu.sh)"
printf 'Segundo smoke Gemma: %s; GPT-OSS: %s\n' "$gemma_smoke2" "$gpt_smoke2"
```

Após conferir `COMPLETED` nos dois smokes:

```bash
sacct -j "$gemma_smoke2,$gpt_smoke2" --format=JobID,State,ExitCode,Elapsed,MaxRSS
unset HORUS_SMOKE_LIMIT HORUS_SMOKE_STRATEGY
last_shard="$("$HORUS_BERT_PYTHON" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["shards"])-1)' "$HORUS_FULL_INDEX")"
gemma_second="$(sbatch --parsable --array="0-${last_shard}%1" --export=ALL scripts/slurm_theme_llm_gemma_abaporu.sh)"
gpt_second="$(sbatch --parsable --array="0-${last_shard}%1" --export=ALL scripts/slurm_theme_llm_gptoss_abaporu.sh)"
printf 'Segunda passagem Gemma: %s; GPT-OSS: %s\n' "$gemma_second" "$gpt_second"
```

## 5. Materializar o dataset derivado

Depois de ambos os arrays da segunda passagem terminarem, repita as
exportações da etapa 1 e recupere os índices salvos caso a sessão tenha sido
encerrada. Execute em um nó de cômputo:

```bash
export HORUS_FIRST_INDEX="$(cat results/logs/theme_full.first_index.path)"
export HORUS_SECOND_INDEX="$(cat results/logs/theme_full.second_index.path)"
export HORUS_SUMMARY="$(srun --partition=h200 --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --mem=128G --time=06:00:00 \
  "$HORUS_BERT_PYTHON" -m horus_benchmark.audit.theme_llm_full_cli \
  --config "$HORUS_FULL_CONFIG" finalize \
  --first-index "$HORUS_FIRST_INDEX" --second-index "$HORUS_SECOND_INDEX")"
printf '%s\n' "$HORUS_SUMMARY" | tee results/logs/theme_full.summary.path
cat "$HORUS_SUMMARY"
```

O diretório do resumo terá `human_theme_assignments.parquet`,
`all_with_themes.parquet`, `train_with_themes.parquet`,
`validation_with_themes.parquet`, `test_with_themes.parquet` e
`theme_distribution.csv`. `theme_original`, saídas dos dois modelos e método
de atribuição permitem auditar cada texto. O metadado derivado serve à
organização e à análise estatística; o resultado dos detectores não deve usar
`theme` como feature nem interpretar o consenso como verdade de referência.
