# Rodada de contexto longo: dois LLMs, mesmos 600 humanos

Esta rodada reutiliza **exatamente** o `sample_manifest.parquet` da análise
anterior. Somente Gemma 4 31B IT e GPT-OSS 120B são executados novamente.
Nenhum modelo BERT é treinado ou inferido. O corpus original não é alterado.

## Decisões metodológicas

- `llm_prompts_v2.yaml` contém uma instrução comum e seis instruções de área.
  Corrige a contradição sobre gênero na área literária e deixa claro que a
  área informacional também contém verbetes. Exemplos são ilustrativos e
  externos à amostra. O mesmo `requests.jsonl`, com o mesmo prompt para cada
  ID, é entregue aos dois modelos. Seus chat templates internos são próprios
  de cada checkpoint.
- Os tokenizadores dos dois checkpoints são fixados por SHA de revisão no
  `theme_llm_600_long.yaml`. O texto integral é preservado quando o prompt
  cabe **em ambos**. Caso contrário, usa-se o maior prefixo comum que cabe
  no orçamento. Não há limite por caracteres. O preparo registra hashes,
  contagens de tokens por modelo e número de textos truncados.
- A janela configurada é de 65.536 tokens, incluindo prompt e resposta.
  Reservam-se 1.024 para resposta e 512 para chat template e diferença de
  tokenização, restando até 64.000 tokens para o prompt contado pelos dois
  tokenizadores. Essa janela é menor que o limite nativo dos dois checkpoints.
  A execução em uma H200 precisa passar por um smoke com os dois textos
  mais longos de cada área antes do lote completo.
- O prompt v2 difere do v1. Mudanças de rótulo entre as rodadas **não podem
  ser atribuídas apenas ao contexto**. O relatório registra os dois hashes
  e marca `isolated_context_change: false`.
- `incerto` e `fora_taxonomia` permanecem saídas válidas. A decisão de
  forçar categoria, aceitar múltiplos temas ou criar novos temas fica para
  um protocolo posterior; não se altera a taxonomia nesta rodada.

O arquivo `comparison.json` informa acordo, Cohen κ condicionado à área,
intervalo bootstrap, cobertura de consenso nominal, `fora_taxonomia`,
`incerto`, mudanças de rótulo, truncamento e recortes por área e fonte.
Concordância não é acurácia: não há rótulo humano de referência.

## Execução no Abaporu

Todos os comandos abaixo são executados pelo pesquisador no login do Abaporu,
a partir do checkout do Benchmark-Horus. O preparo usa CPU; os scripts Slurm
solicitam uma H200 por modelo.

### 1. Atualizar e configurar

```bash
cd /mnt/recod/homes/victor.roquetto/Benchmark-Horus
git pull --ff-only
export HORUS_ABAPORU_HOME=/mnt/recod/homes/victor.roquetto
export HORUS_HF_HUB="$HORUS_ABAPORU_HOME/.cache/huggingface/hub"
export HORUS_BERT_PYTHON="$HORUS_ABAPORU_HOME/venvs/horus_bert_theme/bin/python"
export HORUS_LLM_CONFIG=experiments/01_quality_audit/theme_agreement_600/configs/theme_llm_600_long.yaml
export HORUS_BASE_CONFIG=experiments/01_quality_audit/theme_agreement_600/configs/theme_llm_600.yaml
export HORUS_SAMPLE_PATH=results/theme_llm_600/20261016T020929307099Z_human-theme-llm-agreement-v1-sample-600_83ccf5b472/sample_manifest.parquet
export HORUS_GEMMA_REVISION=842da3794eaa0b77d5f08bae87a17459d91ff475
export HORUS_GPTOSS_REVISION=b5c939de8f754692c1647ca79fbf85e8c1e70f8a
test -s "$HORUS_SAMPLE_PATH"
test -s "$HORUS_HF_HUB/models--google--gemma-4-31B-it/snapshots/$HORUS_GEMMA_REVISION/tokenizer.json"
test -s "$HORUS_HF_HUB/models--openai--gpt-oss-120b/snapshots/$HORUS_GPTOSS_REVISION/tokenizer.json"
"$HORUS_BERT_PYTHON" -m pytest -q tests/test_theme_llm_600.py
```

### 2. Preparar os prompts compartilhados

```bash
export HORUS_REQUESTS_PATH="$("$HORUS_BERT_PYTHON" -m horus_benchmark.audit.theme_llm_cli \
  --config "$HORUS_LLM_CONFIG" prepare --sample "$HORUS_SAMPLE_PATH")"
printf '%s\n' "$HORUS_REQUESTS_PATH" > results/logs/theme_llm_600_long.requests.path
printf 'Prompts: %s\n' "$HORUS_REQUESTS_PATH"
"$HORUS_BERT_PYTHON" - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["HORUS_REQUESTS_PATH"])
m = json.loads((p.parent / "requests_metadata.json").read_text(encoding="utf-8"))
print("IDs:", m["rows"], "SHA-256:", m["requests_sha256"])
print("Truncados:", m["truncated_rows"], m["truncated_by_area"])
print("Máximo de tokens:", m["max_input_tokens_by_model"])
assert m["rows"] == 600
assert m["truncated_rows"] == 0, "Pare e envie este resumo antes de submeter os jobs"
PY
```

O requisito `truncated_rows == 0` garante que **todos os 600 textos
integrais** cabem na janela comum. Se falhar, a janela de 65.536 tokens
não bastou para esta amostra; não submeta os jobs ainda.

### 3. Smoke com os textos mais longos

```bash
export HORUS_SMOKE_LIMIT=12
export HORUS_SMOKE_STRATEGY=longest
export HORUS_GEMMA_OUTPUT=results/theme_llm_600_long/smoke/gemma4_31b
export HORUS_GPTOSS_OUTPUT=results/theme_llm_600_long/smoke/gpt_oss_120b
gemma_smoke="$(sbatch --parsable --export=ALL --time=06:00:00 scripts/slurm_theme_llm_gemma_abaporu.sh)"
gpt_smoke="$(sbatch --parsable --export=ALL --time=06:00:00 scripts/slurm_theme_llm_gptoss_abaporu.sh)"
printf 'Gemma smoke=%s GPT-OSS smoke=%s\n' "$gemma_smoke" "$gpt_smoke"
```

Depois de ambos terminarem, confira `sacct`, os arquivos `.err` em
`results/logs` e as 12 respostas válidas por modelo. O smoke seleciona os
dois maiores prompts de cada área, incluindo o caso de maior contexto.

```bash
sacct -j "$gemma_smoke,$gpt_smoke" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
"$HORUS_BERT_PYTHON" - <<'PY'
import json, os
from pathlib import Path
req = json.loads((Path(os.environ["HORUS_REQUESTS_PATH"]).parent / "requests_metadata.json").read_text(encoding="utf-8"))
for key in ("HORUS_GEMMA_OUTPUT", "HORUS_GPTOSS_OUTPUT"):
    folder = Path(os.environ[key])
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (folder / "annotations.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert meta["requests_sha256"] == req["requests_sha256"]
    assert len(rows) == 12 and all(row["status"] == "valid" for row in rows)
    assert all(not row["input_truncated"] for row in rows)
    print(key, "12 válidos; maior prompt real:", max(row["input_tokens_actual"] for row in rows))
PY
```

Se algum `assert` falhar ou o job sair com erro, pare antes do lote completo
e envie o `.err` correspondente.

### 4. Lote completo

```bash
unset HORUS_SMOKE_LIMIT HORUS_SMOKE_STRATEGY
export HORUS_GEMMA_OUTPUT=results/theme_llm_600_long/inference/gemma4_31b
export HORUS_GPTOSS_OUTPUT=results/theme_llm_600_long/inference/gpt_oss_120b
gemma_job="$(sbatch --parsable --export=ALL scripts/slurm_theme_llm_gemma_abaporu.sh)"
gpt_job="$(sbatch --parsable --export=ALL scripts/slurm_theme_llm_gptoss_abaporu.sh)"
printf 'Gemma=%s GPT-OSS=%s\n' "$gemma_job" "$gpt_job"
```

Depois de ambos terminarem, confira a conclusão antes da comparação:

```bash
sacct -j "$gemma_job,$gpt_job" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
"$HORUS_BERT_PYTHON" - <<'PY'
import json, os
from pathlib import Path
req = json.loads((Path(os.environ["HORUS_REQUESTS_PATH"]).parent / "requests_metadata.json").read_text(encoding="utf-8"))
for key in ("HORUS_GEMMA_OUTPUT", "HORUS_GPTOSS_OUTPUT"):
    folder = Path(os.environ[key])
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (folder / "annotations.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert meta["requests_sha256"] == req["requests_sha256"]
    assert len(rows) == 600 and len({row["record_id"] for row in rows}) == 600
    assert all(row["status"] == "valid" and not row["input_truncated"] for row in rows)
    print(key, "600 válidos; maior prompt real:", max(row["input_tokens_actual"] for row in rows))
PY
```

Os workers retomam saídas válidas se houver interrupção. Reenvie o mesmo
script com as mesmas variáveis para retomar; não altere prompts, checkpoint
ou diretório no meio da rodada.

### 5. Comparar as duas rodadas

```bash
REPORT="$("$HORUS_BERT_PYTHON" -m horus_benchmark.audit.theme_llm_cli \
  --config "$HORUS_LLM_CONFIG" compare-context \
  --sample "$HORUS_SAMPLE_PATH" \
  --baseline-config "$HORUS_BASE_CONFIG" \
  --baseline-gemma results/theme_llm_600/inference/gemma4_31b/annotations.jsonl \
  --baseline-gpt-oss results/theme_llm_600/inference/gpt_oss_120b/annotations.jsonl \
  --gemma "$HORUS_GEMMA_OUTPUT/annotations.jsonl" \
  --gpt-oss "$HORUS_GPTOSS_OUTPUT/annotations.jsonl")" &&
printf 'Relatório: %s\n' "$REPORT"
```

O relatório só é produzido se houver 600 rótulos válidos para cada modelo
e se IDs, amostra, taxonomia, prompts e hashes corresponderem. Produz
`comparison.json`, `comparison_rows.csv` e `coverage_by_factor.csv`,
sem texto integral. Não adicione `requests.jsonl` ao Git: ele contém
trechos do corpus humano.
