# Segunda passagem dirigida: 170 casos sem consenso

Esta etapa usa os mesmos 600 humanos e as duas saídas completas da rodada de
contexto longo. Ela seleciona os **170 textos não Wikipédia sem consenso**,
incluindo os casos em que ambos responderam `incerto`. Os 51 textos da fonte
`Corpus Carolina - Wikipédia` recebem a categoria administrativa `Wikipédia`
somente no relatório derivado. Seus metadados originais permanecem intactos.

Cada LLM recebe exatamente o mesmo prompt, contendo área, lista completa de
temas, texto integral e hipóteses iniciais **sem identificar o modelo que as
propôs**. Não recebe fonte, `theme` original ou resposta da segunda passagem
do outro LLM. Ele pode escolher um tema existente, `incerto` ou
`fora_taxonomia`, listar até dois temas alternativos e dar uma justificativa
curta. Nomes novos não são aceitos nesta etapa.

Um tema é consolidado apenas quando os dois modelos da segunda passagem
escolhem o mesmo tema existente. Ambos `fora_taxonomia` continuam fora da
taxonomia. Qualquer outro resultado continua `incerto`, com as quatro saídas
preservadas. Concordância não constitui referência humana nem acurácia. Como
as hipóteses iniciais aparecem no prompt, a segunda passagem pode aumentar
a concordância por **ancoragem**; essa possibilidade deve constar da análise.

O arquivo derivado `second_pass_rows.csv` não contém o texto integral. A
categoria `Wikipédia` é um **balde de origem**, não um assunto comparável aos
temas dos sintéticos. Esta etapa é um piloto; não rotula o corpus completo.

## Execução no login do Abaporu

### 1. Configuração e verificação

```bash
cd /mnt/recod/homes/victor.roquetto/Benchmark-Horus
git pull --ff-only
export HORUS_ABAPORU_HOME=/mnt/recod/homes/victor.roquetto
export HORUS_HF_HUB="$HORUS_ABAPORU_HOME/.cache/huggingface/hub"
export HORUS_BERT_PYTHON="$HORUS_ABAPORU_HOME/venvs/horus_bert_theme/bin/python"
export HORUS_SECOND_CONFIG=experiments/01_quality_audit/theme_agreement_600/configs/theme_llm_second_pass.yaml
export HORUS_SAMPLE_PATH=results/theme_llm_600/20261016T020929307099Z_human-theme-llm-agreement-v1-sample-600_83ccf5b472/sample_manifest.parquet
export HORUS_FIRST_GEMMA=results/theme_llm_600_long/inference/gemma4_31b/annotations.jsonl
export HORUS_FIRST_GPT=results/theme_llm_600_long/inference/gpt_oss_120b/annotations.jsonl
export HORUS_GEMMA_REVISION=842da3794eaa0b77d5f08bae87a17459d91ff475
export HORUS_GPTOSS_REVISION=b5c939de8f754692c1647ca79fbf85e8c1e70f8a
test -s "$HORUS_SAMPLE_PATH"
test -s "$HORUS_FIRST_GEMMA"
test -s "$HORUS_FIRST_GPT"
"$HORUS_BERT_PYTHON" -m pytest -q tests/test_theme_llm_600.py tests/test_theme_llm_second_pass.py
```

### 2. Preparar as solicitações compartilhadas

```bash
export HORUS_REQUESTS_PATH="$("$HORUS_BERT_PYTHON" -m horus_benchmark.audit.theme_llm_second_pass_cli \
  --config "$HORUS_SECOND_CONFIG" prepare \
  --sample "$HORUS_SAMPLE_PATH" \
  --first-gemma "$HORUS_FIRST_GEMMA" \
  --first-gpt-oss "$HORUS_FIRST_GPT")"
printf '%s\n' "$HORUS_REQUESTS_PATH" > results/logs/theme_llm_second_pass.requests.path
"$HORUS_BERT_PYTHON" - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["HORUS_REQUESTS_PATH"])
m = json.loads((p.parent / "requests_metadata.json").read_text(encoding="utf-8"))
print("Solicitações:", m["rows"], "SHA-256:", m["requests_sha256"])
print("Truncados:", m["truncated_rows"])
print("Máximo de tokens:", m["max_input_tokens_by_model"])
assert m["rows"] == 170 and m["truncated_rows"] == 0
PY
```

Se a seleção não produzir exatamente 170 casos completos, pare e envie o
erro. O preparo confere os hashes dos rótulos da primeira passagem, da
amostra, da taxonomia, dos prompts e dos tokenizadores fixados.

### 3. Smoke de 12 prompts longos

```bash
export HORUS_SMOKE_LIMIT=12
export HORUS_SMOKE_STRATEGY=longest
export HORUS_GEMMA_OUTPUT=results/theme_llm_second_pass/smoke/gemma4_31b
export HORUS_GPTOSS_OUTPUT=results/theme_llm_second_pass/smoke/gpt_oss_120b
gemma_smoke="$(sbatch --parsable --export=ALL --time=06:00:00 scripts/slurm_theme_llm_gemma_abaporu.sh)"
gpt_smoke="$(sbatch --parsable --export=ALL --time=06:00:00 scripts/slurm_theme_llm_gptoss_abaporu.sh)"
printf 'Gemma smoke=%s; GPT-OSS smoke=%s\n' "$gemma_smoke" "$gpt_smoke"
```

Depois que ambos terminarem:

```bash
sacct -j "$gemma_smoke,$gpt_smoke" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
"$HORUS_BERT_PYTHON" - <<'PY'
import json, os
from pathlib import Path
for key in ("HORUS_GEMMA_OUTPUT", "HORUS_GPTOSS_OUTPUT"):
    folder = Path(os.environ[key])
    rows = [json.loads(line) for line in (folder / "annotations.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 12 and all(row["status"] == "valid" for row in rows)
    print(key, "12 respostas válidas")
PY
```

### 4. Executar os 170 e avaliar

```bash
unset HORUS_SMOKE_LIMIT HORUS_SMOKE_STRATEGY
export HORUS_GEMMA_OUTPUT=results/theme_llm_second_pass/inference/gemma4_31b
export HORUS_GPTOSS_OUTPUT=results/theme_llm_second_pass/inference/gpt_oss_120b
gemma_job="$(sbatch --parsable --export=ALL scripts/slurm_theme_llm_gemma_abaporu.sh)"
gpt_job="$(sbatch --parsable --export=ALL scripts/slurm_theme_llm_gptoss_abaporu.sh)"
printf 'Gemma=%s; GPT-OSS=%s\n' "$gemma_job" "$gpt_job"
```

Depois de ambos terminarem, confira `sacct` e os arquivos `.err` em
`results/logs`. Em seguida:

```bash
sacct -j "$gemma_job,$gpt_job" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
"$HORUS_BERT_PYTHON" -m horus_benchmark.audit.theme_llm_second_pass_cli \
  --config "$HORUS_SECOND_CONFIG" evaluate \
  --sample "$HORUS_SAMPLE_PATH" \
  --first-gemma "$HORUS_FIRST_GEMMA" \
  --first-gpt-oss "$HORUS_FIRST_GPT" \
  --second-gemma "$HORUS_GEMMA_OUTPUT/annotations.jsonl" \
  --second-gpt-oss "$HORUS_GPTOSS_OUTPUT/annotations.jsonl"
```

O último comando só produz o relatório quando há 170 respostas válidas por
modelo, com os mesmos IDs e proveniência. Envie `second_pass_summary.json`,
`second_pass_rows.csv` e as duas saídas `annotations.jsonl` com seus
`metadata.json` para analisar recuperação de cobertura por área e fonte.
Não adicione `requests.jsonl` ao Git: ele contém os textos humanos.
