# Concordância de `theme` em 600 textos humanos

Este protocolo usa o corpus imutável `dataset_final_completo`. Ele compara
**Gemma 4 31B IT, GPT-OSS 120B e BERTimbau** nos mesmos 600 textos humanos:
100 por `broad_area`. Reaproveita apenas os **IDs e textos** dos 120 casos do
antigo piloto e sorteia outros 80 por área. Não lê respostas humanas, não
treina com humanos e não considera concordância entre modelos como acurácia.

Os dois LLMs usam os mesmos seis prompts versionados. Cada prompt inclui todos
os temas da área carregados de `themes_full_dataset.csv`, regras de decisão e
exemplos ilustrativos. O texto, a área e a lista de temas são as únicas
informações enviadas ao modelo. `theme` original, fonte, BERT e a resposta do
outro LLM não aparecem no prompt. O texto é truncado nos primeiros 6.000
caracteres quando necessário; o BERT usa os primeiros 512 tokens. A diferença
de janela de contexto limita comparações causais entre arquiteturas.

## Arquivos

- `configs/theme_llm_600.yaml`: amostragem, modelos e análise.
- `configs/llm_prompts_v1.yaml`: instruções e exemplos por área.
- `src/horus_benchmark/audit/theme_llm_600.py`: amostragem e concordância.
- `src/horus_benchmark/audit/theme_llm_inference.py`: vLLM local retomável.
- `scripts/slurm_theme_llm_*_abaporu.sh`: uma H200 por LLM, ambientes separados.
- `scripts/slurm_theme_bert_abaporu.sh`: treino e previsão BERT em ambiente
  Python 3.11 isolado. Previsões BERT já produzidas com o mesmo
  corpus/taxonomia/mapa também podem ser transferidas com seu `run.json`.

Os scripts Slurm são para o **Benchmark-Horus**, não para o `Text-Generator`.
Não execute `cluster/submit_local_campaign.sh`: ele gera textos sintéticos.

## Execução pelo pesquisador

O checkout esperado no Abaporu é
`/mnt/recod/homes/victor.roquetto/Benchmark-Horus`. Crie um ambiente
Python 3.11 separado para BERT e auditoria. O ambiente do Gemma contém a
stack de inferência validada no handoff; o do GPT-OSS é separado. Não atualize
as stacks vLLM durante a campanha sem repetir os smokes.

Na shell de login do Abaporu, instale somente o novo ambiente BERT. O PyTorch
2.8 e o Transformers 4.57.6 reproduzem a combinação já verificada no cluster
anterior; a compatibilidade CUDA no Abaporu ainda deve ser testada em uma H200.

```bash
cd /mnt/recod/homes/victor.roquetto/Benchmark-Horus
git pull --ff-only
export HORUS_BERT_ENV=/mnt/recod/homes/victor.roquetto/miniconda3/envs/horus_bert_theme
export HORUS_BERT_PYTHON="$HORUS_BERT_ENV/bin/python"
/mnt/recod/homes/victor.roquetto/miniconda3/bin/conda create -y -p "$HORUS_BERT_ENV" python=3.11 pip
"$HORUS_BERT_PYTHON" -m pip install 'torch==2.8.0' 'transformers==4.57.6' -e '.[theme]' pytest
"$HORUS_BERT_PYTHON" -m pip check
"$HORUS_BERT_PYTHON" -c 'import torch,transformers,pandas,pyarrow,sklearn,yaml,pytest; print(torch.__version__,torch.version.cuda,transformers.__version__)'
mkdir -p results/logs
export HORUS_AUDIT_PYTHON="$HORUS_BERT_PYTHON"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"$HORUS_AUDIT_PYTHON" -m pytest -q tests/test_theme_llm_600.py
```

Antes de treinar, verifique CUDA **dentro da alocação**. O nó de login pode
mostrar CUDA indisponível mesmo com o ambiente correto.

```bash
srun --partition=h200 --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=2 --mem=8G --time=00:10:00 \
  "$HORUS_BERT_PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

Baixe o checkpoint BERTimbau no login **antes** do job. Os scripts Slurm usam
somente o cache local e exigem esta revisão exata:

```bash
"$HORUS_BERT_PYTHON" - <<'PY'
from huggingface_hub import hf_hub_download
repo = 'neuralmind/bert-base-portuguese-cased'
revision = '94d69c95f98f7d5b2a8700c420230ae10def0baa'
for filename in ('config.json', 'vocab.txt', 'pytorch_model.bin'):
    print(filename, hf_hub_download(repo, filename, revision=revision))
PY
```

Se o login não tiver acesso ao Hugging Face, copie o cache dessa revisão do
cluster anterior para `~/.cache/huggingface/hub` antes de submeter o BERT. Os
três arquivos acima precisam estar disponíveis localmente.

Antes de submeter, confira **sem atualizar os ambientes validados** que os
dois Pythons LLM conseguem importar `numpy`, `pandas`, `pyarrow` e `yaml`,
além de `torch`, `transformers` e `vllm`. Se faltar uma dependência de leitura
de dados, resolva isso separadamente da stack CUDA/vLLM e repita o smoke.

```bash
export HORUS_GEMMA_PYTHON=/mnt/recod/homes/victor.roquetto/venvs/horus_gemma4/bin/python
export HORUS_GPTOSS_PYTHON=/mnt/recod/homes/victor.roquetto/miniconda3/envs/horus_gptoss/bin/python
for python_bin in "$HORUS_GEMMA_PYTHON" "$HORUS_GPTOSS_PYTHON"; do
  "$python_bin" -c 'import numpy,pandas,pyarrow,yaml,torch,transformers,vllm; print(torch.__version__,transformers.__version__,vllm.__version__)'
  "$python_bin" -m pip check
done
```

Para reconstruir no Abaporu os mesmos 120 IDs do piloto anterior, use a
mesma semente e o mesmo corpus completo. A folha nova estará vazia; **nenhuma
resposta humana é usada**. Se o piloto antigo já estiver disponível com seu
`run.json`, você pode apontar `HORUS_PILOT` diretamente para ele. A versão,
hash do corpus, IDs, áreas e textos são conferidos. A amostra final não contém
tema bruto nem resposta humana:

```bash
export HORUS_LLM_CONFIG=experiments/01_quality_audit/theme_agreement_600/configs/theme_llm_600.yaml
export HORUS_PILOT="$("$HORUS_AUDIT_PYTHON" -m horus_benchmark.audit.theme_cli \
  --config experiments/01_quality_audit/human_theme_validation/configs/theme_v1.yaml pilot)"
HORUS_SAMPLE_PATH="$("$HORUS_AUDIT_PYTHON" -m horus_benchmark.audit.theme_llm_cli --config "$HORUS_LLM_CONFIG" sample --pilot "$HORUS_PILOT")"
printf '%s\n' "$HORUS_SAMPLE_PATH" > results/logs/theme_llm_600.sample.path
export HORUS_SAMPLE_PATH
```

Faça um smoke por `sbatch` em diretórios próprios. `HORUS_SMOKE_LIMIT=12`
seleciona dois textos por área; não consuma o diretório do lote final:

```bash
export HORUS_SMOKE_LIMIT=12
export HORUS_GEMMA_OUTPUT=results/theme_llm_600/smoke/gemma4_31b
export HORUS_GPTOSS_OUTPUT=results/theme_llm_600/smoke/gpt_oss_120b
sbatch scripts/slurm_theme_llm_gemma_abaporu.sh
sbatch scripts/slurm_theme_llm_gptoss_abaporu.sh
```

Depois de conferir os logs e as 12 respostas válidas por modelo, submeta o
lote final. O runner retoma respostas válidas após preempção ou limite de
tempo, desde que amostra, prompts, checkpoint e diretório sejam os mesmos.
Respostas inválidas são reprocessadas na retomada. Checkpoints são lidos dos
snapshots locais (`refs/main`) e o SHA efetivo fica em `metadata.json`.

```bash
unset HORUS_SMOKE_LIMIT
export HORUS_GEMMA_OUTPUT=results/theme_llm_600/inference/gemma4_31b
export HORUS_GPTOSS_OUTPUT=results/theme_llm_600/inference/gpt_oss_120b
gemma_job="$(sbatch --parsable scripts/slurm_theme_llm_gemma_abaporu.sh)"
gpt_job="$(sbatch --parsable scripts/slurm_theme_llm_gptoss_abaporu.sh)"
printf 'Gemma job %s; GPT-OSS job %s\n' "$gemma_job" "$gpt_job"
```

O BERT usa exclusivamente sintéticos de `train`, seleciona a época em
`validation` e prevê todos os textos humanos. O ambiente BERT criado acima
precisa passar no teste CUDA; não use os ambientes Gemma ou GPT-OSS para ele.
O checkpoint BERTimbau está fixado pelo SHA da rodada anterior.
Se já houver previsões completas do mesmo corpus, transporte o Parquet e o
`run.json` correspondente, preservando ambos.

```bash
bert_job="$(sbatch --parsable scripts/slurm_theme_bert_abaporu.sh)"
printf 'BERT job %s\n' "$bert_job"
```

Após os três jobs terminarem, rode a comparação com o mesmo Python de
auditoria. Substitua o ID do job BERT se a sessão shell tiver sido encerrada:

```bash
export HORUS_SAMPLE_PATH="$(cat results/logs/theme_llm_600.sample.path)"
export HORUS_BERT_PREDICTIONS="$(cat "results/logs/horus-bert-theme-${bert_job}.predictions.path")"
HORUS_AGREEMENT_REPORT="$("$HORUS_AUDIT_PYTHON" -m horus_benchmark.audit.theme_llm_cli --config "$HORUS_LLM_CONFIG" evaluate \
  --sample "$HORUS_SAMPLE_PATH" \
  --gemma "$HORUS_GEMMA_OUTPUT/annotations.jsonl" \
  --gpt-oss "$HORUS_GPTOSS_OUTPUT/annotations.jsonl" \
  --bert "$HORUS_BERT_PREDICTIONS")"
printf 'Relatório: %s\n' "$HORUS_AGREEMENT_REPORT"
```

O último comando exige **600 rótulos válidos por LLM** e as mesmas 600 IDs
nas previsões BERT. Produz `agreement.json`, `agreement_rows.csv`,
`agreement_by_factor.csv`, `pairwise_confusion.csv`,
`theme_counts_by_model.csv` e `disagreements.csv`. Não inclui texto integral
nas tabelas. Calcula acordo bruto e κ global e condicionado à área, com
intervalo bootstrap de κ; examina área, fonte (quando há pelo menos 20
casos), comprimento, confiança do BERT e presença de mapeamento manual.
`incerto` e `fora_taxonomia` são categorias explícitas. O BERT é comparado
como top-1 forçado e com limiar operacional fixo de 0,95. O mapeamento manual
é registrado como fator, mas **não substitui a previsão BERT** na comparação.
O κ condicionado à área é a análise principal: o κ global convencional é
suplementar, pois os três classificadores já recebem a mesma área conhecida.

O arquivo `fixed_bert_threshold.json` permite, **após inspecionar o relatório**,
executar a materialização BERT existente para obter as tabelas de humanos e
sintéticos. Isso atribui temas por mapa manual ou BERT; nenhum LLM é aplicado
automaticamente a todo o corpus humano. A materialização produz uma versão
derivada e não modifica os Parquets originais.

```bash
HORUS_THRESHOLD="$(dirname "$HORUS_AGREEMENT_REPORT")/fixed_bert_threshold.json"
"$HORUS_AUDIT_PYTHON" -m horus_benchmark.audit.theme_cli --config "$HORUS_LLM_CONFIG" materialize \
  --predictions "$HORUS_BERT_PREDICTIONS" --threshold "$HORUS_THRESHOLD"
```

## Limites de interpretação

κ e acordo entre LLMs/BERT medem **concordância**, não precisão temática.
Modelos podem compartilhar vieses ou reproduzir associações da taxonomia.
Os 600 textos são equilibrados por área, não por tema ou fonte; médias globais
são médias da amostra equilibrada, não estimativas ponderadas da população
completa. Cem textos por área ainda são poucos para avaliar individualmente
os 94 temas. Comparações por fonte e comprimento são exploratórias.
Os temas sintéticos usados para treinar BERT são intenções de geração e podem
não corresponder exatamente ao conteúdo produzido. O relatório e o artigo
devem explicitar a ausência de anotação humana ou especializada.
