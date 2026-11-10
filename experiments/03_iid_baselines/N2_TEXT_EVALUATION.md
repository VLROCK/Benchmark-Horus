# Avaliação textual N2 v1: IID, LOGO, LODO e LOSO

## Escopo congelado

Esta execução consome o release verificado `horus-n2-iid-v1`: **107.566 textos N2**, sendo **53.783 humanos e 53.783 sintéticos**, em seis áreas. O corpus original e os splits não são alterados. Três sementes (`20261105`, `20261106`, `20261107`) definem três versões de holdout IID; dentro de cada desenvolvimento há cinco folds. Os três holdouts reutilizam a mesma coleção de documentos e, portanto, não são estudos independentes.

A unidade desta matriz principal é o **documento selecionado**. As análises preexistentes de parágrafos, frases e pareamento *nuisance* continuam como controles secundários e têm populações diferentes; suas métricas não são misturadas nesta matriz.

`source_name`, `generator`, `broad_area`, `theme` e outros metadados **não entram nos modelos**. Área, fonte e gerador são usados exclusivamente para construir e descrever avaliações. Não existe sonda só de metadados neste pacote.

## Modelos e sondas

| Nome | Entrada | Modelo | Papel |
|---|---|---|---|
| `surface_length` | três medidas de tamanho calculadas do texto | regressão logística | sonda de comprimento |
| `surface_presentation` | parágrafos, linhas, marcadores Markdown/HTML/URL/EMAIL e sinais de formatação | regressão logística | sonda de apresentação |
| `surface_punctuation` | densidade de pontuação, dígitos, maiúsculas e diversidade lexical | regressão logística | sonda superficial/estilométrica |
| `surface_all` | união das três famílias | regressão logística | sonda conjunta |
| `word_lr` | TF-IDF de unigramas e bigramas | regressão logística | baseline lexical |
| `char4_lr` | TF-IDF de 4-gramas de caracteres | regressão logística | baseline de caracteres |
| `char4_svm` | mesmo TF-IDF de caracteres | Linear SVM | baseline clássico alternativo |
| `human_iforest` | TF-IDF de palavras, SVD e Isolation Forest | humano somente | baseline unário |
| `bertimbau` | tokens N2, até 512 tokens | `neuralmind/bert-base-portuguese-cased` | baseline neural |

As sondas detectam separabilidade residual; seus coeficientes não demonstram causalidade nem necessariamente indicam defeito do benchmark. Os modelos TF-IDF usam `min_df=5`, `max_df=0.95` e até 200.000 características. Para modelos lineares, `C ∈ {0.1,1,10}` é escolhido pela média de ROC-AUC nos quatro folds **internos do treino**, com pré-processamento ajustado novamente em cada fold. A validação externa serve para selecionar limiares, e o teste fica intocado. O modelo unário ajusta vocabulário, SVD de 256 dimensões e Isolation Forest de 300 árvores apenas sobre humanos de treino.

BERTimbau usa o commit de checkpoint fixado em `configs/detectors/bertimbau_n2_v1.yaml`, sequência máxima de 512 tokens, até três épocas, AdamW (`2e-5`, decaimento `0.01`, warmup `0.10`), batch 16 e acumulação 2. A época é escolhida pela perda de validação. A fração de textos truncados é registrada. O checkpoint precisa estar no cache local do cluster antes dos jobs GPU.

## Protocolos e métricas

* **IID:** o teste de 15% do split congelado é fixo em cada semente; cada fold de desenvolvimento alterna como validação e os outros quatro treinam.
* **LOGO:** todos os textos selecionados de um gerador sintético são retidos; humanos de controle são pareados por área e célula de comprimento N2.
* **LODO:** todos os textos, das duas classes, de uma área ficam retidos. Não há pareamento adicional de classe.
* **LOSO:** todos os humanos de uma família de fonte, dentro de uma área, ficam retidos; sintéticos de controle são pareados por área e célula de comprimento N2. Aliases explícitos de URL/fonte estão na configuração OOD. Cenários sem suporte mínimo ficam `unsupported`, não são forçados.

Nos protocolos OOD, registros relacionados por `probe_group_id` não cruzam treino, validação e teste. O alvo OOD inclui registros originalmente no desenvolvimento IID; os demais registros do teste IID são embargados do treino. Alvos sem controle pareado permanecem identificados, mas ficam fora de ROC-AUC e outras métricas binárias. `paired_iid_test` marca apenas IDs que podem ser comparados com previsões IID sobre os mesmos documentos. A tabela de pareamento informa quando isso não tem suporte. A diferença IID × OOD reflete o protocolo completo: também pode haver exclusão de controles relacionados no treino OOD, então não isola causalmente um único fator.

ROC-AUC, balanced accuracy e TPR nos FPRs de 1% e 5% são calculados no teste. O limiar de balanced accuracy maximiza Youden na validação. Os outros dois limiares são escolhidos para FPR empírico de validação no máximo 1% e 5%, tratando empates conservadoramente. O FPR realmente observado no teste também é reportado; ele pode exceder o alvo. IC95% percentis usam 2.000 reamostragens estratificadas por classe de grupos `probe_group_id` completos. Os folds do mesmo teste compartilham o mesmo sorteio de grupos. Esses IC são condicionais ao modelo treinado e aos limiares; as três sementes são exibidas como variação descritiva, sem tratá-las como datasets independentes.

Os oito modelos CPU são planejados em **15 rodadas IID por modelo (120 tarefas)**. BERTimbau tem **15 tarefas IID**. No OOD, quatro modelos CPU (`surface_all`, `word_lr`, `char4_svm`, `human_iforest`) cobrem todos os alvos elegíveis nas três sementes e cinco folds; BERTimbau cobre todos os alvos elegíveis em **um fold por semente**, para conter o custo. O número exato de alvos LOGO/LODO/LOSO elegíveis sai de `results/protocols/ood-n2-v1/tasks.csv` após a construção. A macro OOD atribui peso igual a cada alvo elegível e é descritiva; métricas por alvo devem ser consultadas antes de interpretá-la.

O plano IID CPU também aplica os modelos **treinados em N2** aos mesmos IDs na view N3s, sem novo ajuste nem novos limiares. Isso mede uma perturbação de apresentação N2→N3s, não um benchmark N3s treinado separadamente. A N3s deve preservar todos os humanos selecionados byte a byte, requisito conferido na preparação.

## Comandos completos no cluster

Execute na ordem. Primeiro, no computador local, publique este código com `git add -A`, `git commit -m "Implement N2 IID and OOD text evaluation"` e `git push`. No cluster, em `bash`:

```bash
cd /home/victor.roquetto/Benchmark-Horus
git pull
PY="$HOME/miniconda3/envs/horus_theme/bin/python"
"$PY" -m pip install -e '.[dev,theme]'
"$PY" -m pytest tests/test_binary_detection_metrics.py tests/test_benchmark_runner.py tests/test_ood_manifests.py tests/test_text_baselines.py tests/test_bertimbau_baseline.py tests/test_summarize_benchmarks.py tests/test_slurm_text_array.py -q
mkdir -p results/logs results/tmp results/evaluation/n2-v1/plans
```

Confirme que os três Parquets grandes ignorados pelo Git estão materializados no cluster:

```bash
test -s results/releases/horus-n2-iid-v1/selected_index.parquet
test -s results/normalization/full-n2-116035/normalized_corpus.parquet
test -s results/normalization/full-n3s-126552/normalized_corpus.parquet
```

Construa os manifests OOD e o cache de textos em jobs independentes:

```bash
OOD_JOB="$(sbatch --parsable scripts/slurm_ood_n2_v1.sh)"
PREP_JOB="$(sbatch --parsable scripts/slurm_prepare_text_evaluation.sh)"
printf '%s\n' "$OOD_JOB" > results/tmp/last_ood_n2_job.txt
printf '%s\n' "$PREP_JOB" > results/tmp/last_text_prepare_job.txt
echo "OOD=$OOD_JOB PREP=$PREP_JOB"
```

Depois que ambos terminarem, em nova sessão se necessário:

```bash
OOD_JOB="$(cat results/tmp/last_ood_n2_job.txt)"
PREP_JOB="$(cat results/tmp/last_text_prepare_job.txt)"
PY="$HOME/miniconda3/envs/horus_theme/bin/python"
sacct -j "$OOD_JOB,$PREP_JOB" --format=JobID,JobName%24,State,ExitCode,Elapsed
tail -n 30 "results/logs/horus-ood-n2-${OOD_JOB}.out"
tail -n 30 "results/logs/horus-text-prepare-${PREP_JOB}.out"
test -s results/protocols/ood-n2-v1/tasks.csv
test -s results/evaluation/n2-v1/prepared/prepared_manifest.json
"$PY" -m horus_benchmark.protocols.ood_manifests check --output results/protocols/ood-n2-v1
cat results/protocols/ood-n2-v1/summary.json
```

Crie os oito planos. O procedimento `plan` apenas enumera tarefas e confere hashes, não treina modelos:

```bash
CFG=configs/experiments/text_evaluation_n2_v1.yaml
PLAN=results/evaluation/n2-v1/plans
PY="$HOME/miniconda3/envs/horus_theme/bin/python"
"$PY" -m horus_benchmark.evaluation.benchmark_runner plan --config "$CFG" --output "$PLAN/iid_cpu.csv" --protocols iid --models surface_length surface_presentation surface_punctuation surface_all word_lr char4_lr char4_svm human_iforest --counterfactual-n3s
"$PY" -m horus_benchmark.evaluation.benchmark_runner plan --config "$CFG" --output "$PLAN/iid_bert.csv" --protocols iid --models bertimbau
for PROTOCOL in logo lodo loso; do
  "$PY" -m horus_benchmark.evaluation.benchmark_runner plan --config "$CFG" --output "$PLAN/${PROTOCOL}_cpu.csv" --protocols "$PROTOCOL" --models surface_all word_lr char4_svm human_iforest
  "$PY" -m horus_benchmark.evaluation.benchmark_runner plan --config "$CFG" --output "$PLAN/${PROTOCOL}_bert.csv" --protocols "$PROTOCOL" --models bertimbau
done
wc -l "$PLAN"/*.csv
```

Pré-verifique o cache BERTimbau. Se o modelo ainda não estiver baixado, execute `snapshot_download` em um nó com acesso ao Hugging Face antes de submeter GPU. Os jobs GPU permanecem offline:

```bash
PY="$HOME/miniconda3/envs/horus_theme/bin/python"
"$PY" - <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
name = 'neuralmind/bert-base-portuguese-cased'
revision = '94d69c95f98f7d5b2a8700c420230ae10def0baa'
p = Path(snapshot_download(repo_id=name, revision=revision, local_files_only=True))
assert any((p / filename).is_file() for filename in ('model.safetensors', 'pytorch_model.bin'))
AutoTokenizer.from_pretrained(name, revision=revision, local_files_only=True)
AutoModelForSequenceClassification.from_pretrained(name, revision=revision, local_files_only=True, num_labels=2)
print('BERTimbau no cache:', p)
PY
```

Se esse teste falhar por falta do checkpoint e o nó de login tiver acesso ao Hugging Face, baixe apenas os arquivos necessários e repita a pré-verificação:

```bash
"$PY" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='neuralmind/bert-base-portuguese-cased',
    revision='94d69c95f98f7d5b2a8700c420230ae10def0baa',
    allow_patterns=['config.json', 'pytorch_model.bin', 'vocab.txt',
                    'tokenizer_config.json', 'special_tokens_map.json',
                    'added_tokens.json'],
)
PY
```

Submeta as tarefas. O script divide planos em arrays de até 400 tarefas, usa índices locais `0–399` no Slurm e passa o deslocamento do plano ao job; isso respeita clusters cujo `MaxArraySize` não aceita índices globais acima de 799. `HORUS_ARRAY_START` permite retomar um sufixo ainda não submetido e `HORUS_QOS` solicita um QOS explícito quando necessário. O terceiro argumento limita a simultaneidade **de cada array**, não o total de arrays. Preserve o diretório `runs` para que a agregação encontre todas as previsões:

```bash
PLAN=results/evaluation/n2-v1/plans
export HORUS_QOS=normal
bash scripts/submit_text_evaluation_plan.sh "$PLAN/iid_cpu.csv" cpu 4
bash scripts/submit_text_evaluation_plan.sh "$PLAN/iid_bert.csv" bert 1
for PROTOCOL in logo lodo loso; do
  bash scripts/submit_text_evaluation_plan.sh "$PLAN/${PROTOCOL}_cpu.csv" cpu 4
  bash scripts/submit_text_evaluation_plan.sh "$PLAN/${PROTOCOL}_bert.csv" bert 1
done
squeue -u "$USER"
```

Após o fim dos arrays, confira a cobertura antes do job de agregação. O bloco imprime faltantes por plano e retorna erro se houver qualquer tarefa incompleta. Tarefas completas são protegidas contra sobrescrita; repita **somente** os índices faltantes.

```bash
PY="$HOME/miniconda3/envs/horus_theme/bin/python"
"$PY" - <<'PY'
import csv
from pathlib import Path
plans = Path('results/evaluation/n2-v1/plans')
runs = Path('results/evaluation/n2-v1/runs')
missing_total = 0
for plan in sorted(plans.glob('*.csv')):
    with plan.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    missing = [(index, row['task_id']) for index, row in enumerate(rows)
               if not (runs / row['task_id'] / 'metrics.json').is_file()]
    print(f'{plan.name}: {len(rows) - len(missing)}/{len(rows)} completos; faltantes={missing[:20]}')
    missing_total += len(missing)
raise SystemExit(1 if missing_total else 0)
PY
```

Para repetir um único índice faltante, use um array Slurm local de um elemento e passe o índice global em `HORUS_ARRAY_OFFSET`. Para BERT, substitua o script CPU pelo `slurm_text_evaluation_bert.sh`. O exemplo abaixo repete o índice 800 do LOSO CPU:

```bash
PLAN=results/evaluation/n2-v1/plans/loso_cpu.csv
sbatch --qos=normal --array=0 \
  --export=ALL,HORUS_PLAN="$PLAN",HORUS_ARRAY_OFFSET=800 \
  scripts/slurm_text_evaluation_cpu.sh
```

Para retomar todos os índices `800–2159` ainda não submetidos do LOSO CPU, depois de confirmar que nenhum desses índices está ativo:

```bash
HORUS_QOS=normal HORUS_ARRAY_START=800 \
  bash scripts/submit_text_evaluation_plan.sh \
  results/evaluation/n2-v1/plans/loso_cpu.csv cpu 4
```

Agregue métricas, ICs, ablação N2→N3s, tabelas por alvo/subgrupo e figuras:

```bash
SUMMARY_JOB="$(sbatch --parsable scripts/slurm_summarize_text_evaluation.sh)"
printf '%s\n' "$SUMMARY_JOB" > results/tmp/last_text_summary_job.txt
echo "SUMMARY=$SUMMARY_JOB"
```

Depois do último job:

```bash
SUMMARY_JOB="$(cat results/tmp/last_text_summary_job.txt)"
sacct -j "$SUMMARY_JOB" --format=JobID,JobName%24,State,ExitCode,Elapsed
tail -n 40 "results/logs/horus-text-summary-${SUMMARY_JOB}.out"
tail -n 40 "results/logs/horus-text-summary-${SUMMARY_JOB}.err"
cat results/evaluation/n2-v1/summary/summary.json
ls -lh results/evaluation/n2-v1/summary
```

Não interprete os resultados se `summary.json` não indicar conclusão completa. Parquets de corpus, cache, previsões e checkpoints são ignorados pelo Git; tabelas, JSONs, manifests e figuras podem ser trazidos ao repositório para análise. O release N2 continua o único dataset principal congelado nesta etapa. Métodos zero-shot especializados, testes adversariais e classificação de `theme` humano são etapas posteriores, com adaptações PT-BR a definir separadamente.
