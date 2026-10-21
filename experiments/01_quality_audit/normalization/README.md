# Normalização do corpus completo — N1 e N2 candidata

**Entrada única:** recurso `corpus` de `configs/datasets/horus_full.yaml`, hoje
`dataset_final_completo/data/all.parquet` (195.154 registros). Nenhum arquivo
`train`, `validation` ou `test` é lido. O valor da coluna `split` passa intacto
para a view derivada, mas não participa das decisões. O mesmo vale para
`label`, `label_name`, fonte e gerador: só são usados em resumos *após* as
transformações. Não há seleção, reclassificação, reescrita por LLM ou alteração
do corpus de origem. A execução cabe ao pesquisador (regra 19 do `AGENTS.md`).

## Regras e limites

- **N1 técnica:** Unicode NFC; tentativa de desfazer mojibake apenas quando
  uma conversão reversível reduz sinais suspeitos; CRLF/controles/espaços;
  `<br>` e `<p>/<div>` editoriais e entidades HTML fora de blocos de código;
  URL/e-mail literais por `URL`/`EMAIL`, também fora de código.
- **N2 conservadora candidata:** N1 mais remoção de uma linha terminal estrita de
  copyright, atualização editorial ou comentário de plataforma; abstract
  paralelo só quando há resumo português anterior, introdução portuguesa
  posterior e evidência lexical forte de inglês no trecho intermediário.
  Bibliografias **não são removidas** nesta configuração: antes, comparamos
  sua prevalência entre classes por área e inspecionamos sua função nos
  gêneros textuais. Os demais casos são apenas sinalizados. Não remove
  declaração de hipótese, citação interna, lista, título, Markdown, gíria
  ou erro gramatical.
- **Sinalização de alta precisão:** `review` exige evidência mais forte:
  abstract longo com cabeçalho explícito e predomínio de marcadores lexicais
  ingleses; mojibake residual não ASCII; pelo menos três tags HTML reais
  residuais fora de código; perda substancial; texto vazio; ou metadado
  de raciocínio vazado ou hash original divergente.
  Cabeçalhos de referências, bibliografia terminal provável, placeholders,
  preâmbulos, HTML isolado/protegido, Markdown, idioma inválido segundo o
  metadado de origem e provável truncamento ficam apenas como diagnósticos.
  A fila **não é** uma lista de exclusão. Um falso
  negativo sutil é preferível a tratar redação autêntica como defeito. A
  coluna `review_reasons` identifica quais sinais fortes acionaram a fila;
  `flags` também inclui sinais diagnósticos que não a acionam. A trace
  Parquet preserva ambos. A detecção de idioma completo e de incoerência
  semântica **não** é resolvida por essas regras: exige auditoria posterior
  (metadados/inspeção assistida), nunca correção automática do estilo.
  O código usa regras determinísticas, não agentes/LLMs por texto. O indício
  de inglês usa um limiar de 40 palavras e contagens de palavras funcionais
  inglesas versus portuguesas; é uma triagem imperfeita, não uma anotação
  linguística validada. HTML só é reconhecido por nomes de tags conhecidas,
  sem atravessar linhas; desigualdades matemáticas não devem ser tratadas
  como tags. Exemplos de código são protegidos da conversão editorial.
- **Guarda:** com pelo menos 50 palavras originais, perda de pelo menos 20
  palavras e retenção inferior a 90% gera `SUBSTANTIAL_LOSS`. Esses valores
  estão em YAML e são hipóteses de auditoria, não limiares finais de seleção.

O arquivo `normalized_corpus.parquet` tem as mesmas colunas e linhas do corpus
de origem. `content`, `char_count`, `word_count`, `content_hash` e
`normalized_hash` refletem o texto derivado. Neste artefato,
`normalized_hash` é explicitamente SHA-256 de NFKC + casefold + colapso de
espaços; **não** se pressupõe que essa era a semântica da coluna original.
As contagens de palavras derivadas usam separação por espaços (`str.split`),
não tokenização de um modelo.
Todos os demais campos, incluindo grupos documentais e splits legados,
permanecem inalterados. Os hashes e contagens anteriores ficam em
`normalization_trace.parquet`. Grupos de duplicatas existentes continuam
metadados de origem, não uma nova deduplicação; duplicatas surgidas após
normalização precisam de uma auditoria separada.

Cada rodada ainda produz `summary.json`, `review_queue.csv`,
`paired_surface_summary.csv`, `flag_prevalence_by_label_area.csv`,
`verification.json`, tabelas, figuras e
`review_examples_blind.jsonl`/`review_examples_manifest.csv`. A seleção
determinística de até 480 exemplos de revisão cobre flags por área e controles
sem flag; não estima prevalências. A planilha de metadados fica separada do
texto para reduzir viés durante a leitura. As figuras comparam os mesmos IDs
antes/depois, mas somente características
superficiais. Redução de uma diferença entre classes não demonstra, isoladamente,
que a transformação preservou significado ou melhorou o benchmark.
Como os CSVs contêm metadados de origem potencialmente não confiáveis, abra-os
como texto/CSV sem avaliação automática de fórmulas; não edite os textos brutos.
`tables/class_gap_delta.csv` mede mudanças descritivas das diferenças entre
classes por área; não é uma sonda treinada nem substitui controles de gênero,
comprimento ou fonte.
`flag_prevalence_by_label_area.csv` usa o número **total** de linhas por classe
e área como denominador; ao comparar bibliografia, filtre `REFERENCES` (sinal
amplo) e `TERMINAL_REFERENCES_CANDIDATE` (sinal estrito), sem confundi-los com
uma contagem validada de bibliografias. A amostra `review_examples_manifest.csv`
é estratificada por flags e **não** permite inferir prevalência populacional.

## Execução no cluster

Na raiz do checkout que contém o corpus, execute os passos abaixo **um por vez**.
Não cole `set -e` em um terminal interativo: uma falha encerra o shell antes
de você conseguir examinar o erro. O wrapper usa o Python de
`HORUS_NORMALIZE_PYTHON` ou, por padrão, o ambiente `horus_theme` documentado
nos demais scripts do projeto. Ajuste a variável se o ambiente do cluster
for outro. O diretório de saída é novo por ID do job.

```bash
set +e
mkdir -p results/logs
export HORUS_NORMALIZE_PYTHON="${HORUS_NORMALIZE_PYTHON:-$HOME/miniconda3/envs/horus_theme/bin/python}"
"$HORUS_NORMALIZE_PYTHON" --version
"$HORUS_NORMALIZE_PYTHON" -m pip install -e '.[audit,dev]'
"$HORUS_NORMALIZE_PYTHON" -m pytest tests/test_normalization.py tests/test_audit_integrity.py -q
```

**Só depois de todos os testes passarem**, submeta separadamente:

```bash
sbatch --export=ALL,HORUS_NORMALIZE_STAGE=n1 scripts/slurm_normalize_corpus.sh
```

Se qualquer comando falhar, pare antes do `sbatch` e copie a mensagem de erro.
Não execute o wrapper com `bash scripts/slurm_normalize_corpus.sh`: ele precisa
das variáveis de ambiente que o Slurm fornece. Se um job tiver sido submetido
após falha de teste,
verifique seu estado com `squeue -j ID` e `sacct -j ID --format=JobID,State,ExitCode`.
Se ainda estiver ativo, cancele-o com `scancel ID` antes de sincronizar o
código corrigido e repetir o bloco. Preserve a pasta de saída antiga para
diagnóstico, mas não a utilize como versão validada do corpus.

Quando o job N1 terminar, confirme `verification.json` e compartilhe comigo
o caminho impresso no log, `summary.json`, `paired_surface_summary.csv`,
`flag_prevalence_by_label_area.csv` e os exemplos de revisão. Não trate uma
pasta contendo `.part` como execução concluída.
N2 é uma **ablação candidata**, não substitui N1 automaticamente. Após
inspecionarmos N1:

```bash
sbatch --export=ALL,HORUS_NORMALIZE_STAGE=n2 scripts/slurm_normalize_corpus.sh
```

Nenhum desses comandos escolhe o benchmark ou treina detector. Um protocolo
de seleção e seus splits deverão ser definidos depois, usando as views
derivadas e controles de confusão. Os arquivos em `results/normalization/`
contêm textos/trechos do corpus e ficam fora do Git.

O wrapper também prepara `audit_dataset.yaml` e `audit_experiment.yaml` para
uma **auditoria separada e posterior** da view derivada, com as sondas do
protocolo existente. Essa auditoria não é executada pelo job de normalização.
Ela não compara Parquets físicos de split originais contra conteúdo derivado;
preserva a coluna de split apenas para eventual sonda posterior. Para uma
comparação pareada com a auditoria original, use `horus-audit compare` somente
após confirmar que ambas as rodadas têm exatamente os mesmos IDs e metadados
protegidos. As sondas exigem tempo adicional e não são requisito para a
normalização inicial.

Se optar pela auditoria completa posterior, no checkout do cluster:

```bash
export HORUS_NORMALIZE_RUN='results/normalization/full-n1-ID_DO_JOB'
export HORUS_ORIGINAL_AUDIT_RUN='results/audits/RODADA_ORIGINAL_COMPLETA'
sbatch --export=ALL scripts/slurm_audit_normalized.sh
```

Substitua os dois caminhos por rodadas **existentes naquele checkout**. Se a
auditoria original ainda não estiver no cluster, não informe
`HORUS_ORIGINAL_AUDIT_RUN`: o job fará apenas a auditoria derivada; a comparação
pareada poderá ser rodada depois de disponibilizar a rodada original. O job
adicional usa os splits herdados apenas para as sondas diagnósticas, jamais
para decidir a normalização.
