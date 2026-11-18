# Controles secundários no release N2 selecionado

## Pergunta e população

Estas análises medem quanto da separabilidade observada no benchmark documental `horus-n2-iid-v1` depende da unidade textual, do layout e de fatores observáveis de apresentação. O release possui 107.566 documentos, 53.783 por classe, três sementes de holdout IID (`20261105`–`20261107`) e cinco folds de desenvolvimento. Os textos N2, IDs, rótulos e splits são entradas congeladas. Nenhum controle aqui substitui o benchmark documental principal.

O uso de condições textuais alternativas segue a motivação de avaliações de robustez como RAID; aqui o escopo é uma ablação controlada do release Hórus, não uma reprodução dos ataques de RAID.

As regras e sementes estão em `configs/experiments/segment_views_n2_v1.yaml`, `secondary_controls_n2_v1.yaml` e `secondary_views_n2_v1.yaml`. Cada arquivo derivado registra caminho e SHA-256 dos insumos e resultados. A preparação recusa um cache, release ou split que não corresponda aos manifests verificados.

## Unidades e cobertura

Para cada documento, a extração determinística busca **um** parágrafo natural e **uma** frase completa. O ponto médio do segmento deve estar entre 35% e 65% do texto; escolhe-se entre os elegíveis de forma estável. O parágrafo deve ter 25–350 palavras e no máximo 35% do comprimento documental; a frase, 8–80 palavras e no máximo 25%. A análise de elegibilidade e a seleção usam o mesmo algoritmo e registram o motivo de exclusão. Não se substitui um segmento inelegível por outro da introdução ou conclusão.

O segmento carrega `parent_record_id`, `record_id`, `probe_group_id`, rótulo, área, fonte/gerador, posição e hash. `record_id` identifica o documento original para permitir pareamento exato de previsões; `segment_id` identifica a extração. Os splits e folds são herdados do pai, sem sorteio novo. A cobertura e retenção são publicadas por classe, área, fonte/gerador, faixa de comprimento e semente. Isso é essencial, pois a elegibilidade de parágrafos já diferiu muito entre classes no ensaio anterior. Os resultados das unidades se aplicam apenas aos documentos elegíveis.

## Três comparações

1. **Modelo documental aplicado à unidade:** reutiliza o ajuste documental congelado e seus limiares de validação. Para CPU, reconstrói o ajuste com os mesmos dados, configuração, `C` e semente e exige reproduzir os escores documentais salvos antes de pontuar a nova vista. Para BERTimbau, reutiliza o checkpoint SHA-256 congelado e confere escores documentais numa amostra de 128 textos por classe. Os mesmos modelos são aplicados aos parágrafos, frases e duas vistas de layout.
2. **Modelo treinado na unidade:** refaz treino e validação apenas com pais elegíveis, mantendo os folds originais. Parágrafos e frases são experimentos separados. Hiperparâmetros lineares são escolhidos nos folds internos de treino; limiares vêm da validação da unidade. A referência documental é avaliada nos mesmos IDs de teste, mas seu treino original tinha mais documentos. Por isso, a comparação é descritiva; o efeito isolado de mudar a entrada é lido na comparação 1.
3. **Modelo treinado no layout alterado:** refaz treino, validação e teste nos mesmos IDs documentais, com a mesma regra aplicada a ambas as classes. `layout_blank` remove apenas separações de parágrafo por linha vazia. `layout_all` reutiliza o controle anterior que achata quebras de linha. Ambos preservam a sequência de tokens separados por espaço. Esta é uma ablação de apresentação, não uma nova normalização do corpus.

Os nove modelos IID existentes são reutilizados: `surface_length`, `surface_presentation`, `surface_punctuation`, `surface_all`, `word_lr`, `char4_lr`, `char4_svm`, `human_iforest` e `bertimbau`. Cada modelo tem três sementes e cinco folds. Há **675 tarefas**: 135 aplicações documentais às quatro vistas, 270 treinos nas duas unidades e 270 treinos nas duas vistas de layout. As 60 tarefas de novo treino BERTimbau exigem GPU; as 15 aplicações de checkpoint BERTimbau também exigem GPU para inferência. Os escores, limiares e predições por ID são persistidos.

## Controle *nuisance*

O pareamento é definido **antes dos escores** e usa exclusivamente o teste congelado de cada semente. Há pareamento determinístico 1:1 humano/sintético em células de área e `floor(log2(1 + palavras)/0,25)`. Para parágrafo e frase, exige-se também a mesma faixa de comprimento do documento pai. O controle documental mais restrito adiciona faixas de parágrafos (`1`, `2–4`, `5+`) e presença de marcador de lista Markdown. Não há pareamento por `theme` bruto, score do detector ou acerto. Cada par é uma unidade de auditoria de suporte, não uma equivalência semântica entre textos.

São publicados a fração retida por classe/área/fonte/gerador, as células sem suporte, SMD e KS antes/depois. O pareamento não altera o treino ou a calibração. Se a retenção for pequena ou o equilíbrio residual ruim, a interpretação deve ser restrita a sensibilidade em suporte comum; não se deve apresentar esse subconjunto como benchmark principal.

## Métricas e inferência

ROC-AUC, balanced accuracy, TPR e **FPR observado** nos limiares de FPR de validação de 1% e 5% são calculados por modelo, semente e vista. Cada delta compara a vista e o documento **nos mesmos IDs de teste**, nunca populações distintas. Os cinco folds compartilham esses IDs; a média de fold é a unidade de relato. IC95% percentis usam 2.000 sorteios vinculados de `probe_group_id`, estratificados por classe e compartilhados entre os escores do par. As três sementes reutilizam documentos e são mostradas separadamente; sua média em figuras é apenas descritiva.

O resumo exige todas as tarefas, confere hashes, IDs, métricas e limiares salvos e grava tabelas de cobertura, deltas pareados, controles *nuisance* e figuras em `results/evaluation/n2-secondary-v1/summary`. Não se deve interpretar baixa retenção ou melhor desempenho como evidência causal de autoria.

## Execução

Execute os comandos em ordem no cluster, após publicar e puxar o código. Os scripts são somente para Slurm e nunca editam N2 ou o release. O roteiro completo, incluindo testes, checks e retomada, também está na resposta de entrega desta implementação.

```bash
cd /home/victor.roquetto/Benchmark-Horus
git pull
PY="$HOME/miniconda3/envs/horus_theme/bin/python"
"$PY" -m pip install -e '.[dev,theme]'
"$PY" -m pytest tests/test_segment_views.py tests/test_secondary_controls.py tests/test_secondary_views.py tests/test_summarize_secondary_views.py -q
mkdir -p results/logs
for STAGE in segments layout; do
  sbatch --qos=normal --export=ALL,HORUS_STAGE="$STAGE" scripts/slurm_secondary_prepare.sh
done
```

Depois dos jobs, rode `segments_check` e `layout_check`. Então prepare `nuisance` e `secondary_plan`, e confira ambos. Submeta os seis CSVs com `scripts/submit_secondary_views.sh`. Quando todos os `metrics.json` e `predictions.parquet` estiverem completos, execute a array `scripts/slurm_secondary_summary.sh` com índices `0-8%2` e a etapa `secondary_merge`. O resumo só é produzido se toda a matriz estiver completa.
