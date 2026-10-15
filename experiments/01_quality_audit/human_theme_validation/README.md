# Validação e uniformização de `theme` — protocolo v1

## Finalidade e população

Este experimento atribui **somente `theme`** aos textos humanos do corpus
`dataset_final_completo`. `subtheme` não é alvo da classificação; nos sintéticos
é metadado de geração e em parte dos humanos informacionais vem da fonte.
O `broad_area` existente é considerado conhecido nesta etapa. São treinados seis
classificadores BERTimbau independentes, um para cada área, e cada classificador
só pode escolher entre os temas da sua área em `themes_full_dataset.csv`.

Os temas sintéticos são a intenção do prompt, não uma anotação humana de
aderência. Os temas humanos originais foram obtidos de fontes diferentes e não
compartilham necessariamente a mesma semântica. A comparação de distribuições
usa `theme_canonical`, com procedência explícita, e não interpreta a previsão
do classificador como verdade fundamental.

## Decisões pré-definidas

1. O corpus de entrada e seus três splits são recursos imutáveis no catálogo
   `configs/datasets/horus_full.yaml`. O hash de cada entrada é registrado nas
   rodadas. Os Parquets de origem nunca são regravados.
2. Temas humanos já canônicos são aceitos por igualdade de nome dentro da área.
   Correspondências semânticas revisadas estão em `configs/manual_theme_map.csv`.
   Esse CSV é um mapa de **conversões diretas**, não uma lista exaustiva de
   valores originais: uma linha nele atribui o mesmo tema canônico a **todos**
   os textos humanos com aquele valor bruto. O mapa atual usa os valores
   realmente observados no corpus completo.
   Categorias que agrupam múltiplos temas, editorias genéricas, gêneros e
   títulos permanecem sem equivalência manual.
3. O treino usa sintéticos do split `train` de **todos os geradores**, incluindo
   `claude_sonnet_5`. A escolha da melhor época usa sintéticos de `validation`
   de todos os geradores. O teste sintético não participa do ajuste. A opção
   `heldout_generator` pode ser usada numa futura análise separada de mudança
   de gerador; seu valor nesta configuração é `null`.
   Um TF-IDF linear por área usa o mesmo recorte como referência diagnóstica;
   suas previsões não entram no corpus derivado.
   [BERTimbau Base](https://huggingface.co/neuralmind/bert-base-portuguese-cased)
   é um BERT pré-treinado em português brasileiro, escolhido como ponto de
   partida, não como melhor modelo demonstrado para este corpus.
   A configuração aceita outro checkpoint compatível para comparação controlada.
   O valor inicial `main` facilita o primeiro uso; fixe um commit do modelo
   antes de uma rodada destinada à publicação.
4. A entrada do modelo é apenas o conteúdo textual. `source_name`, `generator`,
   `theme` original humano e demais metadados não são features. A tokenização
   trunca ao início de 512 tokens; isto é uma limitação para textos longos e
   deve ser verificado no erro por faixa de comprimento.
5. O piloto contém 20 humanos por área (120). A amostra principal contém 100
   humanos aleatórios por área (600), com alocação proporcional entre fontes
   humanas dentro de cada área e pelo menos um texto por fonte. Se houver
   mais fontes que vagas em uma área, a amostragem pede aumento da cota em
   configuração. O piloto é excluído. Metade é fixada como
   calibração e metade como avaliação. Outros 200 casos diagnósticos priorizam
   baixa confiança e temas previstos raros; não entram na estimativa de
   distribuição do corpus. O segundo anotador, se disponível, recebe 20 dos
   100 textos aleatórios por área (120 em comum), divididos entre calibração
   e avaliação. Essa cota permite estimar a concordância global, com incerteza
   relevante; resultados por área e por tema são apenas diagnósticos.
6. Os anotadores recebem somente `record_id`, área e texto, além de campos
   vazios para resposta e observação. Não recebem tema bruto, sugestão do
   modelo, confiança nem o papel amostral. Devem escolher um único tema
   predominante da área ou `incerto` / `fora_taxonomia`. As divergências entre
   dois humanos podem ser adjudicadas em um terceiro CSV.
7. O limiar é único para os seis modelos. Ele é escolhido apenas nos casos
   **sem mapeamento direto** da metade de calibração. Seleciona a maior
   cobertura cuja cota inferior de Wilson de 95% para precisão alcança 0,90,
   com pelo menos 30 previsões aceitas. Caso nenhuma opção satisfaça a regra,
   o modelo se abstém de todos os casos; mapeamentos explícitos continuam.
   As probabilidades softmax não são tratadas como calibradas. A avaliação
   independente informa precisão e cobertura por área; o limiar global não
   garante precisão por área ou por tema.
8. O corpus derivado preserva o valor original em `theme_original`; a coluna
   `theme` passa a conter o tema canônico ou `incerto`. São acrescentados
   `theme_canonical`, `theme_assignment_method`, `theme_model_prediction`,
   `theme_model_top_probability` e `theme_model_run_id`. A probabilidade é a do
   **tema previsto pelo modelo**,
   mesmo quando um mapeamento manual tem prioridade; não é uma confiança no
   mapeamento.

## Execução em PowerShell — pelo pesquisador

### Levantamento dos temas humanos existentes

[`human_raw_theme_inventory.csv`](human_raw_theme_inventory.csv) lista **todos**
os valores distintos de `theme` humanos no `all.parquet` de
`dataset_final_completo`, com frequência e fontes. A leitura foi limitada às
colunas `label_name`, `broad_area`, `theme` e `source_name`; nenhum texto consta
do inventário. A taxonomia de destino está em `themes_full_dataset.csv` na raiz.

| Área | Textos humanos | `theme` distintos não vazios | `theme` preenchido | `subtheme` preenchido | Mapeamento manual direto |
| --- | ---: | ---: | ---: | ---: | ---: |
| Acadêmica | 13.814 | 9 | 13.814 | 0 | 2.583 |
| Informacional | 10.341 | 11 | 5.000 | 5.000 | 0 |
| Jornalística | 21.357 | 256 | 14.912 | 0 | 5.785 |
| Jurídica | 15.998 | 0 | 0 | 0 | 0 |
| Literária | 8.049 | 2.239 | 7.040 | 0 | 723 |
| Social | 28.100 | 12 | 28.100 | 0 | 587 |

Uma checagem adicional de **todos os campos de metadados humanos** confirmou
que `subtheme` está vazio nas áreas acadêmica, jornalística, jurídica,
literária e social. Na área informacional ele existe apenas nas 5.000 linhas
da Wikipédia Carolina, com 11 categorias em inglês (`sports`, `science` etc.)
que correspondem às mesmas 11 categorias gerais já presentes em `theme`.
`genre`, `persona`, `audience` e `objective` não contêm classes temáticas
humanas. `source_name` e `source_url` identificam origem, não a especialidade
de cada texto. Assim, neste corpus, os 13.814 textos acadêmicos têm somente as
nove grandes categorias do SciELO; rótulos como Medicina ou Química não estão
preservados como metadado separado. Recuperá-los exigiria consultar outra
fonte versionada ou classificar o conteúdo, com a devida validação.
O `source_file` aponta para `data/cleaned_human/dataset_human_clean.csv`, que
não está neste workspace; por isso não é possível concluir aqui se uma
classificação mais fina existia antes da construção do Parquet.

Na área literária, `Poesia` e `Romance` já coincidem exatamente com a taxonomia
sintética (81 e 288 linhas adicionais). Os 2.121 valores exclusivos da fonte Wattpad são em
sua maioria títulos identificados pelo prefixo `Ficção Wattpad -`; título não
é tema. Na área informacional, as categorias humanas são enciclopédicas e os
temas sintéticos são sobretudo guias práticos: a semelhança lexical não basta
para uma atribuição manual segura. Na área social, `Política` e `Artigo de Blog`
são frequentes, mas o primeiro atravessa vários temas sintéticos e o segundo
é gênero/formato. Na área jornalística, `mundo`, `cotidiano`, `tec` e `ciencia`
também atravessam temas; nomes de colunas e de veículos são procedência.

Os totais de mapeamento direto são apenas cobertura por metadado. Eles não
medem correção temática do texto; a amostra humana avaliará isso. Valores
ausentes do mapa ainda podem receber tema via classificador e limiar posterior.

#### Por que a área acadêmica tem nove valores e apenas dois no mapa direto?

| `theme` humano | Linhas | Decisão de conversão direta |
| --- | ---: | --- |
| `Biological Sciences` | 932 | `Biologia` |
| `Engineering` | 1.651 | `Engenharia` |
| `Health Sciences` | 2.629 | Classificar por texto; pode abranger Medicina e outras áreas de saúde. |
| `Human Sciences` | 2.468 | Classificar por texto; abrange várias disciplinas humanas. |
| `Applied Social Sciences` | 1.884 | Classificar por texto; pode abranger Economia e Ciência política, entre outras. |
| `Linguistics, Letters and Arts` | 1.844 | Classificar por texto; mistura Linguística, Artes e Letras. |
| `Exact and Earth Sciences` | 1.331 | Classificar por texto; mistura Matemática, Física, Química e ciências da Terra. |
| `Agricultural Sciences` | 1.051 | Classificar por texto; não há tema agrícola equivalente na taxonomia sintética. |
| `Others` | 24 | Classificar por texto ou `fora_taxonomia`. |

O mesmo critério vale nas outras áreas. Informacional tem 11 categorias
enciclopédicas que não equivalem aos 14 temas sintéticos de guias práticos.
Jornalística tem 256 valores, entre editorias temáticas, editorias genéricas,
nomes de colunas e veículos; 21 conversões são diretas. Literária tem 2.239
valores, com muitos títulos de obras e categorias de gênero; 21 conversões
são diretas, além de `Poesia` e `Romance` por igualdade. Social tem 12 valores:
seis conversões diretas, enquanto `Política`, `Artigo de Blog` e outros valores
abrangem vários temas ou representam formato. Jurídica não tem `theme` humano
preenchido. O inventário completo permite conferir cada valor e sua frequência.

### Etapas computacionais

Instale as dependências no ambiente desejado e rode os comandos **nesta ordem**.
Nenhum estágio é executado automaticamente pela implementação. `train` faz
seis ajustes finos BERT e requer recursos adequados; em CPU pode ser demorado.
O treino informa área, época, macro-F1 de validação e tempo por época no log
(`stderr`); o caminho da rodada permanece sozinho em `stdout` para captura
pelos comandos abaixo.

```powershell
python -m pip install -e ".[theme,dev]"
pytest -q tests/test_theme_validation.py
$cfg = 'experiments/01_quality_audit/human_theme_validation/configs/theme_v1.yaml'
$inventory = horus-theme --config $cfg inventory
```

Examine `human_theme_inventory.csv` e `taxonomy_themes.csv` no diretório
impresso. O mapa manual já foi preenchido a partir do levantamento acima;
revise-o à luz da anotação piloto. Depois de iniciar as previsões,
**congele esse mapa**: os estágios seguintes recusam artefatos feitos com
outra versão dele.

```powershell
$pilot = horus-theme --config $cfg pilot
```

Para anotar o piloto em uma interface local, com um texto por vez e opções
clicáveis da área correspondente:

```powershell
horus-theme --config $cfg annotate-web --sheet $pilot
```

Em Windows, o atalho abaixo **retoma** o piloto local com mais respostas já
salvas. Ele não sorteia outro piloto e não exige carregar o corpus:

```powershell
python scripts/annotate_theme_local.py --pilot
```

Se houver mais de um piloto, o caminho e o número de respostas da folha
selecionada aparecem no terminal. Para escolher uma folha específica, use
`--sheet "C:\caminho\pilot_annotation.csv"`.

A página abre em `http://127.0.0.1:8765/`. Cada resposta é salva no próprio
CSV com gravação atômica; ao reabrir, a interface retoma no primeiro texto
pendente. Só aparecem ID, área e texto, sem tema original, sugestão do modelo
ou papel da amostra. Para acesso manual à URL, use `--no-browser`; para outra
porta, use `--port 8766`. Encerre o servidor com Ctrl+C.

Preencha `theme_human` do piloto e refine o manual de anotação. O piloto não
entra na amostra principal. Se o mapa de temas mudar, confirme-o antes do treino.

Uma checagem opcional de aderência temática dos sintéticos sorteia até dois
textos por área × gerador. Em `synthetic_theme_review.csv`, preencha
`theme_supported` com `sim`, `não` ou `incerto`; esse arquivo é diagnóstico e
não altera automaticamente os rótulos de treino.

```powershell
$syntheticReview = horus-theme --config $cfg synthetic-review
```

```powershell
$baselineRun = horus-theme --config $cfg baseline
$modelRun = horus-theme --config $cfg train
$baselinePredictions = horus-theme --config $cfg baseline-predict --baseline-run $baselineRun
$predictions = horus-theme --config $cfg predict --model-run $modelRun
$manifest = horus-theme --config $cfg sample --predictions $predictions --pilot $pilot
$reviewDir = Split-Path $manifest -Parent
```

Distribua `primary_annotation.csv` e, se houver segunda pessoa,
`second_annotation.csv`, sem enviar `sample_manifest.csv`. Os dois anotadores
devem preencher apenas `theme_human` e `annotation_note`, mantendo os IDs e as
linhas. `incerto` significa que o texto não permite escolher com segurança;
`fora_taxonomia` significa que nenhum tema da área é adequado. O segundo
arquivo contém 120 textos que também constam do primeiro, 20 por área. Se houver
divergências, a etapa de avaliação grava `annotator_disagreements.csv`;
uma nova execução pode receber um CSV de adjudicação contendo apenas essas
linhas, com as colunas `record_id`, `broad_area` e `theme_human`.

Também é possível preencher as planilhas principais na interface, em sessões
separadas para preservar a independência dos anotadores:

```powershell
horus-theme --config $cfg annotate-web --sheet (Join-Path $reviewDir 'primary_annotation.csv')
horus-theme --config $cfg annotate-web --sheet (Join-Path $reviewDir 'second_annotation.csv')
```

O outro anotador pode baixar o repositório e receber **somente** o arquivo
`second_annotation.csv` por um canal privado. Não é necessário transferir o
dataset completo, instalar PyTorch nem receber `sample_manifest.csv`. Em seu
próprio computador, com Python 3.10 ou superior:

```powershell
git pull --ff-only
python scripts/annotate_theme_local.py --sheet "C:\caminho\second_annotation.csv"
```

O script abre `http://127.0.0.1:8765/` localmente e salva cada resposta no
próprio CSV; interromper e executar o mesmo comando novamente retoma no
primeiro texto pendente. O anotador devolve o CSV preenchido por canal privado.
Não coloque planilhas com textos humanos no Git público. O `sample` gera a
folha do segundo anotador automaticamente; não sorteie outra amostra para ele.
Antes do `sample` no cluster, sincronize `second_annotator_per_area: 20` no
`theme_cluster.yaml` local do cluster.

Sem segundo anotador:

```powershell
$threshold = horus-theme --config $cfg evaluate --predictions $predictions --baseline-predictions $baselinePredictions --sample-manifest $manifest --primary (Join-Path $reviewDir 'primary_annotation.csv')
```

Com segundo anotador, acrescente `--second (Join-Path $reviewDir
'second_annotation.csv')`. Após adjudicar as divergências, rode novamente com
`--adjudicated caminho_do_csv`; use o **último** `threshold.json` produzido.

```powershell
$derivedCorpus = horus-theme --config $cfg materialize --predictions $predictions --threshold $threshold
```

O último estágio grava `corpus_with_canonical_theme.parquet`,
`theme_assignments.parquet`, `theme_counts_full.csv`,
`theme_comparison_full.csv`, `theme_counts_by_method.csv` e
`theme_counts_eligible.csv` em uma nova pasta de rodada sob
`results/theme_validation/`. A tabela `eligible` usa apenas a flag
`benchmark_eligible`; **não é o benchmark balanceado congelado**, que ainda
não existe para este corpus completo. Cada rodada inclui configuração e
metadados de proveniência.

## Interpretação e limites

- A matriz de confusão e as linhas de avaliação usam a amostra humana
  reservada (`confusion_evaluation.csv` e `human_evaluation_rows.csv`).
  `evaluation_area_balanced` dá peso igual às seis áreas;
  `evaluation_corpus_weighted` usa o inverso da probabilidade de amostragem
  por área e fonte.
  O relatório compara também as previsões sem abstenção do BERTimbau e do
  TF-IDF na mesma metade humana reservada; essa comparação é diagnóstica.
  A amostra diagnóstica e o piloto ficam fora dessas métricas.
- Cohen's κ mede a concordância **entre humanos** nos 120 textos em comum.
  O relatório inclui intervalo bootstrap estratificado por área. Sem segunda
  pessoa, ele não é calculado. O κ global não valida cada tema;
  categorias raras podem nem aparecer na amostra.
- O mapeamento explícito e o classificador são avaliados juntos no conjunto
  independente, mas o limiar é escolhido só sobre previsões do modelo.
- Uma taxa alta de `incerto` limita o uso das tabelas para afirmar equilíbrio
  temático. Temas previstos não devem ser usados para escolher o benchmark
  depois de conhecer os resultados dos detectores sem registrar essa decisão.
- A taxonomia pode não descrever bem todos os humanos. A proporção de
  `fora_taxonomia`, erros por fonte, área e comprimento devem constar da
  interpretação antes de publicar a versão derivada.
