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
5. Há uma única amostra humana de 120 textos, sorteada previamente com 20 por
   área. O arquivo inicialmente chamado `pilot_annotation.csv` passa a ser a
   amostra final de auditoria; os quatro rótulos já registrados são mantidos.
   Os dois anotadores classificam independentemente os mesmos 120 textos,
   totalizando 240 decisões. Não há segunda amostra nem casos diagnósticos
   adicionais. A seleção é uniforme dentro da área, sem estratificação por
   fonte; logo, conclusões por fonte e por tema são exploratórias.
6. Os anotadores recebem somente `record_id`, área e texto, além de campos
   vazios para resposta e observação. Não recebem tema bruto, sugestão do
   modelo, confiança nem o papel amostral. Devem escolher um único tema
   predominante da área ou `incerto` / `fora_taxonomia`. Divergências são
   relatadas; não há adjudicação obrigatória nem rótulo de consenso inventado.
7. O limiar de probabilidade é **0,95**, fixado antes da comparação com as
   anotações humanas. Nenhum dos 120 rótulos escolhe ou ajusta esse limiar.
   Todas as 120 linhas servem para avaliar concordância e precisão/cobertura
   observadas. Softmax não é calibrado: 0,95 não garante 95% nem 90% de
   precisão humana. Intervalos de Wilson acompanham as estimativas.
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

### Etapas computacionais do protocolo de 120 textos

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
impresso. O mapa manual já foi preenchido. Congele a taxonomia e o mapa antes
de comparar previsões com respostas humanas. Nenhuma das 120 respostas deve
ser usada para escolher o limiar ou revisar o mapa e depois ser chamada de
avaliação independente.

Para **esta rodada**, use o `pilot_annotation.csv` já existente com quatro
respostas. O atalho abaixo escolhe a folha com mais respostas e a retoma;
**não execute `horus-theme ... pilot` novamente** para continuar a anotação.
O comando `pilot` serve apenas para uma reprodução nova, com outros arquivos
de saída.

Em Windows, o atalho abaixo abre a interface local e **retoma** a folha com
mais respostas já salvas. Ele não sorteia outro piloto e não exige carregar o
corpus. A página apresenta um texto por vez e as opções clicáveis da área:

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

Os dois anotadores classificam os mesmos 120 textos. Para gerar a folha cega
do segundo humano a partir da folha existente, sem copiar as respostas e sem
sobrescrever um arquivo já criado:

```powershell
python scripts/prepare_second_theme_sheet.py --pilot
```

O segundo anotador recebe apenas o `second_annotation.csv` produzido, por
canal privado. Ele pode fazer `git pull --ff-only` no próprio computador e
abrir a folha com Python 3.10 ou superior, sem dataset nem PyTorch:

```powershell
python scripts/annotate_theme_local.py --sheet "C:\caminho\second_annotation.csv"
```

Não coloque planilhas com textos humanos no Git público. As respostas ficam
salvas em cada CSV. A pessoa devolve o arquivo preenchido pelo mesmo canal.
Ao final, transfira ambas as folhas preenchidas para o cluster e preserve o
`pilot_annotation.csv` e seu `run.json` originais para conferir IDs e hashes.

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
```

No cluster, `scripts/slurm_theme_train_predict.sh` executa apenas `train` e
`predict`; não gera uma nova amostra. Depois que as duas folhas de 120 linhas
estiverem completas e transferidas, rode `evaluate-fixed` com o piloto original
do cluster e os dois CSVs anotados. O comando grava concordância bruta, κ com
correção de acaso dentro de cada área, intervalo bootstrap estratificado por
área, precisão/cobertura observadas e o `threshold.json` fixo. O TF-IDF é
opcional e entra apenas como referência diagnóstica.

Se `theme_cluster.yaml` foi criado antes desta mudança, atualize nele
`sampling.protocol: fixed_120` e `threshold.policy: fixed` com
`threshold.fixed_probability: 0.95`, preservando o SHA já fixado em
`model.revision`. O job Slurm verifica esses valores antes do treino.

```powershell
$pilot = 'results/theme_validation/<rodada-pilot>/pilot_annotation.csv'
$threshold = horus-theme --config $cfg evaluate-fixed --predictions $predictions --pilot $pilot --primary 'caminho/primary_annotation.csv' --second 'caminho/second_annotation.csv' --baseline-predictions $baselinePredictions
```

O limiar de 0,95 é definido na configuração **antes** de ler as respostas.
Nenhum resultado nesta amostra autoriza anunciar precisão mínima de 90% no
corpus. Compare também a concordância do modelo com cada anotador e relate os
casos de `incerto` e `fora_taxonomia`.

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

- `human_evaluation_rows.csv` contém as 120 comparações, sem o texto integral.
  A amostra é balanceada por área, não por fonte. As métricas globais resumem
  a média das seis áreas por construção, não uma estimativa direta ponderada
  pela frequência das áreas no corpus.
- Cohen's κ usa correção de acaso **dentro de cada área conhecida**, pois os
  anotadores só escolhem entre temas daquela área. O relatório também inclui
  o κ global convencional para comparação, embora a área compartilhada eleve
  o acordo ao acaso nesse cálculo. Há acordo bruto e intervalo bootstrap
  estratificado por área. Com 20 textos por área
  e 94 temas possíveis no total, resultados por área e tema são instáveis.
- Os 120 textos avaliam a transferência sintético → humano e a consistência
  das anotações de forma exploratória. Eles não calibram o limiar. O valor
  fixo de 0,95 é uma regra operacional, não uma probabilidade calibrada nem
  garantia de precisão. Se apenas poucos textos forem aceitos, o intervalo
  de precisão será amplo.
- O mapeamento manual e o classificador são avaliados separadamente e juntos.
  O primeiro anotador é a referência principal; a comparação com o segundo e
  com o subconjunto de concordância é diagnóstica, sem adjudicação automática.
- Uma taxa alta de `incerto` limita o uso das tabelas para afirmar equilíbrio
  temático. Temas previstos não devem ser usados para escolher o benchmark
  depois de conhecer os resultados dos detectores sem registrar essa decisão.
- A taxonomia pode não descrever bem todos os humanos. A proporção de
  `fora_taxonomia`, erros por fonte, área e comprimento devem constar da
  interpretação antes de publicar a versão derivada.
