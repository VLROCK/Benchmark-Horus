# Inventário de dados

Inspeção realizada em 8 de outubro de 2026. Os arquivos existentes não foram alterados.

## Versões encontradas

Não há um corpus bruto materializado na raiz atual. O arquivo `horus_release_candidate/audit/source_inventory.csv` enumera os CSVs de origem usados pelo pipeline anterior, mas esses caminhos não existem neste workspace. O que está disponível é:

1. `themes_full_dataset.csv`: taxonomia usada na geração sintética.
2. `horus_release_candidate/data/all.parquet`: corpus canônico processado e deduplicado.
3. `train.parquet`, `validation.parquet` e `test.parquet`: partições derivadas do corpus canônico.
4. `benchmark_balanced.parquet`: subconjunto experimental pareado por domínio e faixa de comprimento.
5. `horus_release_candidate/audit/`: artefatos derivados de uma auditoria anterior, incluindo dois Parquets intermediários. Eles não são entradas registradas no catálogo atual.

O release candidate deve ser tratado como snapshot importado e imutável. `data/raw/`, `data/processed/` e `data/benchmarks/` recebem somente novas versões explicitamente materializadas e versionadas.

## Formatos e escala

| Recurso | Formato | Linhas | Colunas | Tamanho |
| --- | ---: | ---: | ---: | ---: |
| corpus | Parquet/ZSTD | 183.911 | 57 | 383.694.473 bytes |
| treino | Parquet/ZSTD | 147.252 | 57 | 306.993.371 bytes |
| validação | Parquet/ZSTD | 18.263 | 57 | 38.402.688 bytes |
| teste | Parquet/ZSTD | 18.396 | 57 | 38.792.800 bytes |
| benchmark balanceado | Parquet/ZSTD | 101.176 | 57 | 150.973.164 bytes |
| taxonomia | CSV UTF-8 | 665 | 4 | 40.120 bytes |

Os cinco Parquets têm schema idêntico, mas apenas um row group por arquivo. O loader deve, portanto, projetar somente as colunas necessárias e usar batches. Uma futura rematerialização poderá adotar row groups menores e particionamento físico, sempre como nova versão derivada.

## Rótulos e cobertura

- `label = 0`, `label_name = human`: 97.659 textos.
- `label = 1`, `label_name = synthetic`: 86.252 textos.
- Domínios canônicos: `academic`, `informational`, `journalistic`, `juridical`, `literary` e `social`.
- Splits: 147.252 treino, 18.263 validação e 18.396 teste.
- Os `record_id` são únicos; os três splits não se sobrepõem e sua união reproduz o corpus canônico.
- `benchmark_eligible` é verdadeiro para 182.380 registros e falso para 1.531.

## Taxonomia

`themes_full_dataset.csv` possui colunas `id`, `domain`, `theme` e `subtheme`, com 665 IDs únicos, 92 temas e 659 subtemas. A distribuição é:

| Domínio | Linhas | Temas | Subtemas |
| --- | ---: | ---: | ---: |
| academic | 163 | 19 | 163 |
| informational | 100 | 14 | 100 |
| journalistic | 100 | 20 | 100 |
| juridical | 100 | 13 | 100 |
| literary | 102 | 16 | 102 |
| social | 100 | 12 | 100 |

As 665 combinações sintéticas distintas de `broad_area`, `theme` e `subtheme` correspondem exatamente à taxonomia. Nos textos humanos, `theme` e `subtheme` vieram de fontes heterogêneas e não constituem a mesma taxonomia; não se deve inventar um mapeamento automático.

## Grupos de colunas do corpus

- Identidade e rótulo: `record_id`, `source_record_id`, `label`, `label_name`.
- Domínio e taxonomia: `broad_area`, `theme`, `subtheme`, `genre`.
- Proveniência: `source_name`, `source_url`, `source_file`, `license`, `license_url`, `creation_date`.
- Geração: `generator`, `model_display_name`, `provider`, `persona`, `audience`, `objective`, `temperature`, `top_p`, `repetition_penalty`, `max_tokens`, `prompt_version`.
- Comprimento e idioma: `size_category`, `char_count`, `word_count`, `language`, `pt_confidence`, `pt_variant_evidence`, `code_mixing_flag`, `language_method`.
- Rastreabilidade e splits: `content_hash`, `normalized_hash`, `duplicate_group_id`, `split_group_id`, `split`, `benchmark_eligible`.
- Validações e artefatos: `language_valid`, `prompt_length_valid`, `artifact_any` e flags `artifact_*` específicas.
- Conteúdo: `content`.

O schema físico completo está em `horus_rc_schema.yaml`. Campos de geração podem ser nulos ou vazios em textos humanos; campos de licença e subtema também não têm cobertura completa. As flags de auditoria são metadados derivados, não verdade fundamental nem features implicitamente autorizadas.

## Limitações registradas

- O corpus é provisório e pode receber novas gerações.
- A documentação do release registra pendências de redistribuição; os dados grandes permanecem fora do Git.
- A identificação de português não comprova autoria brasileira.
- O benchmark balanceado é uma view experimental, não substitui o corpus completo.
- Resultados existentes em `horus_release_candidate/audit/` são evidências exploratórias herdadas e não são tratados por esta infraestrutura como conclusões definitivas.
