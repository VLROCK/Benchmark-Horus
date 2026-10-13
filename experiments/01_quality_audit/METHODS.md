# Protocolo de auditoria de integridade e atalhos — v1

## Perguntas e escopo

1. Os registros e metadados do corpus completo estão estruturalmente íntegros?
2. Existem textos idênticos, equivalentes após uma comparação conservadora ou muito semelhantes que atravessam treino, validação e teste?
3. Como classe, domínio, fonte humana, gerador, gênero e comprimento estão distribuídos?
4. Quais sinais superficiais permitem distinguir as classes no corpus atual?

O protocolo é diagnóstico. Ele não define regras de inclusão, corrige textos, altera splits ou mede a eficácia final de detectores. O snapshot atual é provisório; as mesmas regras serão reaplicadas à próxima versão.

## Integridade e leakage

**Identidade e conteúdo.** Verificam-se IDs ausentes/repetidos, rótulo numérico versus textual, splits válidos, domínio, texto vazio, schema e contagens declaradas. O SHA-256 do conteúdo UTF-8 é recalculado e comparado ao `content_hash` armazenado. Uma segunda assinatura independente aplica NFKC, `casefold` e colapso de espaços *somente para comparação*. Essa assinatura pode aproximar textos distintos; cada colisão exige revisão antes de exclusão.

**Partições e benchmark.** Lê-se `record_id`, `content_hash`, `split` e o conteúdo dos três arquivos físicos, verificando a união com `all.parquet`, ausência de IDs sobrepostos, consistência do nome do split, igualdade do hash armazenado e igualdade de um SHA-256 recalculado diretamente do texto. Quando um benchmark materializado consta do catálogo, ele é verificado como subconjunto do corpus e seu conteúdo também é comparado por SHA-256 independente. Se não constar, a checagem específica é omitida e isso aparece no relatório. Testam-se ainda grupos que atravessam splits para `content_hash`, hash independente, `normalized_hash`, `duplicate_group_id`, `split_group_id`, `source_record_id` e URL humana. A tabela `cross_split_url_by_source.csv` permite localizar as fontes responsáveis. URL e ID de origem são alertas de parentesco documental, não duplicatas confirmadas: algumas URLs são páginas de coleção, enquanto outras identificam artigos individuais.

Quando o corpus final trouxer um ID de documento-pai ou de recorte, o nome real da coluna deve ser declarado em `lineage_fields` no YAML. O mesmo teste de agrupamento entre splits será aplicado sem mudar o algoritmo.

**Near-duplicates.** Textos com pelo menos 30 tokens lexicais são transformados em conjuntos de shingles de cinco palavras, após NFKC e `casefold`. Cada shingle recebe um hash BLAKE2b estável de 64 bits. Os 32 menores hashes por texto formam uma assinatura *bottom-k/KMV*. Um índice invertido encontra pares que compartilham ao menos seis impressões digitais; shingles muito frequentes são ignorados por limite configurável. A interseção de assinaturas é calculada imediatamente para cada par visitado, sem guardar milhões de pares fracos. Para os pares candidatos, calculam-se Jaccard e contenção **exatos sobre conjuntos de tuplas de palavras** (sem usar hashes na verificação final), retendo pares com Jaccard ≥ 0,80 ou contenção ≥ 0,90. A contenção é especialmente relevante para introduções e conclusões recortadas de documentos longos.

Essa triagem não tem recuperação garantida: documentos curtos, buckets frequentes, limite de candidatos e pares sem fingerprints compartilhados podem escapar. A contagem é de pares *encontrados*, não uma prova de ausência de outros pares. O script informa os limites efetivamente acionados. A identidade do documento-pai dos recortes acadêmicos não consta do schema atual; quando existir na próxima versão, deve ser usada como chave de agrupamento adicional antes de congelar splits.

`near_pair_review.csv` anexa fonte, URL e grupo de split de cada lado dos pares verificados. `cross_split_near_by_source.csv` resume os pares entre splits por fonte para orientar a revisão humana. URL idêntica ajuda a priorizar, mas URL diferente não invalida similaridade textual alta.

O uso de shingles e similaridade de conjuntos segue a tradição de deduplicação por aproximação de Broder; [FineWeb](https://papers.nips.cc/paper_files/paper/2024/file/370df50ccfdf8bde18f8f9c2d9151bda-Paper-Datasets_and_Benchmarks_Track.pdf) documenta a importância de comparar decisões de deduplicação. Nossa implementação usa bottom-k e verificação exata dos candidatos; não é a implementação de MinHash do FineWeb.

## Mapa de cobertura e atributos

As tabelas contam textos por classe × domínio × faixa de comprimento, fonte humana × domínio, gerador × domínio e gênero × classe × domínio. A tabela de suporte comum mostra quantos pares por célula poderiam ser formados sem modificar os textos; ela não seleciona registros. Cardinalidade e sobreposição dos campos brutos `theme`/`subtheme` são medidas separadamente, pois esses campos têm semânticas diferentes entre textos humanos e sintéticos. Metadados ausentes ou vazios são quantificados por classe. Comprimento é medido novamente pelo texto; divergências com `char_count` e `word_count` armazenados são relatadas.

As características medidas por registro são:

- comprimento em caracteres, palavras separadas por espaços e tokens lexicais;
- letras/caracteres por palavra;
- pontuação Unicode por 100 palavras e taxas separadas de vírgulas, pontos, dois-pontos, ponto e vírgula, perguntas, exclamações, aspas e parênteses;
- dígitos, quebras de linha, parágrafos e linhas não vazias;
- presença de bullets e títulos Markdown, cercas de código, tags HTML, URL, tabulação e caractere de substituição Unicode.

As taxas por 100 palavras reduzem a dependência mecânica do comprimento, mas não removem confusão por domínio, gênero ou fonte. Para atributos numéricos são relatados média, mediana, p10, p90, diferença padronizada das médias (SMD) e distância KS. Para flags binárias, relatam-se prevalências e diferença de prevalência. Essas medidas são descritivas, sem testes de significância inflados pelo tamanho da amostra.

## Sondas de separabilidade

Quatro regressões logísticas usam grupos de features: comprimento, pontuação, apresentação e todas as features superficiais. Uma quinta sonda usa TF-IDF de 4-gramas de caracteres em amostra determinística por classe e split, com teto definido no YAML. Esse teto mantém o experimento leve e deve aparecer no relatório. Nenhuma sonda recebe fonte, gerador, rótulo de tema ou metadados de proveniência como feature.

Todas as transformações estatísticas e o vocabulário TF-IDF são ajustados **somente no treino**. O limiar para balanced accuracy é escolhido na validação pelo índice de Youden; ROC-AUC e balanced accuracy são calculadas no teste. Intervalos de 95% usam bootstrap estratificado por rótulo das previsões de teste. Eles não incorporam variação entre treinos ou fontes e não são intervalos para generalização fora da distribuição.

Uma sonda alta indica separabilidade, não causalidade nem qualidade/defeito automático do benchmark. O TF-IDF pode explorar assunto e fonte além de formatação. A comparação por domínio e a futura classificação temática dos humanos serão necessárias para decompor esses fatores. [RAID](https://aclanthology.org/2024.acl-long.674/) motiva variar condições de geração e apresentação; [M4](https://aclanthology.org/2024.eacl-long.83/) motiva avaliar cobertura por domínio e gerador; [DetectRL](https://proceedings.neurips.cc/paper_files/paper/2024/file/b61bdf7e9f64c04ec75a26e781e2ad51-Paper-Datasets_and_Benchmarks_Track.pdf) analisa comprimento e cenários textuais realistas. As sondas aqui são adaptações diagnósticas do Hórus, não reproduções dos resultados desses trabalhos.

## Comparação com futura normalização

`horus-audit compare` exige o mesmo conjunto de IDs, rótulos, domínios e splits. Ele produz diferenças de gaps por atributo, proporção de textos modificados em cada medida e variação das sondas. Uma queda nas sondas pode ser desejável se vier da remoção de artefatos artificiais, mas também pode decorrer de perda de informação legítima. Cada transformação deve ser documentada e revista em amostras humanas e sintéticas, mantendo texto original e transformação vinculados pelo `record_id`.

## Limites

- Não é possível concluir ausência de contaminação no pré-treino de modelos externos com estes arquivos.
- Near-duplicates não recuperados pela triagem podem permanecer.
- URL compartilhada pode representar páginas relacionadas, não necessariamente o mesmo documento.
- `source_record_id` não equivale automaticamente a ID de documento-pai.
- Textos anteriores a 2022 reduzem risco de IA generativa moderna, sem garantir autoria exclusivamente humana.
- Resultados desta versão provisória não devem ser tratados como estimativas finais da versão de publicação.
