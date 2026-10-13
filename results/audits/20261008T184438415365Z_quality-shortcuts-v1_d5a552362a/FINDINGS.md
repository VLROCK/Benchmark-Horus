# Leitura crítica — snapshot provisório v0

Esta é uma auditoria exploratória do **corpus completo de 183.911 textos**, não do subconjunto balanceado como população de inferência. Os dados originais não foram modificados. Consulte [métodos](../../../experiments/01_quality_audit/METHODS.md), [checagens](tables/integrity_checks.csv), [proveniência dos arquivos](input_files.json) e [configuração resolvida](config.resolved.yaml) antes de citar números.

## 1. Integridade e possíveis vazamentos

- **Partições físicas:** IDs únicos, união exata dos três splits com `all.parquet`, sem sobreposição de IDs; o benchmark balanceado é subconjunto válido. O conteúdo real dos registros correspondentes também coincide por SHA-256 recalculado, não apenas pelo hash armazenado.
- **Hash armazenado:** 424 divergências, todas nos 5.000 registros humanos informacionais da fonte `Corpus Carolina - Wikipédia`. A causa não é identificável a partir destes arquivos; investigar a etapa que produziu `content_hash` antes de congelar a versão final. Ver [discrepâncias por fonte](tables/metadata_discrepancies_by_source.csv).
- **Contagem declarada de palavras:** 123.845 divergências frente à contagem independente por `str.split()`. Isso pode refletir tokenizações diferentes e não prova corrupção; documentar a regra de contagem e recalcular os campos derivados de forma consistente. A contagem de caracteres confere em todos os registros.
- **Duplicatas:** zero grupos com igualdade literal do conteúdo atravessam splits; zero grupos atravessam splits depois de NFKC, `casefold` e colapso de espaços. Há 37 grupos de igualdade sob essa comparação conservadora (74 textos), todos confinados ao mesmo split.
- **Near-duplicates:** a triagem encontrou 4.883 pares com Jaccard de 5-gramas ≥ 0,80 ou contenção ≥ 0,90. São todos humano–humano; **191 pares atravessam splits**, sendo 132 jurídicos, 30 jornalísticos, 13 informacionais, 12 sociais e 4 acadêmicos. Nenhum dos 191 compartilha `split_group_id`; 18 compartilham URL. [Pares para revisão](tables/near_pair_review.csv) e [resumo por fonte](tables/cross_split_near_by_source.csv). A recomputação com shingles literais confirmou os 4.883 pares salvos, com erro numérico máximo de 1,1e-16.
- **URLs:** 2.261 grupos de URL humana aparecem em mais de um split. Algumas URLs são de artigos/documentos, outras são páginas de coleção; a igualdade de URL, isoladamente, não demonstra vazamento. A triagem por fonte está em [URLs entre splits](tables/cross_split_url_by_source.csv).

**Limite de recuperação:** 5.844 textos com menos de 30 tokens não foram indexados pela busca aproximada; 399 fingerprints frequentes foram ignorados. O limite de pares candidatos não foi atingido, mas o método não tem garantia de recuperar todos os near-duplicates. Não há ID de documento-pai para ligar, por exemplo, introduções e conclusões acadêmicas originadas do mesmo artigo. Portanto, não se pode declarar os splits livres de leakage documental.

## 2. Cobertura e fatores de confusão

- Os seis domínios existem nas duas classes, mas as proporções diferem: social tem 28.100 humanos versus 14.376 sintéticos; literário tem 8.049 humanos versus 14.374 sintéticos. [Cobertura por domínio](tables/label_domain.csv) e [figura](figures/01_domain_coverage.png).
- O suporte comum para pareamento por **domínio × faixa de palavras** é limitado. Na faixa acadêmica de 5.000+ palavras há 3.195 humanos e nenhum sintético; não se deve extrapolar avaliação pareada a essa região. A soma dos mínimos por célula permite, no máximo, 114.556 linhas balanceadas sob *somente esses dois critérios*; outros controles podem reduzir esse número. [Tabela](tables/domain_length_common_support.csv) e [mapa](figures/06_domain_length_common_support.png).
- `genre` está ausente em todos os humanos e presente nos sintéticos. Os valores brutos de `theme` praticamente não se sobrepõem entre classes porque a anotação tem origens e semânticas diferentes; isso **não é** uma comparação temática confiável. A classificação dos humanos nas subáreas deve ser um trabalho posterior, como planejado. [Completude](tables/metadata_completeness.csv) e [sobreposição bruta](tables/raw_theme_overlap.csv).
- Há boa distribuição dos geradores pelos domínios, mas fontes humanas e geradores são estruturalmente distintos por classe. Metadados de proveniência não entraram nas sondas de texto. [Tabela gerador × domínio](tables/synthetic_generator_domain.csv) e [figura](figures/05_generator_domain_coverage.png).

## 3. Sinais superficiais e sondas

No conjunto inteiro, a mediana é de 398 palavras para humanos e 385 para sintéticos; essa semelhança central esconde caudas e diferenças fortes por domínio. A mediana de parágrafos é 1 versus 7, e a de quebras de linha por 100 palavras é 0,94 versus 2,81. A marca de bullet Markdown ocorre em 4,91% dos humanos e 18,00% dos sintéticos; no domínio informacional, 0,90% versus 74,74%. Em jurídico, a mediana de dígitos por 100 palavras é 15,38 versus 0,58. [Gaps completos](tables/shortcut_gaps.csv), [gráfico numérico](figures/03_numeric_feature_gaps.png) e [marcas binárias](figures/04_presentation_gaps.png).

As regressões logísticas diagnósticas obtiveram ROC-AUC de **0,747** apenas com comprimento, **0,857** com pontuação, **0,792** com apresentação e **0,911** com todas as features superficiais. Uma sonda TF-IDF de 4-gramas de caracteres, treinada numa amostra de 10.000 e testada em 3.000 textos, obteve **0,998**; ela pode capturar assunto e fonte, não somente artefatos. [Resultados e intervalos bootstrap](tables/shortcut_probes.csv), [gráfico](figures/07_shortcut_probe_auc.png) e [4-gramas de maior peso](tables/top_char4grams.csv).

Esses números mostram separabilidade **neste snapshot e split**, não efeito causal de cada atributo, robustez do benchmark ou capacidade de generalização de detectores. Os intervalos bootstrap condicionam-se ao treino e teste atuais; não incluem variação por fonte, gerador ou splits alternativos. A altíssima separabilidade dos 4-gramas merece controles por domínio, fonte, assunto e comprimento antes de interpretação substantiva.

## Prioridades antes de definir a versão final

1. Rever os 191 pares entre splits, começando pelos 132 jurídicos, e estabelecer agrupamento por documento/obra/artigo-pai antes de refazer splits. Registrar decisão por par ou componente, sem excluir automaticamente textos legítimos.
2. Recuperar ou criar identificadores de linhagem para recortes acadêmicos e outras obras longas. Preencher `lineage_fields` no YAML e repetir o teste. Sem essa chave, a checagem de leakage permanece incompleta.
3. Investigar os 424 hashes divergentes e explicitar a definição de `word_count`; corrigir apenas em uma nova versão derivada, preservando a original.
4. Pré-registrar regras de seleção/normalização por domínio e faixa de comprimento com suporte em ambas as classes. Priorizar formatação artificial e preservar características genuínas de gênero; inspecionar manualmente amostras humanas e sintéticas antes/depois.
5. Auditar a nova versão completa com a mesma configuração versionada. Para medir o **efeito da normalização**, auditar original e normalizado da **mesma versão e dos mesmos IDs** e usar `horus-audit compare`. Comparar este snapshot com um corpus futuro de composição diferente não é uma comparação pareada de normalização.

Não houve normalização nem ataque adversarial nesta etapa. Ambos exigirão protocolos próprios e preservação explícita da procedência/rótulo; textos editados por humanos não devem ser misturados sem marcação aos integralmente sintéticos.
