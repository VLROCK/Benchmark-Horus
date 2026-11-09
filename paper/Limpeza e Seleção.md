# Limpeza e Seleção — Hórus N2 IID v1

> Registro metodológico do corpus completo, das views de normalização, da seleção e do release IID. Atualizado em 2026-11-09. Os números abaixo se referem à versão `complete-with-claude-provisional-v1`; não devem ser transpostos para outra versão do corpus sem repetir a auditoria.

## 1. Escopo e decisões centrais

O corpus completo é a entrada versionada; a *view* normalizada é uma transformação derivada de **todas** as linhas, antes da seleção e sem usar os splits; o benchmark é o subconjunto selecionado. Os arquivos originais não são editados. Cada execução guarda configuração, hashes, rastros de transformação e resultados. O release principal atual é **`horus-n2-iid-v1`**, baseado no texto N2, com seleção `w0.25_s20261103` e três divisões IID. O armazenamento público no Hugging Face ainda não foi preparado.

As decisões tomadas durante a construção foram:

1. **Alta precisão nas alterações:** preservar texto potencialmente legítimo quando uma regra não consegue delimitar o artefato com confiança. Uma flag de revisão não significa exclusão automática.
2. **Limpeza de texto separada da seleção:** corrigir sinais técnicos e editoriais na normalização; controlar a distribuição de comprimento na seleção, sem cortar textos apenas para aproximar classes.
3. **Referências preservadas:** textos acadêmicos sintéticos também apresentam seções de fontes. Cabeçalhos de referências podem ser sinalizados, mas não autorizam remoção de bibliografia inteira em uma classe.
4. **Inglês contextual preservado:** palavras técnicas e trechos ingleses naturais permanecem; apenas abstracts paralelos extensos, delimitados e com evidência lexical suficiente podem ser retirados.
5. **Sem reescrita estilística:** não corrigir gramática, gíria, formulação hipotética ou estilo só porque um texto é sintético.
6. **N2 como texto principal:** a N3/N3s foi examinada como controle de apresentação. A seleção e o release congelado usam N2; a N3s permanece uma *view* comparativa.
7. **Temas humanos não usados para selecionar:** existe uma atribuição automática preliminar das subáreas humanas, porém ela não integrou a seleção. Sua validação e seu uso em análises de cobertura ficam separados deste release.

Agentes e inspeção de amostras foram usados para formular e revisar hipóteses de artefatos. A execução em escala usa regras determinísticas, parametrizadas e auditáveis; nenhum agente ou LLM julga, individualmente, qual operação aplicar em cada linha. Os resultados de inspeções exploratórias sem vereditos por item rastreáveis neste checkout não são apresentados aqui como validação humana formal.

## 2. Entrada e auditoria inicial

Entrada: `dataset_final_completo/data/all.parquet`, **195.154 textos** (**97.659 humanos**, **97.495 sintéticos**) e SHA-256 `fe4dc7a107b8127a4106fc5ea0e3e81e101f4e911a363a89f7e7af494197625c`. O [relatório da auditoria completa](../results/audits/20261013T180229080021Z_quality-shortcuts-full-v1_1727096142/REPORT.md) identificou **4.883 pares aproximados verificados**, dos quais **191** atravessavam splits herdados; **2.261 grupos de URL humana** também atravessavam esses splits. A igualdade exata de conteúdo e os grupos `split_group_id` então disponíveis não atravessavam splits. A auditoria era diagnóstica: não limpou nem selecionou documentos.

A busca aproximada não foi exaustiva: **5.988** textos curtos ficaram sem assinatura de busca, e fingerprints muito frequentes foram limitados. Não havia identificador de documento-pai para ligar com certeza introduções e conclusões retiradas do mesmo artigo. Portanto, os controles posteriores cobrem vínculos **observados**, sem provar ausência total de parentesco textual. Textos humanos anteriores a 2022 reduzem, mas não eliminam, o risco de conteúdo automatizado anterior ou não identificado.

## 3. N1: limpeza técnica conservadora

A configuração [N1](../configs/experiments/corpus_normalization_n1.yaml) aplica, conforme o conteúdo, Unicode NFC, reparo conservador de mojibake, normalização de quebras de linha, controles e espaços, conversão limitada de HTML editorial e entidades, e substituição de URLs/e-mails por `URL`/`EMAIL`. Trechos de código e exemplos HTML/CSS são protegidos das conversões pertinentes. A N1 **detecta**, mas não neutraliza, Markdown e placeholders; também não remove abstracts, bibliografia ou rodapés.

A primeira execução N1, `full-n1-114566`, marcou muitos falsos candidatos de encoding. A flag foi recalibrada para alta precisão, gerando a execução usada nas comparações, [`full-n1-115189`](../results/normalization/full-n1-115189/summary.json). Nela, **195.154 entradas produziram 195.154 saídas**, sem descarte. Havia **931 linhas com `status:review`** e **194.223 com `status:ok`**; as flags de revisão não eram critério geral de exclusão. Houve **238 operações de reparo de mojibake**, **285 conversões de `<br>`**, **50 conversões de parágrafo HTML**, **117 máscaras de URL** e **39 de e-mail**. As contagens de operações podem ser maiores que o número de documentos afetados.

Os contadores de palavras declarados na origem divergiam da contagem lexical recalculada em **131.558 linhas**. A view derivada recalcula contagens e hashes de conteúdo, preservando no rastro os valores originais e os identificadores. Não se inferiu que toda divergência era corrupção textual: uma limpeza anterior à montagem do corpus pode explicar parte dos metadados desatualizados.

## 4. Diagnóstico de hashes e política N2

O [diagnóstico de metadados](../results/normalization/diagnostics-115958/summary.json) encontrou **424 hashes de origem divergentes**; as variantes testadas não explicaram sua causa. A [checagem específica](../results/normalization/hash-check-116021/summary.json) obteve **424 hashes recalculados distintos**, sem pares de textos diferentes com o mesmo SHA-256 e sem duplicata exata envolvendo esses alvos. Os hashes derivados recalculados são usados; os 424 hashes da fonte não foram editados.

Na área acadêmica, foram observados **70 cabeçalhos de fontes em 13.814 humanos** e **19 em 16.249 sintéticos**; cabeçalhos explícitos de referências apareceram em **64 humanos** e **0 sintéticos**. Esses detectores de cabeçalho têm cobertura limitada e não medem toda citação/bibliografia. A evidência de que ambos os lados podem citar fontes, somada à necessidade de preservar o conteúdo, levou à decisão de **não remover bibliografias**.

A configuração [N2](../configs/experiments/corpus_normalization_n2.yaml) mantém as regras N1 e acrescenta somente:

- remoção de um abstract paralelo em inglês com cabeçalho explícito, evidência lexical conservadora, pelo menos **40 palavras**, um bloco delimitado e uma seção seguinte reconhecível;
- remoção de rodapés terminais estritos, como avisos editoriais claramente separáveis do corpo;
- uma guarda de perda: quando o original tem ao menos **50 palavras**, a proposta elimina **20 ou mais** e retém **menos de 90%**, a alteração é vetada e sinalizada.

Textos em inglês fora dessa estrutura e abstracts ambíguos permanecem. A [prévia estrita](../results/normalization/n2-preview-116020/summary.json) indicou **6** abstracts removíveis e **738** rodapés. A [execução N2](../results/normalization/full-n2-116035/summary.json) confirmou **6 operações de abstract** e **738 de rodapé**, manteve **195.154 linhas** e marcou **376 candidatos de referências terminais** sem apagá-los. Havia **230 flags de candidato a abstract em inglês**, mas a maioria não satisfazia a regra estrita de remoção. Os mesmos **931** registros continuaram com status geral de revisão. O [manifesto N2](../results/normalization/full-n2-116035/freeze_manifest.json) registra o corpus derivado com SHA-256 `846d9647f6f049b5b9012e4f06b573da9fe6986c1e0cbff45e5ab3364644419b`; esse congelamento da *view* ainda não era uma seleção ou split.

## 5. Comparações de limpeza: por que N2 foi mantida

As [sondas exploratórias raw → N1 → N2](../results/audits/normalization-view-probes-124508/probe_metrics.csv) usaram os mesmos IDs por granularidade, modelos separados por view e limiares aprendidos na validação. Nelas, o classificador documental apenas de apresentação teve AUC **0,8166** no bruto e **0,8168** na N2; o de 4-gramas de caracteres teve AUC **0,9982** em ambos. Esses resultados **não** são métricas finais do release, pois usaram splits exploratórios anteriores aos controles de linhagem. Mostram que N2 removeu artefatos estritos sem eliminar separabilidade de apresentação e léxico. Em especial, a normalização não foi projetada para igualar comprimentos ou estruturas documentais.

Foi então testada uma N3 simétrica que neutralizava Markdown/placeholders nas duas classes: [32.606 textos alterados](../results/normalization/full-n3-125850/summary.json). Como o artefato de apresentação a investigar era sobretudo sintético, foi criada a [N3s](../configs/experiments/corpus_normalization_n3s.yaml), na qual **somente as operações adicionais de apresentação** são específicas da classe sintética; o restante permanece condicionado ao conteúdo. A [verificação N2 × N3s](../results/normalization/full-n3s-126552/class_specific_verification.json) encontrou **0 de 97.659 humanos alterados** e **32.320 de 97.495 sintéticos alterados**. Marcadores de listas não ordenadas, ênfase, cercas de código, código inline, links Markdown e placeholders explícitos são tratados de forma conservadora; conteúdo de código deve ser preservado. Marcadores numerados foram preservados porque também podem ser seções legítimas. A regra para linhas com travessão pode afetar diálogo literário, motivo para inspeção e cautela.

Na **mesma seleção exploratória** `w0.25_s20261103`, a sonda de apresentação passou de AUC **0,8461 (N2)** para **0,8388 (N3s)**, redução pequena; a de pontuação subiu de **0,6495** para **0,6923**. A [tabela pareada de sondas](../results/selection/candidates-126604/paired_surface_probes.csv) não permite concluir que a N3s produziu um benchmark mais justo. A [tabela de balanço](../results/selection/candidates-126604/paired_balance_smd_ks.csv) registra marcadores de lista numerada ainda fortemente assimétricos e diferença de parágrafos praticamente inalterada. Por isso, a N3s é uma análise de sensibilidade, e a **N2 permaneceu a view principal congelada**.

## 6. Seleção a partir da N2

A [configuração da seleção](../configs/experiments/selection_candidate_n2_v1.yaml) e o [resumo dos cenários](../results/selection/candidates-126604/summary.json) registram a política:

1. Aplicar a N2 ao **corpus completo**, sem splits. Status geral `review` da N2 não elimina automaticamente uma linha. Colocar em espera apenas flags severas `EMPTY_AFTER`, `ENCODING_REMAINS`, `HTML_DENSE_REMAINS`, `SUBSTANTIAL_LOSS` ou `N2_GUARD_RETAINED`, além de conteúdo vazio/conflito entre rótulos. Permanecem diagnósticos: idioma inválido em metadados, hash de origem divergente, truncamento anterior e abstract candidato.
2. Retirar da elegibilidade cópias exatas redundantes do mesmo rótulo, preservando seus registros no manifesto. Resultado: **194.875 elegíveis**, **277 em espera para revisão** e **2 cópias exatas secundárias**; a soma recompõe as 195.154 linhas. Isso é a base elegível, **não** o tamanho do benchmark.
3. Construir grupos conservadores de linhagem com IDs disponíveis, URL de artigo não genérica, igualdade exata de texto e pares verificados com Jaccard de 5-gramas **≥ 0,90**. Pares encontrados apenas por contenção não são declarados mesmo documento; continuam pendentes, mas recebem a mesma partição de sonda/split para impedir vazamento **observado**.
4. Dentro de cada célula `área × floor(log2(1 + palavras_N2) / largura)`, selecionar o mesmo número de humanos e sintéticos, pela ordenação determinística SHA-256 de semente e ID. A escolha congela **largura 0,25** e **semente 20261103**. Não há corte ou reescrita por comprimento, nem pareamento de `theme` ou quotas artificiais de parágrafos.

Foram estudadas larguras **0,25**, **0,5** e **1,0** e três sementes de amostragem. A primeira reteve **107.566 linhas**, a segunda **110.492** e a terceira **120.742**; todos os cenários passaram os gates exploratórios de cobertura. A largura 0,25 foi escolhida por melhor controle local do comprimento, aceitando menor retenção. O cenário principal tem **53.783 por classe**, com a seguinte composição **por classe**:

| Área | Humanos | Sintéticos |
|---|---:|---:|
| Acadêmica | 4.943 | 4.943 |
| Informacional | 7.315 | 7.315 |
| Jornalística | 14.921 | 14.921 |
| Jurídica | 9.332 | 9.332 |
| Literária | 5.505 | 5.505 |
| Social | 11.767 | 11.767 |

O balanço selecionado da N2 tem SMD de `log_words` **0,00026** e KS **0,0115**. Permanecem diferenças substantivas: média de **3,27 parágrafos humanos** contra **9,20 sintéticos** (SMD **0,678**) e prevalência do detector de bullet Markdown de **1,62%** contra **16,49%** (SMD **0,537**). A diferença de parágrafos pode refletir a composição natural de áreas/gêneros e precisa ser medida por subgrupo; não foi “corrigida” apagando quebras. Esses números são descritivos, não inferem que toda diferença seja um atalho ilegítimo. A [tabela de balanço](../results/selection/candidates-126604/paired_balance_smd_ks.csv) contém as medidas por área.

## 7. Splits IID e release congelado

O controle preliminar de comprimento/parágrafos gerou o protocolo IID v2. O [refinamento v3](../configs/experiments/grouped_iid_source_balance_v3.yaml) manteve a mesma seleção e adicionou **balanceamento suave de proveniência** na alocação: fonte humana ou gerador sintético com pelo menos **100 registros** na combinação área/classe. Não impõe quotas e não altera textos. O objetivo considera área, classe e estratos largos de palavras (`0, 100, 250, 500, 1.000, 2.000, 5.000`), respeitando `probe_group_id` indivisível.

Para cada uma das sementes **20261105, 20261106, 20261107**, há **15% de teste reservado** (**16.137 linhas**) e **cinco folds** do desenvolvimento (aproximadamente **18.285–18.287 linhas** cada). O teste não entra na validação cruzada. Todos os **2.232 pares aproximados observados com ambos os membros selecionados** ficaram na mesma partição/fold; **0** atravessaram essas divisões. No v3, a maior diferença de participação de proveniência material foi **0,13–0,34 ponto percentual** no teste e **1,40–1,42 ponto** nos folds; no v2 as diferenças do teste eram próximas de **9,94–12,58 pontos**. O maior desvio dos estratos de comprimento com pelo menos 100 registros foi **0,27 ponto** no teste e **0,41** nos folds. Estes são diagnósticos de composição, não prova de ausência de toda confusão fonte/classe.

O maior `probe_group_id` selecionado contém **610 textos humanos jurídicos distintos** da mesma fonte; o agrupamento é conservador para impedir separação de pares observados, mas também pode superagrupar documentos com fórmulas jurídicas recorrentes. Seu tamanho e composição estão no [diagnóstico v3](../results/protocols/grouped-iid-source-v3-127146/summary.json) e devem ser considerados ao estimar incerteza por grupo.

O [release `horus-n2-iid-v1`](../results/releases/horus-n2-iid-v1/release_manifest.json) registra `verified: true`, IDs selecionados, configuração, SHA-256 dos insumos, três splits e checks de pares. Os textos permanecem no corpus N2; os arquivos de split referenciam IDs e metadados, sem copiar o conteúdo. Os Parquets grandes são ignorados no Git neste checkout; a verificação integral dos bytes requer os arquivos materializados no cluster. O congelamento **não treinou detectores**, **não fez LOGO/LODO/LOSO** e **não calculou desempenho final**.

## 8. Limitações que devem acompanhar o artigo

- A busca de near-duplicates é aproximada e há pares/linhagens desconhecidos. O zero de vazamento refere-se aos **pares observados** e aos grupos construídos.
- Algumas áreas humanas têm uma única fonte: acadêmica usa `scielo.br` e jurídica usa `Corpus Carolina - Juridical` na seleção observada. Balancear fontes dentro da área não remove o vínculo entre fonte e classe. LOSO nessas áreas não é identificável sem nova coleta.
- Aliases de fontes (por exemplo, host com/sem `www`) não foram unificados no v3. Uma futura taxonomia de fontes precisa ser registrada como metadado derivado, sem alterar este release.
- `theme` tem origens diferentes por classe; não se deve interpretar seus valores crus como pareamento temático. Uma [atribuição automática preliminar dos temas humanos](../results/theme_llm_full/20261030T061341704181Z_human-theme-llm-full-v1-materialize-full_d1c87107b5/theme_full_summary.json) já foi materializada, mas contém casos incertos e não constitui acurácia temática verificada; não foi usada nesta seleção. A validação e a análise temática controlada permanecem pendentes.
- A view N2 deixou Markdown/placeholders e estruturas naturais ou potencialmente artificiais. O efeito dessas pistas deve ser medido por sondas e controles na avaliação final, sem prometer sua eliminação por limpeza.
- Alta acurácia de uma sonda lexical ou superficial mostra separabilidade; sozinha não identifica causalmente o artefato, nem valida ou invalida o benchmark. Efeitos de normalização, seleção, fonte e domínio devem ser relatados separadamente.

## 9. Trilhas de evidência e reprodução

| Etapa | Configuração/implementação | Resultado principal |
|---|---|---|
| Auditoria bruta | [`quality-shortcuts-full-v1`](../results/audits/20261013T180229080021Z_quality-shortcuts-full-v1_1727096142/REPORT.md) | Integridade, cobertura e pares aproximados |
| N1 | [configuração](../configs/experiments/corpus_normalization_n1.yaml), [implementação](../src/horus_benchmark/data/normalization.py) | [resumo recalibrado](../results/normalization/full-n1-115189/summary.json) |
| Diagnósticos | [metadados](../results/normalization/diagnostics-115958/summary.json) | [checagem dos 424 hashes](../results/normalization/hash-check-116021/summary.json) |
| N2 | [configuração](../configs/experiments/corpus_normalization_n2.yaml) | [resumo](../results/normalization/full-n2-116035/summary.json), [manifesto](../results/normalization/full-n2-116035/freeze_manifest.json) |
| N3s comparativa | [configuração](../configs/experiments/corpus_normalization_n3s.yaml) | [resumo](../results/normalization/full-n3s-126552/summary.json), [verificação por classe](../results/normalization/full-n3s-126552/class_specific_verification.json) |
| Seleção | [configuração](../configs/experiments/selection_candidate_n2_v1.yaml), [implementação](../src/horus_benchmark/audit/selection_candidate.py) | [cenários](../results/selection/candidates-126604/summary.json), [balanço](../results/selection/candidates-126604/paired_balance_smd_ks.csv) |
| Splits e release | [configuração v3](../configs/experiments/grouped_iid_source_balance_v3.yaml), [freeze](../configs/experiments/freeze_grouped_iid_n2_v1.yaml) | [diagnóstico v3](../results/protocols/grouped-iid-source-v3-127146/summary.json), [manifesto final](../results/releases/horus-n2-iid-v1/release_manifest.json) |

O próximo estágio é definir e aprovar, antes de executá-los, sondas de atalhos, baselines, protocolos IID/OOD, limiares e intervalos. Resultados exploratórios deste documento não substituem esse teste congelado.

## 10. Relação com a literatura

O desenho usa controles de composição, proveniência e shifts de gerador/domínio porque benchmarks como [RAID](https://aclanthology.org/2024.acl-long.674/) e [M4](https://aclanthology.org/2024.eacl-long.83/) mostram a importância de variar essas condições na avaliação de detectores. A normalização conservadora e as sondas de atalhos são **decisões específicas do Hórus**, documentadas acima, e não devem ser atribuídas a esses trabalhos como regras deles. O release IID atual constitui uma base auditada para futuros testes OOD; ainda não demonstra generalização de detector algum.
