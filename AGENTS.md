# AGENTS.md — Projeto Hórus Benchmark

## 1. Contexto do projeto

O **Projeto Hórus** é uma Iniciação Científica desenvolvida na **UNICAMP / RECOD.AI**, sob orientação do Prof. Anderson Rocha, com apoio da FAPESP.

A pesquisa investiga a **detecção de textos sintéticos produzidos por Large Language Models (LLMs)**, com foco no português brasileiro (PT-BR), na qualidade dos dados e na capacidade de generalização dos detectores.

O objetivo atual é consolidar e publicar um **benchmark robusto, auditado e reproduzível** para avaliar métodos de detecção de textos humanos versus textos gerados por IA.

### Dataset

O corpus contém aproximadamente 184 mil textos, distribuídos entre duas classes:

- **Human:** textos autênticos, coletados de fontes anteriores a 2022 para minimizar o risco de contaminação por IA generativa moderna.
- **Synthetic:** textos gerados por diferentes LLMs, proprietários e open-source, por meio de pipelines automatizados.

Os textos abrangem **seis áreas**:

1. Jornalística
2. Acadêmica
3. Social
4. Jurídica
5. Informacional
6. Literária

Cada área possui temas, subtemas e diferentes gêneros textuais.

### Metodologia de geração

Os textos sintéticos são produzidos por meio de prompts estruturados, com variações controladas de:

- Área, tema, subtema e gênero textual.
- Persona, público-alvo e contexto de escrita.
- Comprimento solicitado.
- Modelo gerador e parâmetros de geração.

A geração utiliza pipelines Python, vLLM/Transformers e APIs externas, com execução local ou em clusters de GPU.

O processo inclui validações de idioma, comprimento, integridade, formatação e aderência aos requisitos. Os metadados de geração devem ser preservados.

O corpus completo e o benchmark experimental são objetos distintos. O benchmark utiliza subconjuntos controlados para reduzir fatores de confusão (*confounders*) e permitir comparações justas entre detectores.

**Importante:** textos anteriores a 2022 reduzem o risco de contaminação por LLMs modernos, mas não garantem ausência absoluta de conteúdo automatizado.

## 2. Finalidade deste workspace

Este repositório é dedicado exclusivamente à **auditoria, validação e avaliação experimental do benchmark Hórus**, visando uma publicação científica.

A geração e coleta dos textos pertencem a pipelines externos. Este workspace deve consumir datasets versionados e preservar sua rastreabilidade.

### Objetivos principais

1. Auditar integridade, qualidade, diversidade e possíveis vieses do dataset.
2. Identificar atalhos estatísticos e artefatos de construção.
3. Avaliar detectores supervisionados, unários e zero-shot.
4. Investigar generalização entre geradores, domínios e fontes.
5. Avaliar robustez sob mudanças nas condições de produção textual.
6. Gerar resultados, figuras e tabelas reproduzíveis para publicação.
7. Preparar documentação e disponibilização do benchmark no Hugging Face.

**O foco é avaliar a qualidade e a utilidade do benchmark, não desenvolver um novo detector.**

## 3. Fundamentação científica

Os principais trabalhos de referência são:

- **RAID — Dugan et al. (ACL 2024):** construção de benchmarks robustos, perturbações e avaliação de detectores em diferentes condições.
- **M4 — Wang et al. (EACL 2024):** detecção multigerador, multidomínio e multilíngue; generalização entre distribuições.
- **The Two Paradigms of LLM Detection — Bevendorff et al. (2025):** distinção entre atribuição e verificação de autoria, problemas de conjuntos fechados e abertos.
- **DetectRL — Wu et al. (NeurIPS 2024):** robustez da detecção em cenários mais próximos do mundo real.
- **SemEval-2024 Task 8 — Wang et al. (2024):** protocolos e tarefas de avaliação de textos gerados por máquinas.

Métodos de detecção relevantes incluem **Ghostbuster, Fast-DetectGPT, Binoculars** e classificadores baseados em características estilométricas.

As implementações devem respeitar as metodologias originais, documentando adaptações necessárias para PT-BR.

## 4. Metodologia experimental

### Auditoria de qualidade

- Verificar duplicatas exatas e aproximadas, vazamento entre splits e integridade dos metadados.
- Analisar distribuição de áreas, fontes, geradores e comprimentos.
- Investigar qualidade linguística e aderência aos gêneros textuais.
- Avaliar rótulos de domínio por classificação automática e anotação humana.
- Examinar artefatos antes e depois da normalização.

### Sondas de atalhos

Utilizar modelos simples para investigar sinais discriminativos não necessariamente relacionados à autoria:

- TF-IDF de palavras.
- TF-IDF de 4-gramas de caracteres.
- Características superficiais e estilométricas.
- Classificadores lineares e representações contextualizadas.

### Avaliação de detectores

Comparar métodos de diferentes famílias:

- Supervisionados clássicos: Logistic Regression, Linear SVM.
- Supervisionados neurais: BERTimbau.
- Unários: métodos One-Class.
- Zero-shot e especializados: Fast-DetectGPT, Binoculars e Ghostbuster.

### Generalização

Implementar protocolos independentes:

- **IID:** avaliação convencional em distribuições semelhantes.
- **LOGO:** Leave-One-Generator-Out.
- **LODO:** Leave-One-Domain-Out.
- **LOSO:** Leave-One-Source-Out.
- **LOPO:** Leave-One-Prompt-Out, quando os metadados permitirem.

Avaliações adicionais podem contemplar mudanças simultâneas de distribuição e perturbações textuais.

Métricas principais: ROC-AUC, balanced accuracy, TPR@FPR fixo e intervalos de confiança.

Todos os splits, thresholds e parâmetros experimentais devem ser rastreáveis e reproduzíveis.

## 5. Evidências científicas da qualidade do benchmark

A validação do Hórus deve se concentrar em **seis pilares**, demonstrando que o dataset é confiável e que seus protocolos permitem avaliar detectores de forma válida, justa e reproduzível.

### 5.1. Qualidade e integridade dos dados

**Valor:** garantir que os resultados experimentais sejam fundamentados em dados corretamente rotulados e sem vazamentos.

- **Análises:** duplicatas e near-duplicates, validação de rótulos, inspeção humana, Cohen's kappa e qualidade linguística.
- **Evidência desejada:** dados rastreáveis, confiáveis, com baixa incidência de erros e sem contaminação relevante entre splits.

### 5.2. Ausência de atalhos triviais dominantes

**Valor:** verificar se detectores exploram artefatos de construção em vez de propriedades relevantes da autoria.

- **Análises:** char 4-grams, TF-IDF, características superficiais, comparação antes/depois da normalização e controle de fatores de confusão.
- **Evidência desejada:** identificar e minimizar pistas espúrias, demonstrando que os resultados não dependem predominantemente de vazamentos ou diferenças artificiais de coleta e formatação.

Alta acurácia de classificadores simples deve ser investigada, não interpretada automaticamente como qualidade ou defeito do benchmark.

### 5.3. Diversidade e representatividade

**Valor:** estabelecer a cobertura do benchmark e delimitar quais populações e condições de produção ele representa.

- **Análises:** distribuição por domínio, fonte humana, gerador, comprimento, gênero textual e cobertura das combinações entre variáveis.
- **Evidência desejada:** diversidade documentada e adequada às tarefas propostas, sem fatores excessivamente confundidos ou subconjuntos sistematicamente negligenciados.

Diversidade não implica representatividade universal. As limitações de cobertura devem ser explicitadas.

### 5.4. Capacidade de avaliar generalização OOD

**Valor:** demonstrar que o benchmark consegue revelar o comportamento dos detectores diante de distribuições desconhecidas.

- **Análises:** comparação IID × LOGO × LODO × LOSO, além de LOPO e shifts compostos quando viáveis.
- **Evidência desejada:** protocolos independentes e controlados que permitam identificar estabilidade, quedas de desempenho e diferenças entre famílias de detectores.

**Importante:** não buscamos necessariamente desempenho elevado ou baixo em OOD, mas uma avaliação confiável da generalização.

### 5.5. Robustez a perturbações

**Valor:** verificar se o benchmark permite avaliar detectores sob mudanças realistas nas condições textuais.

- **Análises:** alterações superficiais, normalização, estratificação por comprimento, pequenas edições e revisão humana de amostras.
- **Evidência desejada:** avaliações que revelem a sensibilidade dos detectores às transformações, preservando o controle sobre significado, procedência e rótulos.

Textos parcialmente editados por humanos devem ser identificados separadamente dos textos integralmente sintéticos.

### 5.6. Validade experimental e reprodutibilidade

**Valor:** garantir que os resultados sejam comparáveis, estatisticamente fundamentados e independentes de escolhas arbitrárias.

- **Análises:** splits sem leakage, múltiplas sementes, intervalos de confiança, ablações, baselines diversos e avaliação por subgrupos.
- **Evidência desejada:** resultados reproduzíveis, com protocolos documentados, thresholds definidos na validação e conclusões sustentadas por controles adequados.

### Resultados prioritários para publicação

Consolidar preferencialmente quatro conjuntos de evidências:

1. **Auditoria de atalhos:** separabilidade antes e depois de controles e normalização.
2. **Matriz IID × OOD:** comparação de detectores em LOGO, LODO e LOSO.
3. **Análise de confounders:** efeitos de comprimento, domínio, fonte e gerador.
4. **Qualidade e cobertura:** integridade, diversidade e validação humana dos dados.

Análises estilométricas, de entropia, parágrafos e *nuisance* devem complementar essas evidências, evitando gráficos redundantes.

### Distinções científicas fundamentais

**Não confundir quatro conceitos:**

- **Qualidade do dataset:** dados confiáveis, corretamente rotulados, rastreáveis e adequados às tarefas propostas.
- **Robustez do benchmark:** protocolos e conclusões resistentes a controles experimentais e variações razoáveis.
- **Capacidade de avaliar generalização:** existência de testes OOD válidos, com fatores de confusão controlados.
- **Generalização dos detectores:** capacidade dos modelos de manter desempenho em distribuições não vistas.

**A generalização é uma propriedade dos detectores; a capacidade de medi-la adequadamente é uma propriedade do benchmark.**

Não é necessário que os detectores obtenham alta acurácia, nem que apresentem grandes quedas em OOD, para demonstrar a utilidade do Hórus.

O objetivo é obter **evidências empíricas de validade**, não selecionar resultados favoráveis a uma conclusão predefinida. Hipóteses não confirmadas, resultados negativos e limitações devem ser documentados.

## 6. Organização do repositório

```text
horus-benchmark/
├── configs/
│   ├── datasets/
│   ├── detectors/
│   └── experiments/
├── data/
│   ├── manifests/
│   └── samples/
├── src/
│   ├── data/
│   ├── audit/
│   ├── features/
│   ├── detectors/
│   ├── protocols/
│   ├── metrics/
│   └── evaluation/
├── experiments/
│   ├── 01_quality_audit/
│   ├── 02_shortcut_probes/
│   ├── 03_iid_baselines/
│   ├── 04_generalization/
│   └── 05_robustness/
├── results/
│   ├── tables/
│   ├── figures/
│   └── logs/
├── paper/
├── scripts/
├── tests/
└── README.md
```

### Responsabilidades

- `src/`: implementações reutilizáveis.
- `configs/`: parametrização de datasets, modelos e experimentos.
- `experiments/`: configurações e scripts de execução das análises.
- `data/`: manifests, referências e pequenas amostras; evitar armazenar grandes datasets no Git.
- `results/`: resultados experimentais, tabelas e gráficos.
- `paper/`: materiais destinados à publicação.
- `tests/`: testes de integridade e comportamento dos pipelines.

## 7. Diretrizes para o agente

1. **Priorizar rigor científico, modularidade e reprodutibilidade.**
2. Antes de implementar, inspecionar o código existente e reutilizar componentes adequados.
3. Não modificar datasets originais; transformações devem produzir versões derivadas.
4. Evitar vazamentos entre treino, validação e teste, incluindo etapas de pré-processamento.
5. Preservar metadados, identificadores, sementes e versões dos experimentos.
6. Separar código de processamento, implementação de detectores e protocolos experimentais.
7. Não modificar definições metodológicas ou critérios de avaliação sem aprovação.
8. Não presumir que alta acurácia significa um benchmark de boa qualidade.
9. Não apresentar resultados exploratórios como resultados definitivos.
10. Documentar hipóteses, limitações, dependências e adaptações dos métodos.
11. Preferir configurações declarativas e execução parametrizada em vez de valores hardcoded.
12. Criar testes para transformações, integridade dos splits e cálculos de métricas.
13. Manter o código compatível, quando viável, com execução em cluster Slurm.
14. Não executar experimentos computacionalmente custosos sem solicitação.
15. **Todas as análises devem priorizar a validação e melhoria do benchmark, não a maximização da acurácia dos modelos.**
16. Investigar evidências favoráveis e contrárias à qualidade do benchmark, sem presumir antecipadamente sua robustez ou capacidade de avaliar OOD.
17. Distinguir propriedades dos dados, dos protocolos e dos detectores ao interpretar resultados.
18. Priorizar experimentos que respondam às seis dimensões de validação, evitando análises redundantes ou sem hipótese científica clara.
19. Não execute os programas de normalização, limpeza, seleção, análise, etc. Somente crie o código e envie para mim em ordem os coamndos para que eu possa executar e voce depois analisar.

## 8. Resultado esperado

Ao final, o workspace deve oferecer:

- Dataset validado, versionado e documentado.
- Protocolos de avaliação reproduzíveis.
- Baselines representativos de diferentes paradigmas de detecção.
- Auditoria sistemática de qualidade, vieses e atalhos.
- Experimentos de generalização IID/OOD.
- Evidências científicas da validade e das limitações do benchmark.
- Tabelas, figuras e análises para publicação.
- Estrutura preparada para disponibilização no Hugging Face.

**Princípio central:** o Hórus deve permitir investigar não apenas se um detector acerta, mas em quais condições ele acerta, quais fatores influenciam seu desempenho e até que ponto seus resultados generalizam para dados não vistos.

**A finalidade científica é demonstrar, por meio de evidências e controles experimentais, em que medida o benchmark é confiável, robusto e adequado à avaliação de detectores — reconhecendo explicitamente suas limitações.**