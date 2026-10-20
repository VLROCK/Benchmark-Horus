# Revisão de artefatos e proposta de normalização

**Estado:** protocolo exploratório proposto; não define ainda a seleção ou a
normalização final do Hórus. Nenhum texto original deve ser alterado. A execução
do exportador e de quaisquer testes cabe ao pesquisador, conforme a regra 19 do
`AGENTS.md`.

## Amostragem

O comando abaixo lê o corpus configurado em `configs/datasets/horus_full.yaml` e
o piloto humano já existente. Produz uma planilha de revisão cega, um manifesto
com procedência e um resumo com hashes e contagens em um diretório novo.

- **Núcleo humano:** os mesmos 120 textos do piloto, 20 por área. A amostra
  original não foi estratificada por fonte nem por comprimento.
- **Núcleo sintético:** dois textos por combinação observada de área e gerador.
  Na versão atual, são 6 áreas × 10 geradores = 120 textos. A escolha usa uma
  prioridade derivada de SHA-256 de `seed|record_id`, independente da ordem de
  leitura do Parquet.
- **Extensão humana:** 20 novos textos por área (120 no total), escolhidos para
  ampliar a cobertura de fonte, faixa de comprimento real e split.
- **Extensão sintética:** dois textos adicionais por combinação de área e
  gerador (120 na versão atual), buscando faixas de comprimento, temas, gêneros
  e splits pouco representados no núcleo.

Os 240 por classe **não são uma amostra aleatória simples do corpus**. A
extensão é deliberadamente diagnóstica. Não somar núcleo e extensão para alegar
prevalência populacional. Mesmo no núcleo, comparações entre classes exigem
análise por área e, no sintético, por gerador; o sorteio equilibrado por célula
não preserva necessariamente as proporções reais do corpus. O objetivo imediato
é descobrir categorias de artefatos e testar regras com exemplos variados.

O manifesto preserva rótulo, área, fonte/gerador, split, tema, gênero,
identificadores de agrupamento, hash e contagens do texto original. A planilha
de revisão oculta rótulo e origem para reduzir viés na anotação. A ordem das
linhas é embaralhada deterministicamente. O campo `content` permanece literal.
Como esse CSV contém texto bruto não confiável, abra-o em um visualizador de
texto/CSV que não execute fórmulas automaticamente; não edite o campo `content`.

## Códigos de anotação propostos

Preencher `artifact_codes` com códigos separados por `;`. Um código indica que
o fenômeno foi **observado**, não que o texto deve ser excluído ou transformado.
Usar `span_or_excerpt` para localizar o trecho e `review_note` para explicar a
decisão. `severity`: `baixo`, `medio`, `alto`. `proposed_action`: `manter`,
`normalizar`, `revisar`, `fora_benchmark_principal`.

| Código | Fenômeno a verificar | Decisão inicial |
| --- | --- | --- |
| `ENCODING` | Mojibake, entidade HTML residual, controle ou caractere substituto | Reparo conservador; revisão se ambíguo |
| `HTML_CSS` | HTML editorial, CSS do editor ou fragmento de página | Converter apenas estrutura editorial; proteger exemplos de código |
| `MARKUP` | Markdown/fences/links usados como invólucro de resposta | Neutralizar somente marcadores estritamente editoriais; preservar listas legítimas |
| `URL_EMAIL` | Endereço literal | Substituição consistente por `URL`/`EMAIL` em ambas as classes; guardar contagens |
| `PREAMBLE` | “Aqui está o texto”, aviso de IA ou conclusão fora do texto solicitado | Remoção apenas com padrão de alta precisão |
| `PROMPT_ECHO` | Instruções, campos de prompt, persona ou pedido reproduzidos na resposta | Revisar procedência e limites antes de remover |
| `PLACEHOLDER` | `[insira ...]`, campos vazios e texto-modelo explícito | Revisar; conteúdo central incompleto não deve ser “inventado” |
| `REASONING` | Vazamento de raciocínio interno ou metacomentário do gerador | Revisão de rótulo/conteúdo; não mesclar ao texto autoral |
| `LANG_SECTION` | Abstract, resumo paralelo ou seção substancial em outro idioma | Segmentar por seção; preservar corpo PT-BR se recuperável |
| `REFERENCES` | Lista bibliográfica terminal ou bloco de notas/fonte externo ao corpo | Remover só se limites inequívocos; manter citação no corpo |
| `EDITORIAL` | Navegação, publicidade, crédito, data de atualização, pedido de comentário | Remover apenas segmento inequívoco |
| `AGGREGATION` | Vários posts, manchetes ou documentos concatenados | Revisão ou exclusão do benchmark principal |
| `FRAGMENT` | Texto cortado, apenas cabeçalho, ementa parcial ou fim incompleto | Revisão; não completar por geração |
| `REPETITION` | Laço repetitivo ou parágrafos duplicados no mesmo registro | Revisão; distinguir refrão/estrutura intencional |
| `GENRE_MISMATCH` | Texto não corresponde ao gênero/área declarados | Revisão de metadados, não limpeza textual automática |

Não tratar pontuação, gíria, erros ortográficos, citações jurídicas, marcadores
de lista e títulos de seção como defeitos apenas por serem discriminativos.
Marcar ambos os lados de uma ambiguidade no comentário. Para códigos raros ou
subjetivos, propor dupla anotação e adjudicação antes de fechar regras.

## Evidência quantitativa já disponível, não inspeção manual

A auditoria de 2026-10-13 em
`results/audits/20261013T180229080021Z_quality-shortcuts-full-v1_1727096142/tables/boolean_by_label_domain.csv`
marcou, nos 16.250 sintéticos informacionais, 11.642 com `markdown_bullet`
(71,6%), 735 com `markdown_code_fence` (4,5%) e 544 com `html_tag` (3,3%).
Entre os 10.341 humanos informacionais, 93 foram marcados com
`markdown_bullet` (0,9%). Em acadêmico, o mesmo sinal apareceu em 3.894/13.814
humanos (28,2%) e 387/16.249 sintéticos (2,4%). A regex identifica formato,
**não** prova que se trate de lixo. Listas técnicas, tópicos acadêmicos e
documentos jurídicos podem ser legítimos. A revisão amostral deve determinar
quais instâncias são invólucros removíveis e quais pertencem ao gênero.

## Plano de normalização a submeter à aprovação metodológica

1. **Congelar e rastrear:** preservar corpus original, versão, hash, IDs, grupos
   documentais, splits e metadados. Recontar caracteres (`len` de pontos de
   código) e palavras por espaços no texto original e no derivado; não usar
   contagens declaradas como única base de seleção.
2. **N1 técnica, simétrica:** NFC; reparo de mojibake somente inequívoco;
   conversão de HTML editorial; controles e espaços; neutralização de
   marcadores Markdown estritamente editoriais; substituição de URLs/e-mails;
   preâmbulos e placeholders explícitos. Proteger código/HTML quando forem o
   conteúdo legítimo. Não corrigir gramática, estilo, gíria ou ortografia.
3. **N2 estrutural, com revisão:** segmentar resumos paralelos em outros idiomas,
   bibliografias finais e rodapés/navegação; aplicar a mesma regra a humanos e
   sintéticos quando o mesmo fenômeno ocorre. Não apagar citações internas,
   títulos, listas ou trechos em outro idioma que pertençam ao texto autoral.
   `AGGREGATION`, `FRAGMENT` e `GENRE_MISMATCH` vão para revisão, não para
   reparo agressivo.
4. **Guardas e elegibilidade:** registrar spans retirados, motivo, versão da
   regra, perdas absolutas e relativas. Como candidato, manter a guarda anterior:
   texto com ≥50 palavras originais que perde ≥20 palavras e retém <90% vai
   para revisão. Sinalizar ainda texto vazio, idioma central inválido e
   duplicação surgida após normalização. Não excluir automaticamente por um
   simples marcador superficial.
5. **Validação antes/depois:** comparar exatamente os mesmos IDs e splits na
   *view* original e N1/N2. Repetir auditoria de integridade, near-duplicates,
   gaps de comprimento/pontuação/formatação e sondas de superfície, palavras e
   char-4-gramas. Revisar pares de textos para perda semântica e mudanças de
   gênero. Fazer ablação 2×2 (com/sem seleção × original/normalizado) para não
   atribuir à normalização um efeito causado pela seleção. Definir thresholds
   na validação e manter teste independente.

Uma redução de separabilidade de sondas simples é evidência relevante, mas não
prova isolada de melhoria: a transformação também precisa preservar conteúdo,
rótulo, procedência, diversidade e utilidade dos protocolos OOD. A abordagem
de avaliação multigerador/multidomínio e a robustez a perturbações são alinhadas
com M4, RAID e DetectRL; **as regras N1/N2 são hipóteses próprias do Hórus**, não
etapas prescritas por esses trabalhos.

## Execução pelo pesquisador

No PowerShell, a partir da raiz do repositório, primeiro rode os testes e
depois exporte a amostra. Escolha um diretório de saída novo e vazio.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_artifact_review.py -q
.\.venv\Scripts\python.exe -m horus_benchmark.audit.artifact_review `
  --dataset-config configs/datasets/horus_full.yaml `
  --human-pilot results/theme_validation/20261015T190353941660Z_human-theme-validation-v1-pilot_82c08ae5f1/pilot_annotation.csv `
  --output results/artifact_review/full-v1-seed-20261020 `
  --seed 20261020
```

O comando não normaliza nem seleciona o benchmark final. Após executá-lo,
envie o caminho de `sampling_summary.json` e `blind_review.csv` para a
interpretação qualitativa. Não edite `sample_manifest.csv`: ele permite ligar
as observações cegas a classe, gerador e fonte sem alterar a planilha humana.

