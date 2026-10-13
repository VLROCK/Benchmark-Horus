# Auditoria exploratória — complete-with-claude-provisional-v1

Gerada em 2026-10-13T18:23:17.138230+00:00 (UTC).
Corpus analisado: **195,154** textos. Esta versão é **provisional**.

## Integridade e leakage

- Checagens estruturais com contagem positiva: 1 erros e 2 avisos.
- Veja `tables/integrity_checks.csv` para todas as contagens e `tables/repeated_groups.csv` para IDs de revisão.
- Colisões de `content_sha256_audit` são igualdade exata de bytes UTF-8; `comparison_sha256_audit` usa NFKC, casefold e espaços colapsados apenas para comparar.
- Grupos de URL humana ou `source_record_id` em mais de um split são indícios de possível parentesco documental; precisam de revisão da proveniência.
- Este catálogo não inclui um benchmark materializado; as checagens específicas do benchmark não foram executadas.
- content_sha256_audit: 0 grupos atravessam splits.
- comparison_sha256_audit: 0 grupos atravessam splits.
- split_group_id: 0 grupos atravessam splits.
- human_source_url: 2261 grupos atravessam splits.
- Busca aproximada: 4883 pares verificados, 191 entre splits; 5988 textos curtos sem assinatura.
- A busca por assinaturas é uma triagem com limites de recuperação. Zero pares encontrados não comprova ausência de near-duplicates.

## Cobertura e possíveis atalhos

As tabelas de cobertura incluem classe × domínio × comprimento, fonte humana × domínio, gerador × domínio e completude de metadados.
`shortcut_gaps.csv` registra diferenças padronizadas de médias e KS para medidas numéricas, além de diferenças de prevalência para marcas de apresentação.
Essas diferenças são descritivas: misturam autoria, domínio, fonte, gênero e condições de coleta/geração; não identificam causa isoladamente.

## Sondas diagnósticas

Sondas lineares usam apenas atributos superficiais ou TF-IDF de 4-gramas de caracteres. O vocabulário, a imputação e a escala são ajustados apenas no treino; o limiar é escolhido na validação e a métrica é calculada no teste.
Os intervalos são bootstrap estratificado por rótulo sobre previsões do teste; não incluem variação de treino, fonte ou gerador.
Alta separabilidade indica sinal explorável e exige controle adicional; não demonstra que o sinal é espúrio nem valida o benchmark.

## Limites da inferência

Esta auditoria não testa contaminação dos dados de pré-treino de LLMs externos. IDs de documento-pai/trecho acadêmico não aparecem no schema atual, logo parentesco entre introdução e conclusão não pode ser descartado por completo.
Não houve limpeza, seleção, normalização ou alteração de rótulos nesta execução.
