# Diagnóstico de seleção N2/N3

As duas views usam os mesmos IDs e o corpus completo. N3s preserva exatamente o conteúdo humano da N2 e aplica Markdown/placeholder só aos sintéticos. Nenhum critério de seleção está aprovado e nenhum corpus final foi escrito.

- `common_support.csv`: contagem e potencial de pares por área/faixa.
- `provenance_by_length_bin.csv`: fontes e geradores em cada faixa.
- `balance_smd_ks.csv`: diferenças padronizadas e KS por atributo/área.
- `composition_retention.csv` e `provenance_retention.csv`: perda de cobertura.
- `surface_probes.csv`: sondas exploratórias com ajuste no treino e limiar na validação.
- `candidate_overlap.csv`: alteração dos IDs selecionados ao mudar a view.
- `n3_changes_by_class_area.csv`: efeito da N3 real em cada classe.
- `candidate_membership.parquet`: IDs das simulações; arquivo local ignorado pelo Git.
- `length_by_area_*.png` e `scenario_tradeoff.png`: figuras.

`n2_fixed_*` em tabelas de métricas aplica os IDs escolhidos na N2 às outras views, isolando o efeito da apresentação. Os demais cenários são selecionados independentemente em cada view. Os splits dos probes agrupam `split_group_id` quando presente, mas near-duplicates e documentos-pai ainda não estão inteiramente resolvidos; resultados não são teste final nem OOD.
