# Protocolo IID agrupado e controle de layout — candidatos

A seleção N2 `w0.25_s20261103` fornece os mesmos IDs para as duas condições. `iid_holdout_5fold_seed*.parquet` reserva 15% para teste e marca os 85% restantes como desenvolvimento com `fold` de 0 a 4. Teste tem `fold=-1`. Em cada rodada, o fold indicado é validação e os outros quatro são treino. Todos os pares aproximados observados ficam na mesma partição e fold. `layout_control_corpus.parquet` troca quebras de linha por espaços em ambas as classes; a sequência de tokens separados por whitespace é idêntica à N2. A N2 original continua sendo a condição principal.

`holdout_*`, `dev_fold_*`, `cv_round_roles_*` e `largest_groups_*` permitem avaliar a composição antes de congelar os splits. `format_samples_manifest.csv` identifica amostras por fonte, área e motivo; `format_samples_blind.jsonl` contém os textos bruto/N2/N3s e fica fora do Git. O layout original dos sites não está nos Parquets: a causa do parágrafo único não pode ser concluída apenas com este material.

Estes splits são candidatos IID. Não há folds OOD, treinamento ou release final.
