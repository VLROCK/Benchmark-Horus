# Fonte/gerador no IID — comparação exploratória v2 × v3

Os mesmos IDs, grupos e sementes são usados na v2 e v3. A v3 acrescenta um termo suave no objetivo de alocação, dentro de área e classe, para fontes humanas e geradores sintéticos com ao menos 100 textos. Fontes únicas não fornecem contraste. Nenhum texto é removido.

`v2_v3_split_comparison.csv` compara desvios de proveniência e comprimento. `provenance_shift_*.csv` contém cada fonte/gerador e o desvio de participação em relação ao conjunto correspondente. `iid_holdout_5fold_*.parquet` contém os novos splits candidatos. `largest_group_summary.csv` e `literary_dash_loss_by_genre_generator.csv` resumem as inspeções. Os arquivos `*_blind.jsonl` contêm textos e não entram no Git. A visão de controle de layout da v2 é reutilizada, sem recomputação.

Os gates do resumo são exploratórios: não há aprovação automática, freeze, treinamento ou OOD.
