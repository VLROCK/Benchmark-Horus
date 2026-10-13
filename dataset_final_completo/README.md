# Dataset final completo Horus

Corpus documental canônico, não normalizado, construído sobre o release anterior. Os arquivos-fonte e o benchmark congelado não foram alterados.

Os textos Claude já presentes foram identificados por modelo/ID e SHA-256 de conteúdo. Apenas os novos textos foram adicionados; as colunas de qualidade deles foram calculadas pelo mesmo código de auditoria do release anterior. A campanha forneceu 15.000 registros válidos nos CSVs; 5 registros com caractere/token corrompido foram colocados em quarentena, não reparados silenciosamente. Entre os 11.243 textos novos incluídos, 11.224 receberam benchmark_eligible=True.

Dados: `data/all.parquet`, `data/train.parquet`, `data/validation.parquet`, `data/test.parquet`. Auditoria e contagens: `audit/`.

Este corpus completo não é o benchmark balanceado nem a view normalizada. Antes de treinar ou publicar uma nova versão, refaça os controles de normalização, leakage e licenças apropriados para a nova composição.
