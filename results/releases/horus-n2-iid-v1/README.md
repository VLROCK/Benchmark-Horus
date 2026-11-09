# Hórus N2 IID v1

Seleção `w0.25_s20261103` sobre N2, com teste reservado de 15% e cinco folds de desenvolvimento por seed. Cópias dos três splits v3 estão neste diretório e são verificadas por SHA-256; os textos permanecem no corpus N2. `selected_index.parquet` contém IDs e metadados principais, sem conteúdo textual. Execute `python -m horus_benchmark.protocols.freeze_iid check --release-dir results/releases/horus-n2-iid-v1` antes de usar o release. Esta versão não inclui avaliações de detectores ou protocolos OOD.
