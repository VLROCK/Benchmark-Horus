# Hórus Benchmark

Infraestrutura para auditoria, validação e avaliação reprodutível do benchmark Hórus de detecção de textos humanos versus sintéticos em português brasileiro.

O workspace prepara o acesso aos dados e a rastreabilidade dos experimentos. Ele não implementa, nesta etapa, limpeza, normalização, balanceamento adicional, treinamento ou avaliação de detectores.

## Estado dos dados

`dataset_final_completo/data/all.parquet` é o corpus documental completo de
trabalho, com 195.154 textos e as novas gerações Claude. O catálogo atual é
[`configs/datasets/horus_full.yaml`](configs/datasets/horus_full.yaml) e sua
proveniência está em
[`data/manifests/horus_full_v1.yaml`](data/manifests/horus_full_v1.yaml).
Esse corpus permanece imutável e ainda não equivale a um benchmark balanceado
congelado ou a uma versão pronta para publicação.

O [inventário anterior](data/manifests/DATA_INVENTORY.md) e o manifesto
`horus_rc_v0.yaml` descrevem o release candidate de 183.911 textos e são
referências históricas. Os CSVs-fonte originais não estão neste workspace.
Pendências de licença e redistribuição precisam ser resolvidas antes de uma
publicação pública.

Os arquivos em `dataset_final_completo/data/` não entram no Git. Após clonar o
repositório no cluster, transfira os quatro Parquets para esse mesmo diretório;
o catálogo usa esses caminhos relativos. Modelos e saídas volumosas também
permanecem locais e estão listados em `.gitignore`.

## Instalação

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Em Linux/Slurm, ative o ambiente com `source .venv/bin/activate`. O pacote não depende de caminhos absolutos: todos os recursos são resolvidos a partir do catálogo YAML.

## Uso básico

Inspecionar um recurso sem carregar a coluna de texto:

```powershell
horus-data --config configs/datasets/horus_full.yaml inspect --resource corpus
```

Validar existência, contagem de linhas e schema dos recursos registrados:

```powershell
horus-data --config configs/datasets/horus_full.yaml validate
```

Executar os testes:

```powershell
pytest
```

Os testes de integração usam apenas metadados ou colunas projetadas do snapshot local; são ignorados automaticamente quando os dados não estão disponíveis.

## Auditoria de integridade e atalhos

A [auditoria anterior](results/audits/20261008T184438415365Z_quality-shortcuts-v1_d5a552362a/FINDINGS.md)
e `configs/experiments/quality_audit_rc.yaml` referem-se ao release candidate
anterior. Não são resultados da composição completa com Claude. O método e
seus limites estão em
[experiments/01_quality_audit/METHODS.md](experiments/01_quality_audit/METHODS.md).
Qualquer nova auditoria do corpus completo deve usar uma configuração própria
e registrar novamente hashes e cobertura.

Os comandos para auditar o corpus completo de 195.154 textos, sem executá-los
automaticamente, estão em
[experiments/01_quality_audit/RUN_FULL.md](experiments/01_quality_audit/RUN_FULL.md).

## Organização

- `configs/`: catálogos de datasets e configurações declarativas de experimentos.
- `data/raw/`: destino futuro de snapshots brutos imutáveis.
- `data/processed/`: versões derivadas, sempre sem sobrescrever entradas.
- `data/benchmarks/`: subconjuntos e splits materializados para avaliação.
- `data/manifests/`: inventário, schema, versões e proveniência.
- `data/samples/`: pequenas amostras autorizadas para desenvolvimento.
- `src/horus_benchmark/`: componentes Python reutilizáveis.
- `experiments/`: futuras análises separadas por finalidade científica.
- `results/`: tabelas, figuras e logs produzidos por execuções.
- `tests/`: testes unitários e estruturais.
- `paper/`: materiais da publicação.

## Decisões arquiteturais

1. Os snapshots importados não são movidos, copiados ou alterados pelos pipelines.
2. O acesso é centralizado em um catálogo YAML e em um loader baseado em PyArrow, com projeção de colunas, filtros e leitura em batches.
3. Metadados de tema, fonte, gerador e geração são preservados para auditoria; não são implicitamente tratados como features de detectores.
4. Toda transformação futura deve gerar um novo recurso versionado e um novo manifesto.
5. Configurações resolvidas, sementes e metadados de execução podem ser persistidos por `horus_benchmark.runtime`.

As diretrizes científicas completas permanecem em `AGENTS.md`.

## Padronização temática dos textos humanos

O protocolo de classificação de `theme` por seis modelos BERTimbau e revisão
humana está em [experiments/01_quality_audit/human_theme_validation/README.md](experiments/01_quality_audit/human_theme_validation/README.md).
Ele usa `dataset_final_completo` como corpus de entrada e produz somente
artefatos derivados, sem alterar o snapshot original. Os comandos de execução
estão documentados em ordem para o pesquisador rodar quando desejar.
