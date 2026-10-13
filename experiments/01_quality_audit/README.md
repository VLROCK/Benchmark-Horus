# 01 — Integridade, leakage e cobertura

O protocolo inicial está em [METHODS.md](METHODS.md). Ele avalia o corpus completo da versão cadastrada, sem selecionar ou transformar textos.

Para o corpus completo atual com Claude, use a configuração própria e os comandos em [RUN_FULL.md](RUN_FULL.md). Os comandos abaixo permanecem como referência da auditoria anterior do release candidate.

```powershell
horus-audit run --config configs/experiments/quality_audit_rc.yaml
```

Cada execução cria `results/audits/<run_id>/` com `REPORT.md`, tabelas CSV, figuras PNG/SVG, hashes dos arquivos de entrada, configuração resolvida e `features.parquet` por registro. A nova versão do corpus pode ser auditada trocando `dataset_config` por um catálogo novo.

O snapshot provisório atual foi auditado em [FINDINGS.md](../../results/audits/20261008T184438415365Z_quality-shortcuts-v1_d5a552362a/FINDINGS.md). Para regenerar apenas as tabelas/gráficos descritivos ou revalidar as partições contra os mesmos arquivos de entrada:

```powershell
horus-audit refresh --run results/audits/<run_id> --config configs/experiments/quality_audit_rc.yaml
horus-audit recheck-integrity --run results/audits/<run_id> --config configs/experiments/quality_audit_rc.yaml
horus-audit verify-near --run results/audits/<run_id> --config configs/experiments/quality_audit_rc.yaml
```

Depois de gerar uma versão normalizada com os mesmos `record_id`, rótulos, domínios e splits, execute a mesma auditoria nessa view e compare as duas rodadas:

```powershell
horus-audit compare --before results/audits/<original> --after results/audits/<normalizado> --output results/audits/comparison-original-normalizado
```

O comparador rejeita versões com IDs, rótulos, domínios ou splits diferentes. Para comparar corpora de composição diferente, faça primeiro uma análise de cobertura separada; a redução de um gap em amostras diferentes não mede o efeito da normalização.

