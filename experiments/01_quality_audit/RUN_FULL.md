# Executar auditoria do corpus completo com Claude

Esta rodada usa `dataset_final_completo/data/all.parquet` (195.154 textos), os três splits físicos e a taxonomia registrada em `configs/datasets/horus_full.yaml`. O arquivo original não é alterado. A configuração em `configs/experiments/quality_audit_full.yaml` preserva as definições da auditoria anterior para permitir leitura lado a lado; a composição do corpus mudou, portanto diferenças entre rodadas não medem isoladamente o efeito dos novos textos.

**Não há benchmark balanceado materializado neste catálogo.** A auditoria executa as checagens do corpus e dos splits, e registra explicitamente que as checagens específicas de benchmark foram omitidas. `lineage_fields` permanece vazio porque o schema não declara ID de documento-pai.

No PowerShell, execute **nesta ordem**:

```powershell
Set-Location -LiteralPath 'C:\Users\Victor\Documents\ic horus\benchmark'

# 1. Dependências da auditoria e testes.
.\.venv\Scripts\python.exe -m pip install -e '.[audit,dev]'

# 2. Testes unitários e estruturais, inclusive os do corpus completo.
.\.venv\Scripts\python.exe -m pytest -q

# 3. Presença de arquivos, contagens e schemas declarados.
.\.venv\Scripts\horus-data.exe --config configs/datasets/horus_full.yaml validate

# 4. Auditoria completa: hashes, integridade, duplicatas, leakage,
# cobertura, gaps superficiais, sondas TF-IDF e figuras.
$AuditRun = & .\.venv\Scripts\horus-audit.exe run --config configs/experiments/quality_audit_full.yaml
if ($LASTEXITCODE -ne 0) { throw 'A auditoria falhou; consulte o log da rodada em results/audits.' }
$AuditRun = $AuditRun.Trim()
if (-not (Test-Path -LiteralPath (Join-Path $AuditRun 'audit_summary.json'))) {
    throw 'A auditoria não produziu o resumo esperado.'
}
Write-Output "Rodada: $AuditRun"

# 5. Verificação independente dos pares aproximados salvos.
.\.venv\Scripts\horus-audit.exe verify-near --run $AuditRun --config configs/experiments/quality_audit_full.yaml
if ($LASTEXITCODE -ne 0) { throw 'A verificação dos pares aproximados falhou.' }

# 6. Resumo e localização dos resultados para compartilhar depois.
Get-Content -LiteralPath (Join-Path $AuditRun 'audit_summary.json')
Get-Content -LiteralPath (Join-Path $AuditRun 'near_duplicate_literal_verification.json')
Write-Output $AuditRun
```

Pare e envie o erro se alguma etapa falhar; não interprete uma pasta parcial como rodada concluída. A etapa 4 pode demorar, pois lê todo o corpus e treina as sondas diagnósticas. Ela grava `REPORT.md`, `audit_summary.json`, `input_files.json`, tabelas em `tables/` e figuras PNG/SVG em `figures/` dentro da pasta impressa em `$AuditRun`.

Para uma conferência posterior da integridade contra os mesmos arquivos de entrada, sem repetir sondas nem busca aproximada:

```powershell
.\.venv\Scripts\horus-audit.exe recheck-integrity --run $AuditRun --config configs/experiments/quality_audit_full.yaml
```

Não use `horus-audit compare` entre o release anterior de 183.911 textos e este corpus de 195.154: ele exige exatamente os mesmos IDs. Essa comparação pareada é destinada ao corpus atual **antes e depois de uma futura normalização**. Nenhuma normalização, seleção, limpeza ou classificação temática humana é executada pelos comandos acima.
