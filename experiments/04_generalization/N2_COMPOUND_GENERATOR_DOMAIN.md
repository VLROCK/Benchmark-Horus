# OOD composto: gerador e domínio desconhecidos

## Hipótese e construção

Este protocolo pergunta como os detectores se comportam quando o **gerador sintético e a área textual** estão ausentes do treino. É complementar a LOGO e LODO isolados; uma queda ou estabilidade é uma propriedade dos detectores sob este desenho, não uma prova automática da qualidade do dataset.

A combinação de mudanças de gerador e domínio responde à questão de generalização multigerador e multidomínio estudada em M4. A população, o pareamento e as limitações aqui são próprios do Hórus.

O construtor `src/horus_benchmark/protocols/compound_ood.py` lê o release `horus-n2-iid-v1` e os manifests LOGO já verificados. Para cada semente e par gerador×área, o teste é exatamente a fatia daquela área do teste LOGO: sintéticos do gerador, humanos de controle pareados por área e faixa de comprimento N2. Os alvos sintéticos sem controle continuam identificados e são avaliados somente por taxas unárias; não entram nas métricas binárias. O desenvolvimento exclui todas as linhas do gerador, todas as linhas da área e todos os membros dos respectivos `probe_group_id`. O teste IID não é realocado para treino. Os folds de desenvolvimento originais são preservados.

Uma célula requer ao menos **200 textos por classe** no teste binário e as duas classes em cada fold de desenvolvimento. As células sem suporte ficam em `tasks.csv` com razão explícita. Não se selecionam alvos usando resultados de modelos. A origem dos humanos pode continuar concentrada dentro da área; isso é descrito, não corrigido artificialmente.

## Modelos, calibração e comparação

São repetidos os modelos OOD já usados: `surface_all`, `word_lr`, `char4_svm` e `human_iforest` em cinco folds por semente; `bertimbau` no fold zero por semente. O modelo é treinado de novo para cada alvo; seleção de hiperparâmetros e todos os pré-processadores ficam no treino. O limiar de Youden e os limiares de FPR de 1% e 5% são definidos na validação **do próprio protocolo**. Publicam-se ROC-AUC, balanced accuracy, TPR, FPR efetivo de teste, tamanho da validação e resolução mínima de FPR, cobertura pareada e taxas nos sintéticos sem controle.

O agregador compara `logo_lodo` a LOGO, LODO e IID por gerador×área×semente×modelo, usando apenas os **IDs de teste comuns**. Na comparação IID, a interseção pode ser pequena ou não ter ambas as classes; esses casos são marcados sem forçar métrica. Os cinco folds compartilham o mesmo teste e são agregados antes dos deltas. IC95% percentis de 2.000 reamostragens usam grupos `probe_group_id` vinculados entre os dois protocolos. As três sementes reutilizam o mesmo corpus, portanto as médias macro e as figuras são descritivas. Esses deltas refletem todo o protocolo, inclusive populações diferentes de treino e validação; não estimam uma interação causal isolada gerador×domínio.

## Rastreabilidade

Configurações: `configs/experiments/compound_ood_n2_v1.yaml`, `text_evaluation_compound_n2_v1.yaml` e `compound_ood_summary_n2_v1.yaml`. Cada manifest, plano, score e shard registra hashes. O merge final exige todos os alvos elegíveis, cinco modelos e três referências; produz tabelas por alvo, delta, grade sem suporte, macro por semente e mapas de ROC-AUC/FPR. Os arquivos originais e o release N2 permanecem intactos.

Execute `compound_ood`, `compound_check`, `compound_texts`, `compound_plan` e `compound_summary_plan` nessa ordem. Submeta os dois planos de modelos com `scripts/submit_text_evaluation_plan.sh` após exportar os caminhos específicos do experimento composto. Após a conclusão, submeta `scripts/submit_compound_summary.sh` e `compound_merge`. O roteiro completo de comandos está na resposta de entrega desta implementação.
