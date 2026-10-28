# Protocolo exploratório de atalhos: corpus original, N1 e N2

Esta etapa não seleciona textos, não altera corpus/splits e não constitui um
benchmark final. Congelamos a view N2 candidata com hash de cada artefato,
mantendo N1 e o original como entradas imutáveis para comparação.

## Hipótese e desenho

Queremos medir se sinais superficiais ou lexicais que discriminam as classes
mudam com a limpeza. A mesma amostra determinística de IDs é usada nas três
versões; os modelos são ajustados separadamente em cada versão, sempre no
split de treino herdado. O limiar de balanced accuracy e o de TPR@FPR=1% são
fixados apenas na validação. Reportamos AUC, balanced accuracy, TPR e FPR
observado no teste, com bootstrap pareado e estratificado por classe para as
diferenças raw→N1, N1→N2 e raw→N2.

Sondas documentais: TF-IDF de unigramas/bigramas de palavras + regressão
logística; TF-IDF de 4-gramas de caracteres + regressão logística; e quatro
grupos de características superficiais (comprimento, pontuação, apresentação,
conjunto completo). Os vetorizadores, imputadores e escaladores são ajustados
somente no treino. Metadados como fonte e gerador jamais entram como features.

Para granularidade, uma frase é uma sentença completa detectada por regra
conservadora de pontuação; um parágrafo é um bloco separado por linha em
branco. Selecionamos no original um único segmento por documento cujo centro
está entre 35% e 65% do comprimento. A fração máxima do documento impede
que um parágrafo único com título/abstract/rodapé inteiro seja chamado de
"miolo". A identificação em N1/N2 exige correspondência única e similaridade
de pelo menos 0,92. Se removido, ambíguo ou com fronteira alterada, o segmento
não é substituído pelo novo segmento naquela posição. Tabelas de cobertura
guardam toda a perda de amostra por classe e split; só tripletos completos
entram nas sondas pareadas de frase e parágrafo.

Os controles *nuisance* incluem sondas superficiais e leitura das métricas por
área e por faixas de comprimento **do documento original**, fixas nas três
versões. Essas estratificações não equivalem a um matching causal de gênero,
fonte e comprimento; a seleção final tratará esses fatores separadamente.

## Limites

Os splits herdados ainda possuem near-duplicates entre partições e não há ID
de documento-pai. Assim, este resultado é exploratório, não leakage-free. O
rótulo da frase/parágrafo é herdado do documento e pode falhar em citações,
transcrições ou conteúdo misto. A segmentação por pontuação/linha em branco é
uma aproximação, não anotação humana. O alinhamento de alta precisão descarta
casos incertos, possivelmente com viés de cobertura; sua taxa deve ser lida
junto às métricas. OOD (LOGO/LODO/LOSO) será avaliado **depois** da seleção e
dos novos splits documentais.

## Execução no cluster

Veja `scripts/slurm_compare_normalization_views.sh`. Antes do `sbatch`, rode
`python -m horus_benchmark.data.freeze_normalization freeze` uma única vez para
N2, ou `check` se o manifesto já existir. Não execute este protocolo no laptop.
