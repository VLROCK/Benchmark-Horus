# Detectores especializados e robustez no Abaporu

## Escopo e entradas

Campanha derivada de `horus-n2-iid-v1` (107.566 documentos N2, 53.783 por classe), sem modificar o release. O pesquisador executa os programas no cluster. As análises de unidades, seleção e OOD já submetidas continuam independentes. As novas famílias começam no documento; a bateria adversarial começa no IID.

O acesso é o habitual do pesquisador: headnode, bifrost e `ssh abaporu`. Os scripts são executados **depois** desse acesso, no checkout `/mnt/recod/homes/victor.roquetto/Benchmark-Horus`. O cache HF é `/mnt/recod/homes/victor.roquetto/.cache/huggingface/hub`. Nenhum script depende de `$HOME` resolver para `/mnt`.

Entradas grandes que o Git não transfere:

- `results/releases/horus-n2-iid-v1/selected_index.parquet` e os três `split_seed*.parquet`;
- `results/evaluation/n2-v1/prepared/selected_n2_texts.parquet` e seu `prepared_manifest.json`;
- `results/protocols/ood-n2-v1/` com os manifests Parquet e índice existentes;
- `results/protocols/compound-ood-n2-v1/` para a fase composta;
- planos/previsões/metrics de `results/evaluation/n2-v1/` e os 15 checkpoints IID BERTimbau, para concordância de erros e ataques nos detectores originais.

Esses diretórios precisam estar materializados no Abaporu, mantendo os caminhos relativos. A fase `prepare` verifica hashes e identidade. Não recrie a seleção ou os splits para resolver uma entrada ausente. A avaliação limpa dos novos métodos exige o cache selecionado, não o corpus bruto completo.

## Métodos e adaptações

As configurações estão em `configs/experiments/advanced_detectors_n2_v1.yaml`, `configs/detectors/classical_adapted_v1.yaml`, `configs/detectors/lm_pairs_v1.yaml` e `configs/experiments/robustness_n2_v1.yaml`.

| Configurações | Definição |
|---|---|
| `stylo_lr`, `stylo_svm` | MATTR50, palavras funcionais portuguesas fixas, UPOS, pontuação, caixa, comprimento e legibilidade PT; C em 0,1/1/10 por CV interna |
| `stylo_*_no_structure` | Ablação das medidas explícitas de tamanho/layout; não garante independência estatística de comprimento |
| `alta_nb` | TF-IDF palavras 1–2 gramas, min_df5/max_df0,95/max200k, MultinomialNB alpha1 |
| `alta_delta` | Burrows Delta sobre 100/500/1000 palavras frequentes, vocabulário/perfis/padronização somente no treino |
| `alta_readability` | Flesch PT e medidas brutas; PCA até7 componentes, KNN k5 |
| `alta_pos` | Frequências UPOS; KNN k3 |
| `alta_entropy` | Entropia cruzada por maiores correspondências de caracteres; histórico até2M caracteres/classe; autômato de sufixos, com separadores documentais |
| `alta_vote` | Cinco componentes anteriores; pesos por kappa OOF dentro do treino, negativos zerados e fallback uniforme |
| `ghostbuster_ptbr_adapted` | Qwen0.5B/1.5B base; unigramas/trigramas humanos de treino; desconto0,9; 2.534 expressões de profundidade3; busca progressiva por AUC interna; LR C10 |
| `binoculars_falcon` | Falcon7B base/7B-Instruct, fórmula publicada, escore orientado como negativo da razão original |
| `fastdetect_gptj_neo` | Referência GPT-J6B, scorer GPT-Neo2.7B; discrepância condicional analítica |
| `fastdetect_falcon` | Mesma fórmula usando o par Falcon compartilhado com Binoculars |
| `binoculars_qwen_adapted`, `fastdetect_qwen_adapted` | Par Qwen2.5-7B base/Instruct, adaptação multilíngue identificada separadamente |

ALTA foi publicado para sentenças inglesas em documentos híbridos. Aqui os componentes também são adaptados para documentos PT-BR; nenhum resultado é uma reprodução literal do shared task. Sílabas são aproximadas pela hifenização Pyphen pt_BR, com biblioteca fixada. Stanza1.10.1 usa tokenize/mwt Bosque e POS Bosque sem charLM; os arquivos de modelo e resources.json são fixados por hash. A amostragem de históricos de entropia é determinística por ID, com teto igual por classe. Correspondências nunca atravessam documentos. Posições sem correspondência recebem comprimento1 para obter uma estimativa finita.

Ghostbuster preserva a gramática e os sete atributos manuais do repositório original, inclusive a convenção publicada de outliers aplicada a probabilidades. Não se troca silenciosamente esse atributo por um limiar de surpresa. Qwen substitui ada/davinci e os humanos do treino substituem Brown. N-gramas, seleção e escalas são ajustados novamente dentro de cada fold interno. Validação externa só define os limiares. Não há limite de iterações escolhido a partir do teste.

As quatro duplas LM são carregadas apenas para extração; pesos FP16/BF16, estatísticas FP32, batch1, limite512, texto direto sem template de chat. Escores por documento são compartilhados entre os protocolos. O caso de cabeça preenchida GPT-J50400/Neo50257 só é aceito após verificar o mapa de IDs do prefixo comum e o alinhamento de cada texto. Binoculars exige cabeças iguais e mantém a convenção oficial de máscara/denominador; não misturamos distribuições de tokenizers arbitrários. O uso de estatísticas FP32 é registrado, não implica equivalência bit a bit à precisão de todo upstream.

Snapshots são resolvidos para commits antes do download e fixados em `models.lock.json`, com hashes. A extração guarda escores, IDs de tokens/probabilidades fracas necessárias e cobertura; nunca salva os logits completos. Falhas de tokenização/variância e textos com menos de2 tokens ficam explícitos. A população válida pode diferir por método e a comparação usa sua interseção com retenção publicada.

## Protocolos e orçamento

- IID: três sementes congeladas, cinco folds, todos os16 modelos/configurações (240 tarefas lógicas).
- LOGO/LODO/LOSO: fold0 nas três sementes, com `stylo_svm`, `alta_vote`, Ghostbuster e os cinco escores congelados (8 configurações). O índice existente tem156 alvos/sementes elegíveis:1.248 tarefas lógicas.
- OOD composto: mesmo esquema,180 alvos/sementes no índice atual:1.440 tarefas lógicas.
- A execução agrupa até quatro tarefas lógicas por alocação CPU. As configurações e o suporte final dos índices determinam as contagens, publicadas nos planos. Isso não repete inferência LM por alvo.
- Prefixo comum: controle adicional com o mesmo trecho literal que cabe em512 tokens de todos os tokenizers LM e BERTimbau. Reajusta os novos modelos nos mesmos splits;16 comparações pareadas por modelo, cada uma com três sementes. O treino e a calibração também podem mudar, portanto o delta não isola causalmente apenas o truncamento. Os metadados de comprimento do pai permanecem identificados como documentais.

Limiar Youden e FPR1%/5% são calibrados somente na validação de cada protocolo. Reportam-se AUC, BA, TPR, FPR de teste e IC95% condicionais com2.000 sorteios de grupos. Cinco folds compartilham o teste; as sementes reutilizam o corpus. Deltas de protocolo usam os mesmos IDs; macro entre alvos e média entre sementes são descritivas. Os subgrupos IID registram área, fonte humana, gerador e comprimento; células pequenas têm contagens e limites explícitos.

O piloto tem240 documentos exclusivamente do desenvolvimento,20 por classe/área, cobrindo posições na distribuição de comprimento. Registra tempo e memória LM e custo dos ajustes CPU; não escolhe configuração por acurácia no teste. A projeção linear dos LMs não estima com precisão a busca de atributos Ghostbuster, que pode escalar de forma não linear. Reservas iniciais: uma H200/job LM,16 CPUs/128GB RAM por job, até2 jobs GPU por array; o piloto informa consumo real antes de ampliar. Nenhum job usa quantização4/8bits adicionada pelo pipeline.

## Ataques e interpretação

Seis ataques independentes: um separador de parágrafo próximo do centro, caixa, espaço extra, erros ortográficos de dicionário PT, U+200B e homoglifos. As cinco operações esparsas usam doses1%/5% das posições elegíveis; parágrafo tem uma condição: total11. Todos partem do texto N2 limpo. Código, URLs, e-mails e números são protegidos. `no_op` é preservado, inclusive quando a dose arredonda para zero.

As condições são derivadas somente da união dos testes IID congelados. Edições, intensidade efetiva, sementes, hashes e lineage são registrados. Os mesmos25 detectores/configurações (16 novos +9 anteriores), três sementes e fold0 geram75 tarefas de inferência, cada uma cobrindo11 condições. Os modelos CPU anteriores são reconstruídos e precisam reproduzir seus escores documentais; BERT usa seu checkpoint verificado. Atualizações de biblioteca podem impedir essa reprodução: não afrouxe a tolerância para obter aprovação automática.

Avaliamos sintéticos atacados/humanos limpos, humanos perturbados/sintéticos limpos e ambos perturbados. Modelo e limiar permanecem os do treino/validação limpa. São reportados AUC/BA/TPR/FPR, deltas, inversões e evasão condicionada aos sintéticos originalmente detectados. `all_valid` inclui no-ops; `changed_only` é um controle secundário. Amostras inválidas mantêm denominadores e razões. ICs compartilham sorteios entre os textos limpos/alterados.

Estas doses são adaptações esparsas PT-BR, não reprodução literal das intensidades RAID. Não há ataques guiados por consulta ao detector, nem treino adversarial nesta campanha. Paráfrase/sinônimos ficam para a fase semântica posterior, como aprovado; exclusão de artigos, grafia britânica/PT-PT e troca de números não integram a bateria inicial. Não se afirma validação humana independente ou garantia de preservação semântica por consenso de LLMs.

## Comandos em ordem

Publique os arquivos de código no computador local (`git add`, `git commit`, `git push`). Depois do acesso habitual até `ssh abaporu`:

```bash
cd /mnt/recod/homes/victor.roquetto/Benchmark-Horus
git pull --ff-only
source scripts/abaporu_advanced_env.sh
bash scripts/run_advanced_abaporu.sh install
bash scripts/run_advanced_abaporu.sh prepare
bash scripts/run_advanced_abaporu.sh download
bash scripts/run_advanced_abaporu.sh pilot
```

`install` cria o venv separado com `/usr/bin/python3.11`, instala ferramentas, executa somente fixtures pequenos e salva `environment.freeze.txt`. `prepare` verifica os insumos grandes; se faltarem, transfira os arquivos existentes, em vez de recriar o benchmark. `download` busca os8 checkpoints das duplas, tokenizer BERT e Stanza. `pilot` faz a pré-verificação em uma GPU e a estimativa CPU. Consulte os resultados antes de executar a campanha completa:

```bash
cat results/evaluation/n2-advanced-v1/cpu_pilot.json
ls -lh results/evaluation/n2-advanced-v1/cache/pilot/*/shards/*.json
```

Após revisar o piloto:

```bash
bash scripts/run_advanced_abaporu.sh cache
bash scripts/run_advanced_abaporu.sh iid
bash scripts/run_advanced_abaporu.sh ood
bash scripts/run_advanced_abaporu.sh compound
bash scripts/run_advanced_abaporu.sh common
```

As próximas fases geram e pontuam as perturbações. A extração atacada pode ser mais cara do que a limpa, pois há11 variantes por documento do teste; consulte `attack_coverage.csv` e o número de jobs antes de `attack-cache`:

```bash
bash scripts/run_advanced_abaporu.sh attacks
cat results/evaluation/n2-advanced-v1/attacks/attack_coverage.csv
bash scripts/run_advanced_abaporu.sh attack-cache
bash scripts/run_advanced_abaporu.sh robustness
bash scripts/run_advanced_abaporu.sh reports
```

Cada comando aguarda suas etapas dependentes e para se um job falhar. Não há `exit` na shell interativa pai. É conveniente usar `tmux` para manter o lançador ativo; ao reconectar, repetir a mesma fase reutiliza jobs ainda ativos e pula comandos completos. Não execute duas instâncias simultâneas da mesma fase.

```bash
bash scripts/run_advanced_abaporu.sh status
```

Scripts usam `h200`,16CPUs/128GB e24h; `HORUS_ADV_PARTITION`, `HORUS_ADV_QOS`, `HORUS_CPU_CONCURRENT` e `HORUS_GPU_CONCURRENT` permitem ajustar apenas recursos. O limite de simultaneidade é por array (índices locais até399); vários arrays somam suas simultaneidades. Não altere seeds/folds/checkpoints/hiperparâmetros após gerar os planos. Esses arquivos são fixados por hash e mudanças exigem uma nova versão de experimento.

Resultados finais: `summary/` (matriz e subgrupos), `errors/` (concordância), `contrast/<model>/` (prefixo comum) e `robustness/summary/`, sob `results/evaluation/n2-advanced-v1/`. A fase `reports` exige IID, OOD e composto completos; isso evita congelar um sumário parcial e omitir protocolos adicionados posteriormente. As tabelas/figuras/manifests podem ir para o Git; modelos, probabilidades e Parquets permanecem locais.

## Fontes metodológicas

- Thomas, Hoang e Mitchell (ALTA2024), [Simple models are all you need](https://aclanthology.org/2024.alta-1.19/). O artigo não fornece todos os hiperparâmetros; adaptações estão acima.
- Bao et al. (ICLR2024), [Fast-DetectGPT](https://arxiv.org/abs/2310.05130), [código oficial](https://github.com/baoguangsheng/fast-detect-gpt).
- Hans et al. (ICML2024), [Binoculars](https://proceedings.mlr.press/v235/hans24a.html), [código oficial](https://github.com/ahans30/Binoculars).
- Verma et al. (NAACL2024), [Ghostbuster](https://aclanthology.org/2024.naacl-long.95/), [código oficial](https://github.com/vivek3141/ghostbuster).
- Dugan et al. (ACL2024), [RAID](https://aclanthology.org/2024.acl-long.674/), [ataques oficiais](https://github.com/liamdugan/raid/tree/main/generation/adversarial).

Fórmulas e convenções foram conferidas nos arquivos oficiais `Binoculars/binoculars/metrics.py`, `fast-detect-gpt/scripts/fast_detect_gpt.py` e `ghostbuster/utils/{symbolic,n_gram,featurize}.py`. A implementação local e as revisões de modelo têm hashes próprios; não se presume que um branch upstream chamado main seja uma versão imutável.
