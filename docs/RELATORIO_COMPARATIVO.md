# Relatório comparativo

**V1 — resultados salvos, conferidos em 19/09/2026.** Não houve retreino ou nova consulta
ao teste final nesta revisão. [Fonte agregada](portfolio-results.json).

## Comparação controlada

Três experimentos concluídos usam a mesma versão de 704 amostras, cinco classes, linhagem,
estratégia `normalized`, protocolo `session` e semente `42`. São 342 exemplos de treino
(duas sessões), 198 de validação (uma sessão) e 164 reservados para teste (uma sessão).
O autor confirmou **uma única pessoa, na mesma posição**. A separação por sessão está
registrada, mas sua diversidade e independência física não foram demonstradas.

### Validação — os mesmos 198 exemplos

| Modelo | Acurácia global | Macro-F1 | Cobertura | Acerto entre aceitas | Limiar | Classificador¹ | Treino |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dummy | 19,19% | 0,0644 | 100,00% | 19,19% (38/198) | 0,0 | 0,31 ms | 0,063 s |
| Regressão logística | 96,97% | 0,9700 | 94,95% | 98,40% (185/188) | 0,7 | 0,47 ms | 0,180 s |
| Random Forest | 89,90% | 0,8979 | 63,13% | 100,00% (125/125) | 0,8 | 14,56 ms | 1,776 s |

¹ Média local de 20 chamadas de `predict_proba` para uma linha de atributos. Não inclui
captura, detecção MediaPipe, desenho, suavização ou latência fim a fim. Tempos não constituem
benchmark em outras máquinas. MLP não foi executado neste experimento real.

**Como ler:** acurácia global considera a classe prevista em todos os exemplos, antes da
rejeição. Cobertura é a proporção aceita pelo limiar. Acerto entre aceitas só considera esse
subconjunto. O Forest rejeitou 73 exemplos; a logística rejeitou dez. Portanto, comparar
apenas 100% contra 98,40% esconde uma diferença importante de cobertura.

### Escolha do modelo

O autor informou ter escolhido o Random Forest pela “maior porcentagem de acertos”. Não há
registro do indicador exato consultado nem de uma exigência prévia de cobertura mínima.
O maior percentual do Forest na validação se verifica **entre as previsões aceitas**,
não na acurácia global. Isso explica um possível atrativo da escolha, mas não comprova que
foi esse o indicador consultado. Não atribuímos retrospectivamente uma política de produto
ou uma seleção sistemática que não foi documentada.

No desenvolvimento observado, a logística liderou acurácia global, macro-F1, cobertura e
velocidade; o Forest liderou acurácia e macro-F1 condicionais à aceitação. Um novo experimento
deve declarar antes a prioridade e a cobertura aceitável. Não é necessário esconder esse
aprendizado para apresentar o projeto no portfólio.

### Teste final — somente Random Forest

O registro final corresponde ao mesmo experimento Forest e coincide com o relatório salvo
no histórico. Não foi reexecutado nem usado para avaliar outro modelo nesta revisão.

| Medida | Resultado |
| --- | --- |
| Exemplos | 164 |
| Acurácia global | 96,34% (158/164, antes da rejeição) |
| Macro-F1 | 0,9646 |
| Cobertura | 78,05% (128/164) |
| Acerto entre aceitas | 100,00% (128/128) |
| Rejeitadas | 36 |

**Não é um reconhecimento geral de 100%.** Tampouco é comparação final entre modelos:
Dummy e logística não foram avaliados nesse teste. Uma única pessoa e condição limitam a
conclusão. Melhor resultado no teste do que na validação não comprova generalização robusta.

### Confusões e desempenho por classe

O JSON inclui precisão, recall, F1, suporte e matrizes antes da rejeição. Os rótulos foram
substituídos por aliases; linhas representam a classe real e colunas a classe prevista.

- Na validação, a logística confundiu seis exemplos de `class_5` com `class_3`; as outras
  quatro classes tiveram recall 1,0 nesse conjunto.
- O Forest teve a maior dificuldade em `class_5`: recall 0,6190 (26 de 42), incluindo
  15 previsões como `class_3` e uma como `class_1`.
- No teste final do Forest, ocorreram quatro confusões de `class_1` para `class_3` e duas
  de `class_4` para `class_2`. Não inferir a causa visual sem revisão humana dos exemplos.

### Conferência e privacidade

`scripts/export_portfolio_report.py` lê SQLite com `mode=ro`, `query_only` e uma transação
de leitura. Não inicializa `Store`, não carrega Parquet/joblib, não abre câmera e não faz
avaliação ou treino. Permite apenas uma comparação concluída com mesma versão, linhagem,
estratégia, protocolo e semente, e uma execução por modelo. Recusa ambiguidades e inconsistências.

O recorte atual exige que as linhas do snapshot correspondam a treino + validação + teste;
snapshots que incluam também dados de desafio ficam fora deste exportador. A presença de
uma avaliação de desafio é sinalizada, mas suas métricas não são publicadas por este script.
Contagens da coleta atual ficam separadas das métricas históricas do snapshot.

Saem apenas valores numéricos permitidos, tags fixas e aliases de classe; não saem IDs,
pseudônimos, caminhos, datas de captura, rótulos livres, pontos ou artefatos de modelos.
Esse resumo reduz a exposição, mas não é uma técnica de privacidade diferencial.

```powershell
.\.venv\Scripts\python.exe scripts/export_portfolio_report.py
.\.venv\Scripts\python.exe scripts/export_portfolio_report.py --check docs/portfolio-results.json
```

O primeiro comando imprime JSON; o segundo compara, sem sobrescrever. Uma nova instalação
não inclui o banco pessoal. A conferência reproduz a extração de resultados registrados,
**não a coleta humana, o treinamento nem uma certificação independente dessas métricas**.

## V1 × V2

**Pendente:** há seis exemplos difíceis aguardando revisão, nenhum ganho medido. O teste
final da V1 já foi consultado; não pode orientar novas escolhas. Para uma nova conclusão
final, planejar dados independentes e outro diretório de coleta conforme o protocolo.

Hipótese: exemplos difíceis revisados podem melhorar generalização. Registrar:

- Avaliação de desenvolvimento congelada e IDs.
- Base comum de treino; quantidade adicionada; revisão e origem dos exemplos.
- Mesmas classes e estratégia; modelos e parâmetros controlados.
- Sementes executadas e resultados de todas as tentativas.
- Classes que melhoraram e pioraram; cobertura e latência.

## Experimento opcional: aleatórios × difíceis

Compare o mesmo orçamento de amostras adicionais, sem sobreposição com validação/teste,
mantendo a avaliação fixa. Repita com várias sementes quando houver dados suficientes.
Não compare adicionar 100 exemplos aleatórios com 20 difíceis sem controlar o orçamento.
Resultado negativo também é um resultado. Mantenha o teste final fora desse ciclo.

### Executar o experimento opcional

Use `python -m gesturelab.cli --data-dir data split dataset_SEU_ID` para identificar as
partições persistidas. Selecione IDs somente de **treino**, com todas as classes na base.
O pool difícil precisa ter origem em revisão humana (`quality.source=difficult_review`).
Crie um JSON local, substituindo os exemplos abaixo por IDs reais:

```json
{
  "dataset_id": "dataset_SEU_ID",
  "base_ids": ["sample_BASE_1", "sample_BASE_2"],
  "random_ids": ["sample_ALEATORIO_1", "sample_ALEATORIO_2"],
  "difficult_ids": ["sample_REVISADO_1", "sample_REVISADO_2"],
  "budget": 2,
  "seeds": [11, 42, 73],
  "strategy": "normalized",
  "model": "logistic",
  "protocol": "session"
}
```

Execute `python -m gesturelab.improvement experimento.json --data-dir data`.
São avaliados base, base+aleatórios e base+difíceis para cada semente, com configuração
fixa do classificador e os mesmos IDs de validação. Cada braço adicional recebe exatamente
o orçamento solicitado. O relatório completo é salvo em `data/improvement-reports/`.
Nenhum modelo é ativado automaticamente e o teste final não é consultado.

## Conclusão

Há evidência local de aprendizado acima da referência Dummy e de um trade-off entre
acerto condicional e cobertura. A logística lidera as métricas globais da validação;
o Forest tem um teste final salvo, de escopo limitado. Não há evidência de robustez entre
pessoas, rejeição confiável de desconhecidos nem melhoria V1 × V2. O valor do case é mostrar
o ciclo completo e interpretar seus limites, sem transformar um ensaio restrito em produto validado.
