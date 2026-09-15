# Relatório comparativo

**Pendente de coleta real.** Testes sintéticos validam software; não medem reconhecimento.

## Comparação controlada

Preencher somente com experimentos concluídos e metodologicamente compatíveis. Relatar
diferenças de versão, classes, divisão ou protocolo em vez de escolher o maior número.

| Modelo | Dataset / divisão | Acurácia | Macro-F1 | Cobertura | F1 aceito | Latência |
| --- | --- | --- | --- | --- | --- | --- |
| Dummy | Pendente | — | — | — | — | — |
| Regressão logística | Pendente | — | — | — | — | — |
| Random Forest | Pendente | — | — | — | — | — |
| MLP | Pendente | — | — | — | — | — |

Anexar matriz de confusão, suporte por classe e relatório exportado do experimento.
Explicar confusões específicas e não apenas a média global.

## V1 × V2

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

Ainda não há evidência para afirmar qual modelo reconhece melhor gestos reais nem que a V2
melhora a V1. Registre essa conclusão após coleta e avaliação independente.
