# Model card — classificadores GestureLab

**Estado em 19/09/2026:** três experimentos reais concluídos e um teste final salvo do
Random Forest. Conferência por leitura do banco, sem retreino ou nova avaliação.
Fonte agregada: [portfolio-results.json](portfolio-results.json).

## Uso previsto

Laboratório educacional e demonstração local de gestos estáticos. Não usar para autenticação,
acessibilidade crítica, tradução de língua de sinais, decisão médica ou controle de dispositivos.
Não há demonstração de generalização entre pessoas: a coleta foi de uma pessoa na mesma posição.

## Dois componentes distintos

1. **Hand Landmarker:** detector/extrator pré-treinado do MediaPipe, executado em CPU.
2. **Classificador GestureLab:** treinado com exemplos rotulados pelo usuário; recebe atributos
   dos pontos. Dummy, regressão logística e Random Forest têm resultados reais nesta V1.
   MLP está implementado, mas não foi executado nesta comparação real.

Normalização, seleção de atributos e transformações estatísticas fazem parte do pipeline
persistido. O mesmo processamento é usado na inferência. [Especificação dos atributos](core.md).

## Configuração observada

- Mesma versão de 704 amostras, cinco classes, linhagem, protocolo `session` e semente `42`.
- Atributos `normalized`, preset `quick`; 342 exemplos de treino, 198 de validação e 164 de teste.
- Dummy: estratégia `prior`, limiar 0,0.
- Regressão logística: `C=3.0`, `solver=lbfgs`, `max_iter=800`, limiar 0,7.
- Random Forest: 120 árvores, profundidade sem limite, mínimo de uma amostra por folha,
  `class_weight=balanced`, `n_jobs=1`, limiar 0,8.
- Hiperparâmetros selecionados na validação. O limiar maximiza macro-F1 entre previsões
  aceitas, com cobertura mínima de 60% na validação; empate favorece maior cobertura.
  É a regra do software, não uma garantia de cobertura em dados novos.

## Resultado e seleção

| Modelo | Acurácia de validação | Macro-F1 | Cobertura | Acurácia entre aceitas |
| --- | --- | --- | --- | --- |
| Dummy | 19,19% | 0,0644 | 100,00% | 19,19% |
| Regressão logística | 96,97% | 0,9700 | 94,95% | 98,40% |
| Random Forest | 89,90% | 0,8979 | 63,13% | 100,00% |

O autor relatou ter escolhido o Forest pela **maior porcentagem de acertos**. O maior
percentual registrado para ele na validação é o acerto **entre previsões aceitas**:
125 acertos em 125 aceitas, mas 73 de 198 exemplos foram rejeitados. Não foi registrado
um requisito prévio de produto que justificasse preferir menor cobertura; não se deve
reconstruir essa justificativa depois do resultado. O relato não especifica qual indicador
da interface foi consultado no momento da escolha.

Teste final único salvo do Forest: 164 exemplos, **96,34% de acurácia global**, macro-F1
**0,9646**, cobertura **78,05%**; **128 acertos entre 128 aceitas**, **36 rejeitadas**.
A acurácia global usa a classe prevista antes da rejeição e não é a taxa de ações corretas
entregues pela interface. O resultado final não permite comparar Forest e logística no teste.

Relatório por classe e matrizes estão no JSON, com aliases estáveis no recorte exportado.
A [comparação detalhada](RELATORIO_COMPARATIVO.md) inclui tempos, confusões e interpretação.

## Limitações e próximos ensaios

- Uma pessoa, mesma posição, quatro IDs de sessão: não prova robustez entre pessoas,
  iluminação, distância, orientação ou sessões fisicamente independentes.
- Nenhuma avaliação de desafio salva; gestos desconhecidos podem receber pontuação alta.
- Seis exemplos difíceis pendentes de revisão; não existe ganho V1 × V2 demonstrado.
- Probabilidades não calibradas; rejeitar mais previsões pode elevar a acurácia condicional.
- Suavização melhora continuidade visual e acrescenta atraso; não altera estas métricas
  individuais. O microbenchmark do classificador não mede a latência completa da webcam.
- Uma consulta ao teste final já está registrada. Não reutilizar esse conjunto para novas
  escolhas ou alegar uma segunda avaliação independente; planejar nova coleta separada.

Modelos locais usam joblib/pickle. Não carregue arquivos de terceiros como confiáveis.
Histórico e hash conferem integridade, não procedência. Modelos e dados pessoais não são
versionados. O MediaPipe 0.10.35 tem uma [pendência de telemetria](CORRECAO_COLETA.md);
processamento local não significa ausência comprovada de tráfego.
