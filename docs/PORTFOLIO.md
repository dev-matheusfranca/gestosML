# GestureLab no portfólio

Material preparado em 19/09/2026; acesso ao código conferido em 21/09/2026. O
[repositório está público](https://github.com/dev-matheusfranca/gestosML). Esta revisão não
altera sua visibilidade nem o site do portfólio e não produz um vídeo de reconhecimento.
Texto e roteiro podem ser usados depois de revisar a apresentação e a privacidade da gravação.

## Texto curto para o card

**GestureLab — laboratório desktop de machine learning**

Aplicação em Python para aprender o ciclo de reconhecimento de gestos: coletar exemplos
pela webcam, comparar classificadores e revisar erros. Inclui versionamento de datasets,
avaliação separada por sessão e demonstração ao vivo. Experimento inicial com 704 amostras,
cinco classes e três modelos; coleta de uma pessoa na mesma posição, sem generalização
comprovada para outras pessoas.

**Tecnologias:** Python, PySide6, OpenCV, MediaPipe, scikit-learn, SQLite e Parquet.

**Categoria:** projeto experimental educacional. Não apresentar como produto em produção
ou solução de acessibilidade validada. O card pode apontar para o repositório público;
não oferecer um botão de demonstração enquanto não houver vídeo ou aplicação acessível.

## Case: o que apresentar em uma entrevista

### Problema e público

Quem aprende machine learning precisa entender mais do que o número final de acertos:
a origem dos exemplos, a separação dos dados e o efeito de rejeitar previsões incertas.
O laboratório reúne essas etapas numa interface local para experimentação educacional.

### Solução implementada

- Webcam contínua com coleta supervisionada e controles de pausa/finalização.
- Pontos da mão extraídos pelo MediaPipe; classificadores próprios treinados com os rótulos.
- Persistência local, snapshots e divisão por grupos para reduzir vazamento entre partições.
- Referência Dummy, regressão logística, Random Forest e suporte a MLP.
- Relatórios por classe, matriz de confusão, cobertura e microbenchmark do classificador.
- Inferência ao vivo e revisão humana de exemplos difíceis.
- Exportação agregada de resultados, sem distribuir coletas ou modelos pessoais.

### Resultado medido e aprendizado

A primeira comparação real usou 704 amostras de cinco classes, separadas por sessão:
342 para treino, 198 para validação e 164 para teste. Os três modelos efetivamente executados
foram Dummy, regressão logística e Random Forest.

Na validação, a logística teve 96,97% de acurácia global, contra 89,90% do Forest e 19,19%
da referência Dummy. O Forest alcançou 100% entre as previsões aceitas, mas aceitou apenas
63,13% dos exemplos. O autor relatou ter escolhido o Forest pela maior porcentagem de
acertos; a análise evidencia por que esse percentual precisa ser lido junto da cobertura.

O teste final salvo do Forest teve 96,34% de acurácia global, cobertura de 78,05% e
128 acertos entre 128 previsões aceitas, com 36 rejeitadas. Não resumir isso como “100%
de precisão”. Não alegar que o Forest superou a logística no teste: ela não foi avaliada nele.

**Aprendizado central:** maior acerto entre previsões aceitas não significa melhor modelo
para qualquer finalidade. Dados, cobertura e objetivo precisam participar da decisão.

### Limitações assumidas

Uma única pessoa, mesma posição, sem diversidade física documentada entre as quatro sessões.
Não demonstrado: generalização para outras pessoas, gestos desconhecidos, ganho V1 × V2 ou
ausência total de telemetria da dependência MediaPipe. Há seis exemplos difíceis ainda sem
revisão. Esses limites fazem parte do case, não são detalhes a esconder.

Fontes: [comparação](RELATORIO_COMPARATIVO.md), [dataset card](DATASET_CARD.md),
[model card](MODEL_CARD.md), [JSON agregado](portfolio-results.json) e [validação](VALIDACAO.md).

## Vídeo real de 75 segundos — pendente de gravação

| Tempo | Imagem a gravar | Mensagem |
| --- | --- | --- |
| 0–10 s | Tela inicial e objetivo | “Um laboratório local para aprender o ciclo de machine learning com gestos.” |
| 10–25 s | Webcam real, escolha de rótulo, contagem e pausa | “O MediaPipe extrai pontos; os exemplos rotulados treinam os classificadores.” |
| 25–35 s | Dataset e divisão por sessão | “São 704 amostras nesta V1, de uma pessoa na mesma posição. A divisão não prova robustez entre pessoas.” |
| 35–50 s | Comparação salva e matriz de confusão | “A logística teve maior acerto global; o Forest aceitou menos previsões e acertou mais dentro desse subconjunto.” |
| 50–65 s | Inferência real, incluindo falha ou estado incerto observado | “Rejeitar é diferente de acertar. Gestos desconhecidos ainda não foram avaliados sistematicamente.” |
| 65–75 s | Lista de exemplos difíceis | “Há seis casos pendentes. Revisão e uma coleta mais diversa são os próximos passos.” |

### Preparação da gravação

1. Usar captura de tela local; ocultar caminhos, pseudônimos, notificações e informações privadas.
   Enquadrar somente a mão, sem expor outras pessoas.
2. Para demonstrar uma **nova coleta**, usar outro diretório, por exemplo:
   `.\.venv\Scripts\python.exe -m gesturelab --data-dir data/demo-portfolio gui`.
   Esse diretório fica sob `data/`, já ignorado pelo Git. Não modificar o dataset V1 para montar o vídeo.
3. Fechar essa instância e abrir a coleta original para mostrar resultados salvos e inferência.
   Explicar o corte: a demonstração de coleta é separada do experimento documentado.
4. Selecionar o Forest concluído da V1 quando demonstrar esse modelo; não clicar em avaliar
   novamente, não retreinar e não alterar limiares para simular o resultado do relatório.
5. Mostrar uma limitação que realmente acontecer. Se não ocorrer uma falha na gravação, usar
   a matriz salva e dizer que é um erro registrado, sem encená-lo como previsão da webcam.
6. Na V1, apenas mostrar a revisão pendente. Uma correção efetiva exige rótulo humano e altera
   os dados; não afirmar que houve melhoria medida sem executar um experimento controlado.

Não substituir a webcam por animação sintética, não fabricar falhas ou acertos e não inserir
um link fictício. O screenshot inicial do README mostra a interface, não prova reconhecimento.

## Rascunho para LinkedIn

Desenvolvi o GestureLab, um laboratório desktop em Python para explorar coleta de dados,
comparação de classificadores e revisão de erros no reconhecimento de gestos.

O MediaPipe extrai os pontos da mão; os classificadores aprendem com os exemplos rotulados.
Na primeira comparação real, registrei 704 amostras e comparei Dummy, regressão logística e
Random Forest com a mesma divisão por sessão.

O principal aprendizado foi interpretar os números: 100% de acerto entre previsões aceitas
não é 100% de reconhecimento geral. O Forest acertou as 128 previsões aceitas no teste salvo,
mas rejeitou 36 das 164 entradas; sua acurácia global foi de 96,34%.

A coleta foi feita somente por mim, na mesma posição. O resultado é um experimento inicial,
não uma prova de desempenho com outras pessoas. Os próximos passos são uma demonstração
real gravada e uma coleta mais diversa, com novo conjunto final independente.

### Antes de publicar o case ou vídeo

- [ ] Gravar e revisar o vídeo real, incluindo limitações.
- [ ] Conferir números com o relatório e não usar “100%” sem cobertura e escopo.
- [ ] Revisar cada frase em primeira pessoa para representar o trabalho e aprendizado do autor,
      inclusive o uso de ferramentas de assistência quando relevante.
- [x] Confirmar o acesso público ao código (21/09/2026).
- [ ] Não publicar banco, snapshots, modelos, pseudônimos, coordenadas ou vídeos privados.
- [ ] Não prometer ausência de telemetria enquanto a dependência não for validada.
