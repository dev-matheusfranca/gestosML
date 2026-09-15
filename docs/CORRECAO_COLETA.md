# Controles de coleta — 15/09/2026

## Sintoma e causa confirmada

A câmera mostrava pontos da mão, mas iniciar/pausar/finalizar pareciam não produzir efeito.
Cada frame atualizava o mesmo QLabel da contagem e apagava o aviso da operação. O início
também exigia um gesto selecionado; sem seleção, apenas a barra inferior mostrava um aviso
temporário. A consulta de diagnóstico encontrou zero sessões e zero amostras no banco do usuário.

## Alteração

- Separação entre detecção da câmera e estado da coleta.
- Instruções persistentes para escolher o gesto e iniciar a contagem.
- Contagem, pausa e término continuam visíveis durante a chegada dos frames.
- Botões habilitados conforme câmera, seleção, contagem e sessão existentes.
- Trocar o gesto pausa a captura de exemplos e exige nova contagem.
- Finalizar mantém o número salvo visível; nova sessão começa com contador zerado.
- Falha de gravação pausa a coleta e deixa aviso persistente.

**A prévia continua ativa depois de pausar ou finalizar.** Para desligar a webcam é
necessário clicar em Fechar. As amostras são gravadas à medida que o contador aumenta;
Finalizar não é um botão adicional de salvar.

## Evidência

Novos testes de regressão reproduziram mensagens apagadas antes da alteração. Depois,
validaram cliques de iniciar/pausar/retomar/finalizar, frames concorrentes, conservação
dos rótulos e amostras em SQLite temporário e instrução para ausência de gesto.
Não foram criados dados de treino no diretório do usuário.

O fluxo humano de apresentação de gestos reais ainda precisa ser repetido na janela
atualizada. O teste automatizado não mede a qualidade do reconhecimento.

## Achado independente nos logs

A distribuição binária MediaPipe 0.10.35 tentou enviar estatísticas via Clearcut.
As linhas observadas reportam falha de envio; não comprovam envio de imagens ou de pontos.
Não há opt-out oficial na BaseOptions dessa versão. Isso contraria o requisito original
de ausência de telemetria; a afirmação anterior de ausência total não está validada.

Uma alternativa investigada é um ambiente separado Python 3.12 / MediaPipe 0.10.21 /
NumPy abaixo de 2, ou build a partir do código aberto. O wheel antigo inspecionado não
contém os marcadores de uploader encontrados no atual, e seu TaskRunner não cria
TasksLogger. Ainda seria necessário validar dependências, câmera e comportamento de rede.
Não foram alteradas dependências nem regras de rede nesta correção dos controles.

Fontes oficiais: [discussão sobre telemetria](https://github.com/google-ai-edge/mediapipe/issues/6291),
[TaskRunner 0.10.21](https://github.com/google-ai-edge/mediapipe/blob/v0.10.21/mediapipe/tasks/cc/core/task_runner.cc),
[TaskRunner 0.10.35](https://github.com/google-ai-edge/mediapipe/blob/v0.10.35/mediapipe/tasks/cc/core/task_runner.cc).
