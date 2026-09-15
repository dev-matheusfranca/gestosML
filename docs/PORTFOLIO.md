# Materiais de portfólio — rascunhos locais

Nada foi publicado. Antes de publicar, substitua campos pendentes por evidências reais e
revise o enquadramento para não expor pessoas ou informações privadas.

## Vídeo de 75 segundos

| Tempo | Imagem a gravar | Mensagem |
| --- | --- | --- |
| 0–10 s | Tela inicial e objetivo | “Um laboratório local para entender como um classificador aprende gestos.” |
| 10–25 s | Coleta real: webcam e 21 pontos | “O MediaPipe encontra pontos; eu atribuo os rótulos dos gestos para treinar os classificadores.” |
| 25–35 s | Dataset por sessão | “Os dados são separados por sessão para reduzir vazamento entre treino e avaliação.” |
| 35–50 s | Comparação real de modelos e matriz | Mostrar resultados reais e uma confusão concreta. Se ainda não houver dados, dizer “avaliação pendente”. |
| 50–65 s | Demonstração real e estado incerto | Mostrar reconhecimento e uma falha; explicar pontuação e suavização. |
| 65–75 s | Revisão de exemplo difícil | “Corrigir rótulos e coletar dados melhores é parte do experimento. Melhoria precisa ser medida.” |

Não use animação de pontos sintéticos como se fosse reconhecimento pela webcam.

## Rascunho para LinkedIn

Desenvolvimento em andamento: um laboratório desktop de reconhecimento de gestos com Python,
OpenCV, MediaPipe, scikit-learn e PySide6.

A proposta é tornar visível o ciclo de machine learning: coletar exemplos, revisar rótulos,
treinar modelos, avaliar em sessões independentes e investigar erros.

O MediaPipe extrai os pontos da mão. Os classificadores são treinados com os dados coletados
no laboratório. O fluxo inclui uma referência Dummy, regressão logística, Random Forest e
uma rede MLP, além de revisão humana de exemplos difíceis.

Neste estágio, a validação do software deve ser distinguida da qualidade do reconhecimento.
Resultados com gestos reais: **pendentes de coleta e avaliação independente**.

Antes da publicação, acrescentar: resultado medido, limitações observadas, vídeo real e link
do repositório, caso seja publicado por decisão posterior. Não atribuir experiência pessoal
ou resultados que não estejam documentados.
