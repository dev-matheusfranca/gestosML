# Dataset card — coleta V1

**Estado conferido em 19/09/2026:** 704 amostras reais locais, cinco classes e quatro sessões
com dados. Os testes automatizados usam dados sintéticos temporários, separados desta coleta.
Fonte numérica: [relatório agregado](portfolio-results.json). Condições humanas: relato do autor.

## Origem e finalidade

Dados produzidos localmente pela webcam, com rótulo atribuído pelo operador. Objetivo:
aprender coleta supervisionada, comparação de classificadores e investigação de erros.
Não é um dataset de língua de sinais, identidade, comportamento ou biometria autenticadora.

## Características conhecidas

| Campo | Evidência disponível |
| --- | --- |
| Amostras / classes | 704 / 5 |
| Pessoas | Uma, segundo o autor; não inferido dos pseudônimos |
| Condições | Mesma posição, segundo o autor |
| Sessões | Quatro com dados (132, 164, 198 e 210 amostras), mais uma vazia |
| Propósito das sessões com dados | Desenvolvimento; todas contêm as cinco classes |
| Versões locais | Dois snapshots, com 132 e 704 amostras; comparação usa o de 704 |
| Extrator registrado | MediaPipe 0.10.35, uma versão de extrator na coleta |
| Divisão persistida | Treino: 342 amostras / 2 sessões; validação: 198 / 1; teste: 164 / 1 |
| Revisão de casos difíceis | Seis pendentes; nenhum revisado ou descartado |
| Desafio de gestos desconhecidos | Sem avaliação salva |

Os IDs e hashes dos snapshots permanecem no histórico local, fora do relatório público.
As classes aparecem como `class_1` a `class_5` no relatório, substituindo rótulos livres;
a ordem é consistente entre modelos e entre os dois eixos das matrizes.

**Limite principal:** separar por ID de sessão evita misturar esse ID entre partições,
mas não torna a coleta diversa. O autor informou mesma pessoa e posição; pausas entre
rodadas, distância, iluminação e independência física não foram documentadas suficientemente.
Não apresentar esta divisão como avaliação em pessoas novas ou ambientes diferentes.

## Estrutura e privacidade

- ID da amostra e sessão, pseudônimo opcional, categoria e instante da captura.
- Lateralidade retornada pelo extrator; 21 pontos × 3 coordenadas antes das transformações.
- Versão do extrator e informações geométricas de qualidade disponíveis.
- Propósito da sessão: desenvolvimento ou desafio.
- Metadados locais em SQLite; snapshots em Parquet; hashes de integridade.

A aplicação não armazena imagens/vídeos nem faz upload das coletas nesta versão.
Pseudônimos e coordenadas ainda merecem proteção. `data/` é ignorado pelo Git; somente
métricas agregadas e aliases são incluídos na documentação. A alegação sobre coletas
locais não elimina a [pendência de telemetria do MediaPipe](CORRECAO_COLETA.md).

## Ainda não documentado ou validado

- Resolução e câmera efetivas durante esta coleta (o smoke técnico a 640 × 480 é outro ensaio).
- Período detalhado, pausas, lateralidade predominante, diversidade de luz e distância.
- Revisão humana integral dos rótulos, descartes e duplicatas desta versão.
- Generalização entre pessoas e robustez sob oclusões ou mudança de posição.

Para ampliar o experimento, seguir o [protocolo](PROTOCOLO.md), registrar as condições e
usar coleta independente para uma nova avaliação final. Não inventar diversidade retroativa
nem dividir uma gravação contínua em novas sessões para satisfazer a contagem mínima.

Excluir o conjunto atual não apaga snapshots e modelos históricos. Uma eliminação integral
exige revisar também versões, exportações e artefatos derivados, sem publicá-los por engano.
