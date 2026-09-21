# Validação da entrega

## Condições de pronto

**Software implementado e validado tecnicamente:** fluxo local de coleta, persistência,
versionamento, treinamento, relatórios, inferência, revisão e comparação disponível.
**Experimento real inicial registrado:** a revisão de 19/09/2026 encontrou três experimentos
concluídos e um teste final salvo, posteriores ao estado inicial descrito nesta documentação.
São resultados de uma pessoa na mesma posição; não demonstram generalização entre pessoas
ou condições. A consulta aos registros não equivale a repetir ou certificar a coleta humana.
O repositório não distribui coletas pessoais; uma instalação nova começa sem amostras.

## Revisão do portfólio — 19/09/2026

- Leitura do SQLite existente em modo somente leitura, sem inicializar o Store da aplicação.
- 704 amostras, cinco classes, quatro sessões com dados e uma vazia; duas versões locais.
- Mesma versão de 704 amostras, linhagem, protocolo, estratégia e semente nos três experimentos.
- Treino/validação/teste: 342/198/164; uma avaliação final salva do Forest, conferida entre
  o registro de reserva final e o relatório do experimento.
- Seis casos difíceis pendentes, sem comparação V1/V2 ou avaliação de desafio salva.
- Autor confirmou uma pessoa e mesma posição; independência física das sessões não comprovada.
- Novo exportador agregado não carrega modelos, não reavalia teste final, não abre webcam,
  não escreve no banco nem inclui dados livres da coleta no relatório.

Resultados numéricos: [relatório comparativo](RELATORIO_COMPARATIVO.md) e
[portfolio-results.json](portfolio-results.json).

| Verificação executada nesta revisão | Resultado |
| --- | --- |
| Suíte completa (`python -m pytest -q`) | **56 aprovados**, sendo 29 novos testes do exportador |
| Ruff (`python -m ruff check gesturelab scripts tests`) | Aprovado |
| Compilação dos dois arquivos Python novos (`compileall`) | Aprovada |
| Dependências (`python -m pip check`) | Sem incompatibilidades declaradas |
| JSON agregado (`scripts/export_portfolio_report.py --check docs/portfolio-results.json`) | Coincide com os resultados salvos |
| Integridade do banco durante a conferência | SHA-256 idêntico antes e depois |
| Higiene do diff (`git diff --check`) | Aprovada |

Os testes novos cobrem ausência de dados privados no JSON, reordenação das matrizes,
acesso somente leitura, banco inexistente sem criação, coorte ambígua, números inválidos,
split inconsistente, métricas condicionais sem previsões aceitas, contagens incompatíveis,
avaliação final vazia/divergente/duplicada e detecção de relatório desatualizado
sem sobrescrita. As mutações desses testes ocorrem apenas em bancos sintéticos temporários.
Nenhuma coleta, revisão humana, ativação de modelo, treino, avaliação final ou gravação
de vídeo foi executada sobre os dados reais nesta revisão.
Revisão independente do exportador e da consistência documental concluída sem bloqueios,
após os ajustes e testes de integridade do relatório.

## Evidências históricas executadas — 15/09/2026

Os testes e smokes a seguir pertencem à entrega inicial. Não foram todos repetidos na
revisão de documentação e exportação; em particular, a webcam não foi reaberta.

| Verificação | Resultado |
| --- | --- |
| Testes automatizados | **27 aprovados**, incluindo regressões dos controles de coleta |
| Verificação estática Ruff | Aprovada |
| Compilação dos módulos Python | Aprovada |
| Consistência das dependências (`pip check`) | Sem incompatibilidades declaradas |
| Integridade do detector | SHA-256 conferido |
| CLI de ajuda, validação vazia e instalação editável | Executados com sucesso |
| Janela Qt e oito páginas | Testadas, inclusive estados vazios |
| Treino pela interface e relatório por classe | Testado com Store real e dados sintéticos temporários |
| Revisão humana, desenho e exclusão com cancelar/confirmar | Testados com dados sintéticos temporários |
| Treino, persistência, recarga e inferência dos quatro modelos | Testados com dados sintéticos temporários |
| Orçamento equivalente aleatórios × difíceis | Testado, mesma validação e sem consulta ao teste final |
| Liberação de conexões SQLite | Teste específico de fechamento de handles no Windows |

Os dados sintéticos só verificam o software. Nenhuma métrica desses testes é apresentada
como qualidade real de reconhecimento no README, no model card ou na tela inicial.

## Webcam e hardware — ensaio histórico de 15/09/2026

Windows 10 x64 10.0.19045; Intel Core i7-7700HQ, 4 núcleos / 8 threads; Python 3.13.14.
HD WebCam, índice 0, resolução solicitada e recebida 640 × 480, processamento CPU.

- **Captura pura:** 60 frames em cada uma de duas aberturas, cerca de **29,7 FPS**.
  Mede leitura de câmera, não classificação de gestos.
- **Detector em imagem preta:** 5 aquecimentos, 60 medições com `perf_counter`, mediana
  **16,08 ms**, percentil 95 **30,70 ms**. Zero mãos. Não mede desempenho sobre mãos reais.
- **Worker real:** duas aberturas, 45 frames cada, encerramento confirmado e sem erros.
  O índice 99 produziu erro compreensível de câmera indisponível, sem deixar thread ativa.
- **Janela com câmera física:** abriu Coleta, parou, reabriu Ao vivo e fechou com captura
  ativa. Recursos liberados, zero amostras salvas. O detector produziu leituras sem rótulo;
  não houve comparação com verdade humana nem classificação treinada nesse smoke.

Relatórios completos, com método e contexto:

- [runtime-validation.json](runtime-validation.json).
- [camera-worker-validation.json](camera-worker-validation.json).
- [ui-hardware-validation.json](ui-hardware-validation.json).

O FPS mostrado ao vivo usa suavização de intervalos entre frames e varia com a carga da
máquina; não equivale ao FPS de classificação em um benchmark. Extração, classificação
e atraso da suavização são mostrados separadamente. A coleta real precisa medir latência
com o classificador escolhido e registrar o cenário.

## Marcos entregues

| Marco | Implementação e conceito | Validação | Etapa humana |
| --- | --- | --- | --- |
| A — Base | Ambiente virtual, OpenCV, Hand Landmarker, worker Qt; detector pré-treinado | Importações, hash, webcam real e encerramento | Conferir uma e duas mãos, ocupação por outro aplicativo |
| B — Dados | Sessões, rótulos, pontos, SQLite, shards Parquet, versões; diversidade e integridade | Inválidos, revisão, exclusão, hash e histórico; 704 amostras reais registradas | Coletar com diversidade e independência documentadas |
| C — ML | Atributos, quatro modelos, grupos, validação/teste, relatórios; generalização | Testes sintéticos e resultados reais salvos de três modelos | Validar generalização em nova coleta independente |
| D — Demonstração | Pontuação, incerteza, consenso temporal e marcador interno; latência e estabilidade | Neutro, cooldown, estados e janela com webcam | Testar gestos semelhantes e novos gestos |
| E — Experimentação | Revisão de erros, V2 e orçamento equivalente; hipótese controlada | Rótulos humanos, IDs congelados, braços equivalentes | Comparar V1/V2 com dados reais |
| F — Portfólio | Interface, case, roteiro e resultados agregados; comunicação de evidências | Captura histórica da interface e relatório conferido com o banco | Gravar demonstração real e decidir publicação |

## Decisões e correções verificadas

- MLP sem `early_stopping` interno: evita divisão escondida por frames.
- Protocolo e amostras de validação/teste congelados; trocar semente ou snapshot não
  libera o teste. Trocar protocolo exige uma coleta independente em outro diretório.
- Uma consulta final por linhagem; repetir o mesmo experimento reutiliza o relatório.
- Cancelamento antes de publicar; modelos cancelados/falhos não aparecem como válidos.
- Revisão incorpora uma única amostra com rótulo humano, mantendo extrator e instante.
- Conexões SQLite fechadas explicitamente; transação por si só não fecha o handle.
- Ao perder a mão, a interface limpa a previsão e rearma a demonstração; o worker
  só é descartado depois de sinalizar encerramento.

## Não validado e limites restantes

1. Generalização dos cinco gestos para outras pessoas e condições; a V1 é restrita a uma pessoa.
2. Desafio de gestos desconhecidos e comparação V1/V2 reais. Métricas da V1 já estão documentadas.
3. Duas mãos reais e câmera ocupada por outro aplicativo, com julgamento humano.
4. Diversidade de participantes, luz, oclusões, inclinação e mudança de distância.
5. Distribuição como executável instalável e outros sistemas operacionais; esta entrega
   usa o ambiente Python local. A publicação do código não inclui um executável instalável.
6. Gravação de demonstração real, instalação limpa atual e ausência total de telemetria da
   dependência MediaPipe; a execução local, sozinha, não comprova ausência de tráfego.

Cancelamento é cooperativo e pode aguardar um ajuste/fold curto em andamento. O teste final
é uma proteção de protocolo local, não uma barreira contra alguém editar o banco manualmente.
Dados numéricos e pseudônimos continuam sendo dados que merecem cuidado ao compartilhar.

## Reproduzir

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check gesturelab scripts tests
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/export_portfolio_report.py --check docs/portfolio-results.json
.\.venv\Scripts\python.exe scripts/probe_runtime.py --camera 0
.\.venv\Scripts\python.exe scripts/probe_camera_worker.py
.\.venv\Scripts\python.exe scripts/probe_ui_hardware.py
.\.venv\Scripts\python.exe scripts/capture_ui.py
```

Os três comandos de câmera abrem o dispositivo explicitamente. Os dois últimos relatórios
de smoke são atualizados pelos scripts, sem armazenar frames. A captura visual documenta
somente a interface inicial, com cinco categorias, zero amostras e nenhum classificador.
