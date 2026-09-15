# Componentes de terceiros

O código do GestureLab está sob MIT. Essa licença não substitui as licenças de
dependências, binários, fontes ou modelos. Versões exatas estão em `requirements.lock`.
Inventário conferido nos metadados dos pacotes instalados nesta entrega.

| Componente | Licença declarada |
| --- | --- |
| MediaPipe | Apache-2.0 |
| OpenCV contrib / empacotamento Python | Apache-2.0; componentes incluídos têm avisos próprios |
| NumPy | BSD-3-Clause, 0BSD, MIT, Zlib e CC0-1.0 nos componentes distribuídos |
| pandas | BSD-3-Clause |
| scikit-learn / joblib | BSD-3-Clause |
| SciPy | BSD-3-Clause e avisos das bibliotecas numéricas distribuídas |
| PySide6 / Qt | LGPL-3.0-only ou GPL-2.0-only ou GPL-3.0-only conforme componente/licenciamento |
| Matplotlib | Licença própria baseada na PSF; avisos adicionais de componentes |
| Apache Arrow / pyarrow | Apache-2.0 |
| pytest | MIT |

Os textos completos vêm com as distribuições em `.venv/Lib/site-packages`, incluindo
diretórios `.dist-info/licenses`, arquivos `LICENSE*` e os avisos do Qt. A aplicação usa
as bibliotecas instaladas separadamente; esta entrega não gera um executável monolítico.
O arquivo de dependências inclui componentes transitivos além da tabela resumida.

## Modelo de pontos da mão

Origem: Google MediaPipe **Hand Landmarker**, pacote `float16/1`, instalado por download
explícito de URL oficial fixa. Não é um classificador dos gestos do GestureLab.

- [Documentação oficial e modelo](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker).
- [Model card oficial: Hand Tracking Lite/Full](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20%28Lite_Full%29%20with%20Fairness%20Oct%202021.pdf): licença Apache-2.0 declarada.
- [Licença do MediaPipe](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE).
- SHA-256: `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`.

O binário `.task` não é incluído no histórico Git; o instalador baixa e verifica o arquivo.
As capturas da interface documentadas no projeto são produzidas pela própria aplicação.
Fontes de sistema Windows são usadas quando disponíveis e não são copiadas para o projeto.
