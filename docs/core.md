# Núcleo de dados e ML

O GestureLab guarda tudo localmente no diretório `data/` (ou o valor de
`--data-dir`). O arquivo SQLite contém categorias, sessões, metadados,
histórico de experimentos e divisões. Cada coleta grava um shard numérico
Parquet em `sample_shards/`, evitando regravar todo o conjunto durante a
captura. Um export cria uma projeção única Parquet quando necessária. Shards e
exports podem ser regenerados pelo SQLite; não alteram o histórico.

## Dados e versões

Uma amostra contém 21 pontos `(x, y, z)`, portanto 63 números finitos, além de
sessão, rótulo, lateralidade, versão do extrator e qualidade disponível. A
coleta deve descartar ausência de mão, múltiplas mãos e pontos inválidos antes
de chamar `add_sample`.

`snapshot` copia o conjunto atual para `snapshots/` em Parquet e registra
SHA-256. Uma edição, exclusão ou relabel posterior só muda o conjunto atual:
um experimento continua apontando para a cópia e o hash que usou.

```powershell
python -m gesturelab.cli --data-dir data validate
python -m gesturelab.cli --data-dir data snapshot
python -m gesturelab.cli --data-dir data export resultado.parquet
```

## Atributos

Os pontos crus permanecem guardados. As estratégias atuais são:

- `raw`: as 63 coordenadas que vieram do extrator;
- `normalized`: punho na origem, escala pela distância punho--MCP médio;
- `distances`: nove distâncias selecionadas no espaço normalizado;
- `angles`: quinze ângulos de segmentos dos dedos no espaço normalizado.

Para dados normalizados, uma mão `Left` tem apenas a coordenada X invertida no
espaço de atributos. A prévia espelhada da câmera é outro assunto e não muda
os dados salvos. A normalização reduz efeitos de posição e escala; não calibra
profundidade e não elimina rotação, oclusão nem uma pose mal extraída.

`StandardScaler` fica dentro do `Pipeline` do scikit-learn e é ajustado só com
o treino. A mesma transformação é chamada pela inferência.

## Protocolo de avaliação

O protocolo padrão agrupa por `session`; `--protocol participant` agrupa pelo
participante pseudônimo e exige que ele esteja preenchido. O núcleo nunca faz
fallback para divisão por frames. Para criar treino, validação e teste final
com todas as classes, são necessários pelo menos quatro grupos e cada gesto
deve aparecer nos grupos independentes apropriados.

A primeira divisão por linhagem é persistida. O primeiro protocolo escolhido
(`session` ou `participant`) também fica bloqueado para essa linhagem, pois
mudar o agrupamento poderia expor o teste reservado. Em snapshots posteriores da
mesma Store, sessões já reservadas para teste continuam reservadas, mesmo que
se troque a semente. Sessões novas entram no desenvolvimento. Isso impede que
um teste final seja liberado só porque foi criado outro snapshot.

Durante `train`, grade, modelo, atributos e limiar consultam apenas treino e
validação. `evaluate` consulta o teste final reservado uma vez e armazena o
resultado; chamadas seguintes devolvem o resultado armazenado. O limiar escolhe
o melhor Macro-F1 das previsões aceitas com cobertura mínima de 60%, e relata
cobertura e desempenho aceito para que rejeitar quase tudo não pareça melhoria.

```powershell
python -m gesturelab.cli --data-dir data train dataset_SEU_ID --models dummy logistic forest mlp --strategy normalized --protocol session
python -m gesturelab.cli --data-dir data evaluate experiment_SEU_ID
python -m gesturelab.cli --data-dir data compare experiment_A experiment_B
```

`compare` declara a comparação controlada quando linhagem e protocolo coincidem:
as amostras de validação e teste têm IDs e hashes congelados. Dataset, atributos,
modelo e semente aparecem como variáveis experimentais (por exemplo, para
comparar atributos ou uma versão de coleta ampliada). Uma melhoria com exemplos difíceis é uma
hipótese: revise e rotule esses exemplos antes de adicioná-los; a previsão do
modelo nunca vira rótulo automaticamente.

## Artefatos e cancelamento

Cada experimento registra semente, versões relevantes, parâmetros, duração,
métricas de validação, split e hash do artefato. O arquivo joblib é escrito em
arquivo temporário e publicado por troca atômica somente ao terminar. O
cancelamento é verificado antes e entre ajustes; uma biblioteca de ML não é
interrompida no meio de um único `fit`, mas o resultado desse `fit` não é
publicado quando o cancelamento já foi solicitado.

Joblib/pickle pode executar código ao abrir arquivos maliciosos. Por isso
`load_model` aceita somente caminhos abaixo de `data/artifacts`, confere o
SHA-256 gravado pelo Store e exige versões compatíveis de Python e
scikit-learn. Não abra artefatos recebidos de terceiros.
