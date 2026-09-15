# Model card — classificadores GestureLab

**Estado: nenhum classificador treinado com gestos reais nesta entrega.**

## Uso previsto

Laboratório educacional e demonstração local de gestos estáticos. Não usar para autenticação,
acessibilidade crítica, tradução de língua de sinais, decisão médica ou controle de dispositivos.

## Dois componentes distintos

1. **Hand Landmarker:** detector/extrator pré-treinado do MediaPipe, executado em CPU.
2. **Classificador GestureLab:** Dummy, regressão logística, Random Forest ou MLP, treinado
   com exemplos rotulados pelo usuário; recebe atributos dos pontos.

Normalização, seleção de atributos e transformações estatísticas fazem parte do pipeline
persistido. O mesmo processamento é usado na inferência. Especificação dos atributos: [core.md](core.md).

## Resultados a preencher

| Informação | Resultado real |
| --- | --- |
| Experimento / modelo / parâmetros | Pendente |
| Dataset / hash / grupos / semente | Pendente |
| Protocolo e IDs de treino / validação / teste | Pendente |
| Acurácia / macro-F1 de validação | Pendente |
| Precisão / recall / F1 / suporte por classe | Pendente |
| Cobertura e desempenho aceito | Pendente |
| Aceitação de gestos desconhecidos no desafio | Pendente |
| Tempo de treino / latência de classificação | Pendente |
| Teste final, após congelar escolhas | Pendente |

## Limitações

Dependência da qualidade dos pontos, oclusões, orientação, distância e iluminação. Dados de
uma única pessoa não demonstram desempenho em outras. Pontuações de modelos diferentes não
são necessariamente calibradas. Suavização melhora continuidade visual e acrescenta atraso;
não altera as métricas individuais do classificador. Gestos não cadastrados podem receber
pontuação alta. Mais dados difíceis podem não melhorar o modelo.

Modelos locais são serializados com joblib/pickle. Nunca carregue arquivos recebidos de
terceiros como se fossem modelos confiáveis. O histórico registra dependências e integridade,
mas não é uma assinatura de procedência.
