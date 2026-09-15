# Aprender com o GestureLab

## Conceitos em um minuto

- **Amostra:** os 21 pontos de uma mão em um instante. Frames vizinhos são correlacionados.
- **Rótulo:** o gesto correto atribuído por uma pessoa. Uma previsão não é verdade de treino.
- **Atributo:** número oferecido ao classificador, como uma coordenada relativa ou distância.
- **Normalização:** retirar posição absoluta e escala para tornar comparáveis mãos em lugares diferentes.
- **Treino:** dados usados para ajustar pesos, árvores ou uma rede.
- **Validação:** dados independentes usados para escolher modelo, parâmetros e limiar.
- **Teste final:** dados reservados para uma avaliação após terminar as escolhas.
- **Overfitting:** aprender detalhes da coleta que não se repetem numa sessão nova.
- **Vazamento:** usar informação da avaliação na preparação ou seleção, inflando o resultado.
- **Macro-F1:** média do F1 por classe; cada gesto conta igualmente, mesmo com quantidades diferentes.
- **Matriz de confusão:** linhas reais e colunas previstas; fora da diagonal estão as confusões.

## Papel de cada modelo

Dummy é a referência mínima sem reconhecimento útil. Regressão logística aprende uma
separação linear dos atributos. Random Forest combina árvores para relações não lineares.
MLP é uma rede neural pequena, que pode exigir mais dados e ajuste de escala.
Maior complexidade não garante melhor generalização.

## Caderno de exercícios

Para cada exercício, registre hipótese **antes** de rodar e guarde IDs dos experimentos.
Não registre só a melhor tentativa; inclua resultado negativo e classes com baixo suporte.

| Exercício | Hipótese a investigar | Procedimento |
| --- | --- | --- |
| Sessões distintas | A avaliação cai fora da sessão de treino? | Colete os mesmos gestos em ao menos 4 sessões; separe por sessão, reservando teste final |
| Bruto × normalizado | Remover posição e escala ajuda? | Mesmo dataset/divisão/modelo/semente; varie apenas atributos |
| Comparação de modelos | A complexidade ajuda neste dataset? | Compare os quatro modelos na mesma validação e observe macro-F1 e latência |
| Classe minoritária | Acurácia esconde um gesto mal aprendido? | Crie outra versão com menos exemplos de uma classe no treino; mantenha avaliação independente |
| Gestos parecidos | Quais classes confundem entre si? | Observe a matriz; revise exemplos de desenvolvimento antes de coletar mais |
| Luz e orientação | A extração ou a classificação falha primeiro? | Nova sessão com condição distinta; registre rejeições de qualidade e confusões |
| Aleatórios × difíceis | Seleção por erro é melhor que volume? | Mesma base, mesmo orçamento adicional, avaliação congelada e várias sementes |

### Registro por exercício

```text
Hipótese:
Dataset / hash:
Protocolo e IDs da divisão:
Variável alterada:
Condições mantidas:
Sementes:
Resultado de validação (macro-F1 / cobertura / latência):
Classes que pioraram:
Interpretação e limitações:
Próximo teste de desenvolvimento:
```

## Como melhorar sem contaminar a avaliação

Treine V1; use dados de desenvolvimento para achar erros; marque exemplos difíceis;
revise o rótulo correto; colete variações em sessões de treino; crie V2; compare com
avaliação compatível. A V2 pode piorar. Não retreine usando automaticamente a previsão.
O conjunto desafio contém gestos não cadastrados, usado apenas para relatar aceitação
indevida pelo limiar já escolhido na validação.

## Depois da primeira versão

Apenas propostas: sequências de gestos dinâmicos, modelos temporais, novos participantes,
calibração de probabilidades, reconhecimento de categorias desconhecidas, exportação para
outros runtimes e comparação de atributos manuais com representações aprendidas.
Essas extensões não fazem parte da implementação desta versão.
