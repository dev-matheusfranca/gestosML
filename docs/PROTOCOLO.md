# Protocolo de coleta e avaliação

## Antes da coleta

1. Abra a câmera em **Coleta**, com resolução padrão de 640 × 480.
2. Use uma pessoa e uma mão por vez. Não mostre rosto ou material privado se puder evitar.
3. Use um pseudônimo, como `p01`, se quiser medir generalização entre pessoas.
4. Crie uma sessão por rodada independente de coleta. Não divida artificialmente uma
   gravação contínua em várias sessões para aumentar a contagem de grupos.
5. Comece com cinco categorias: palma, punho, positivo, V e indicador apontando.

## Quantidade e diversidade

Sugestão inicial: ao menos **4 sessões completas**, com todas as classes em cada sessão;
preferencialmente 5–8 sessões para uma análise mais informativa. Uma frequência de 2 Hz e
20–40 exemplos variados por classe por sessão é um ponto de partida, não uma garantia.
Faça pausas e reposicione a mão entre rodadas. Varie distância, inclinação e condições
de uso, sem registrar centenas de frames quase iguais como se fossem independentes.

O software bloqueia divisões sem classes suficientes. Não existe fallback para sorteio de frames.
Para generalização por participante, use ao menos 4 pessoas pseudonimizadas com todas as classes;
mais participantes são desejáveis. Quatro pessoas são apenas o mínimo estrutural, não evidência robusta.

## Critérios da coleta

Nenhuma amostra com zero ou duas mãos; pontos devem ter 21 × 3 valores finitos, palma com
escala não degenerada e mão dentro do enquadramento. Na interface é possível ajustar o
limiar do detector e o tamanho mínimo da palma. A pontuação de lateralidade indica a
confiança em esquerda/direita; não é confiança de cada ponto nem da qualidade do gesto.

O eixo z é estimado. Imagens e vídeos não são salvos. Espelhamento ocorre somente na prévia.
Revise desenhos dos pontos e rótulos; descarte exemplos inválidos com confirmação.
Valide duplicatas antes de criar uma versão. Versões preservam o histórico mesmo se
você excluir ou corrigir amostras do conjunto atual.

## Protocolos

| Protocolo | Unidade de separação | O que mede |
| --- | --- | --- |
| Sessão | `session_id` | Generalização para outra rodada de coleta; não prova generalização para outra pessoa |
| Participante | pseudônimo | Generalização para pessoas ausentes do treino; requer participantes identificados por pseudônimos consistentes |

Todos os modelos de uma comparação usam IDs de divisão persistidos. O teste final fica
reservado; seleção de atributos, modelos, hiperparâmetros e limiar utiliza desenvolvimento.
Os detalhes da reserva persistente e os bloqueios estão em [core.md](core.md).

## Incerteza e desafio

O limiar é escolhido pela validação. Registre também **cobertura**, a proporção de previsões
aceitas, e desempenho entre as aceitas. Uma aparente melhora pode vir de rejeitar quase tudo.
Crie sessões do tipo **desafio** para gestos não presentes nas classes de desenvolvimento.
Não use o desafio como classe de treino nem para escolher o limiar após observar o resultado.
Relate a aceitação indevida nesse conjunto sem prometer rejeição universal de desconhecidos.

## Ciclo de melhoria

Use somente falhas de desenvolvimento para selecionar exemplos difíceis. A revisão humana
é obrigatória. Mantenha protocolo e avaliação congelados ao comparar versões; diferenças
de dataset devem ser sinalizadas. Mudanças de classes ou extrator exigem novo desenho do
experimento. Nunca ajuste e escolha novamente usando o teste final já consultado.

## Próxima coleta após a V1 documentada

A V1 já tem teste final salvo, mas foi coletada por uma pessoa na mesma posição. Não
reescrever essa condição nem reutilizar o teste para selecionar outro modelo. Para ampliar
a evidência, criar uma coleta separada, mantendo a V1 e seus resultados preservados:

```powershell
.\.venv\Scripts\python.exe -m gesturelab --data-dir data/coleta-v2 gui
```

1. Definir o objetivo antes: robustez para a mesma pessoa em novas condições ou para outras
   pessoas. O segundo exige protocolo por participante e participantes realmente distintos.
2. Registrar por rodada as condições, com pausas reais e variações planejadas de posição,
   distância e iluminação. Não atribuir independência apenas por abrir outra sessão.
3. Coletar todas as classes em cada grupo e reservar grupos de validação/teste antes das escolhas.
   As quantidades mínimas acima permitem dividir os dados, mas não garantem validade estatística.
4. Definir o critério de seleção: por exemplo macro-F1 global mais uma cobertura mínima
   declarada. Registrar cobertura e acerto condicional juntos para todos os modelos.
5. Rever erros somente do desenvolvimento. Criar desafio separado de desconhecidos e
   consultar o novo teste final uma vez, depois de congelar as escolhas.

As seis pendências de revisão da V1 não foram corrigidas automaticamente: exigem rótulo
humano. Uma melhora só pode ser anunciada quando medida, não por haver mais amostras.

## Checklist humano

- [ ] Zero mãos, uma mão, duas mãos e mão parcialmente fora da imagem.
- [ ] Câmera ocupada por outro aplicativo; índice inexistente.
- [ ] Abrir, pausar, parar, reabrir; fechar aplicação com captura ativa.
- [ ] Dataset real vazio, insuficiente e suficiente.
- [ ] Nova sessão com as mesmas classes, sem reutilizar frames de treino.
- [ ] Gestos parecidos e gestos fora das categorias.
- [ ] Marcar erro, revisar rótulo, criar nova versão e comparar desenvolvimento.
- [ ] Registrar reconhecimento real e limitações; somente então avaliar teste final.
