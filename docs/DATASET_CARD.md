# Dataset card

**Estado inicial: sem amostras reais coletadas.** Os testes automatizados geram somente
dados numéricos sintéticos temporários, que não são incorporados ao dataset do usuário.

## Origem e finalidade

Dados produzidos localmente pela webcam, com rótulo atribuído pelo operador. Objetivo:
aprender reconhecimento de gestos estáticos e avaliar generalização por sessão ou pessoa.
Não é um dataset de língua de sinais, identidade, comportamento ou biometria autenticadora.

## Estrutura

- ID da amostra e da sessão, pseudônimo opcional, categoria e instante da captura.
- Lateralidade retornada pelo extrator.
- 21 pontos × 3 coordenadas, preservados antes das transformações.
- Versão do extrator e informações geométricas de qualidade disponíveis.
- Propósito da sessão: desenvolvimento ou desafio.
- Metadados locais em SQLite; números e snapshots em Parquet; hashes de integridade.

**Privacidade:** não contém imagens por padrão nem faz upload. Pseudônimos não devem conter
nomes, e-mails ou outras informações identificáveis. Coordenadas ainda merecem proteção local.

## Preencher após coleta

| Campo | Resultado |
| --- | --- |
| Versão e hash | Pendente |
| Período de coleta | Pendente |
| Pessoas pseudonimizadas / sessões | Pendente |
| Classes e quantidade por classe | Pendente |
| Resolução, câmera e condições | Pendente |
| Diversidade de mão / posição / luz | Pendente |
| Exclusões e motivo | Pendente |
| Duplicatas e revisão de rótulos | Pendente |
| Limitações e vieses da coleta | Pendente |

Excluir o conjunto atual não apaga snapshots e modelos históricos. Para eliminação integral
dos dados de uma pessoa, é necessário revisar também versões, exportações e artefatos derivados;
faça essa manutenção local conscientemente e não publique esses arquivos por engano.
