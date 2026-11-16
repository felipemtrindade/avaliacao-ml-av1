# Resultados

## Conferência dos dados

A base tinha 48.842 linhas. Depois de remover 52 cópias iguais, ficaram 48.790. Os dados faltantes estavam em `workclass` (2.799), `occupation` (2.809) e `native-country` (857). Essas contagens são da base original. Nas colunas numéricas usadas, não havia valores faltando.

As idades vão de 17 a 90 anos e as horas de trabalho vão de 1 a 99 por semana. O ganho de capital chega a 99.999 e a perda a 4.356, mas pelo menos 75% dos registros têm zero nesses dois campos. Mantive os valores altos, porque só serem diferentes da maioria não significa que estejam errados.

Após tirar as cópias completas, 13.901 registros ainda tinham os mesmos valores de outras linhas nas colunas escolhidas. Eles foram mantidos, mas ficaram agrupados na divisão entre treino e teste.

| Conjunto | Até 50 mil | Acima de 50 mil | Total |
| --- | ---: | ---: | ---: |
| Treino | 29.672 | 9.351 | 39.023 |
| Teste | 7.437 | 2.330 | 9.767 |

Cerca de 24% das pessoas estão na faixa de renda maior. Isso explica por que um modelo que sempre responde a faixa menor já consegue bastante acerto, mesmo sem aprender a identificar a outra faixa.

## Como o modelo se saiu

Estes resultados são dos 9.767 registros de teste, que ficaram fora do treinamento. Nas métricas abaixo, a classe de interesse é a renda acima de 50 mil.

| Métrica | Valor |
| --- | ---: |
| Acurácia | 81,79% |
| Precisão | 57,91% |
| Recall | 86,57% |
| F1 | 0,6940 |
| ROC AUC | 0,9187 |

A acurácia mostra que o modelo acertou cerca de 82 em cada 100 casos. O modelo de comparação, que sempre responde até 50 mil, acertou 76,14%, mas teve F1 igual a zero para a faixa maior.

O recall mostra que foram encontrados cerca de 87 em cada 100 casos de renda maior. Já a precisão mostra que, entre as pessoas que o modelo colocou nessa faixa, só cerca de 58 em cada 100 realmente pertenciam a ela. Ou seja, ele encontra muitos casos, mas também dá bastante alarme falso.

O F1 resume o equilíbrio entre precisão e recall. A AUC mostra a capacidade de dar probabilidades maiores aos casos da faixa maior do que aos da menor. Ela não significa 91,87% de acertos.

## Onde ele errou

Na matriz abaixo, a linha é a renda real e a coluna é o que o modelo respondeu.

| Renda real / resposta | Até 50 mil | Acima de 50 mil |
| --- | ---: | ---: |
| Até 50 mil | 5.971 | 1.466 |
| Acima de 50 mil | 313 | 2.017 |

Foram 7.988 acertos e 1.779 erros. Em 1.466 casos, o modelo respondeu renda maior quando a pessoa estava na faixa menor. Em outros 313, aconteceu o contrário.

Usei pesos para dar mais atenção à faixa menos comum. O resultado teve recall alto e precisão mais baixa, mas não fiz uma comparação sem os pesos para dizer quanto eles mudaram o resultado.

O modelo melhorou em relação à comparação simples, mas ainda erra bastante ao indicar renda maior. Para melhorar, seria possível testar outros parâmetros usando uma parte do treino como validação. O teste final deve continuar separado dessa escolha.

Os números completos estão em `resultados/metricas.json`. As respostas de cada registro do teste estão em `resultados/previsoes_teste.csv`.

## Teste da API

O exemplo do README retornou `<=50K`, com 0,108 de probabilidade para a faixa acima de 50 mil. O arquivo `testar_api.py` também conferiu chamadas com campos faltando, valores inválidos, valores nulos e categorias novas. Os testes passaram.

O modelo salvo manteve as mesmas previsões depois de ser carregado novamente. A API funcionou localmente e no Docker. O contêiner ficou ativo na porta 5000, a chamada do exemplo retornou HTTP 200 e os testes de `testar_api.py` passaram dentro dele.