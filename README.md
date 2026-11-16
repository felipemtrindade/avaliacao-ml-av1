# Previsão de faixa de renda

**Aluno:** Felipe Raffael Moura Trindade

Neste trabalho, usei a base Adult para prever se uma pessoa recebe até 50 mil dólares por ano ou mais que isso. É um problema de classificação, porque o modelo escolhe entre duas faixas. Ele não calcula o salário da pessoa.

## Contexto acadêmico

Projeto desenvolvido para a AV1 da disciplina de Machine Learning
do IDP, no semestre 2026.2, sob orientação do professor Klayton Castro.

A avaliação propõe um projeto de Machine Learning de ponta a ponta:
escolha e preparação dos dados, treinamento e avaliação do modelo,
disponibilização por API Flask e execução com Docker.

Material de referência:
[Prática de MLOps do professor Klayton Castro](https://github.com/klaytoncastro/idp-machinelearning/tree/main/mlops).

A implementação e os resultados deste projeto são de autoria do aluno.

## Dados usados

A base é da UCI e tem 48.842 registros, com 14 características e a coluna de renda. São dados do censo americano de 1994. Escolhi essa base porque ela tem números, textos e informações faltando, permitindo trabalhar a preparação antes do treinamento.

- Página: https://archive.ics.uci.edu/dataset/2/adult
- Download: https://archive.ics.uci.edu/static/public/2/adult.zip
- Autores: Barry Becker e Ronny Kohavi (1996). Licença CC BY 4.0.

Os arquivos estão em `dataset`. Juntei `adult.data` e `adult.test` e fiz uma nova separação entre treino e teste. Por isso, esta avaliação não usa a divisão original da UCI. Usei a base inteira, sem criar dados artificiais.

A coluna que quero prever é `income`. No código, `<=50K` vira 0 e `>50K` vira 1.

| Coluna usada | Significado |
| --- | --- |
| `age` | Idade |
| `education-num` | Escolaridade, em uma escala de 1 a 16 |
| `capital-gain` | Ganho de capital |
| `capital-loss` | Perda de capital |
| `hours-per-week` | Horas de trabalho por semana |
| `workclass` | Tipo de vínculo de trabalho |
| `marital-status` | Estado civil |
| `occupation` | Ocupação |
| `relationship` | Relação dentro da família |
| `native-country` | País de origem |

Retirei `education`, porque a escolaridade já aparece em `education-num`. Também não usei `fnlwgt`, que é um peso da pesquisa. As colunas `race` e `sex` ficaram de fora. Mesmo assim, isso não elimina possíveis desigualdades nos dados.

## Preparação e treinamento

No `treinamento.py`, comecei verificando os tipos das colunas, os dados faltando, as repetições e as estatísticas dos números. Essa análise fica salva em `resultados/analise_dados.json`.

Removi espaços nos textos e o ponto que aparece no fim da renda em um dos arquivos. O sinal `?` foi tratado como dado ausente. Para preencher o que falta, usei a mediana nos números e o valor mais comum nos textos. Usei `OneHotEncoder` para transformar as categorias em colunas numéricas.

Encontrei 52 linhas iguais e retirei essas cópias. Sobraram 48.790 registros: 39.023 para treino e 9.767 para teste. A divisão ficou próxima de 80% e 20%, com proporções parecidas das duas classes. Registros com os mesmos valores nas colunas usadas ficam juntos, para não aparecerem nos dois conjuntos. O código separa cinco partes e usa uma delas como teste; não treina cinco modelos.

Mantive os valores altos de ganho e perda de capital, pois podem ser casos reais. Não fiz normalização, já que o modelo usa árvores e não precisa colocar os números na mesma escala. O preenchimento e a conversão das categorias aprendem somente com o treino.

Usei Random Forest, que combina várias árvores de decisão. Ela consegue considerar diferentes combinações de idade, escolaridade, ocupação e outras características. Os parâmetros foram:

- 150 árvores: para combinar várias previsões sem deixar o treino tão pesado.
- Profundidade máxima de 18 e mínimo de 3 registros por folha: para limitar regras muito específicas do treino.
- `class_weight='balanced'`: para dar mais peso à faixa de renda menos comum.
- `random_state=42`: para repetir a divisão e o treinamento.

Há bem mais pessoas na faixa de até 50 mil. Por isso, também comparei com um modelo simples que sempre responde essa faixa. Não fiz busca de parâmetros usando os resultados do teste.

## Resultado

O modelo teve **81,79% de acurácia**, contra 76,14% da comparação mais simples. Para a faixa acima de 50 mil, a precisão foi 57,91%, o recall foi 86,57% e o F1 foi 0,6940. A ROC AUC ficou em 0,9187.

Ele encontrou boa parte das pessoas da faixa de renda maior, mas também colocou muitas pessoas da faixa menor nesse grupo. Então, acertar cerca de 82% no geral não significa acertar igualmente nas duas faixas. Os erros e a matriz de confusão estão em [RESULTADOS.md](RESULTADOS.md).

Esses dados são antigos e dos Estados Unidos. O resultado não representa a renda atual no Brasil.

## Modelo salvo e API

O treinamento salva `modelo.pkl` usando `joblib`. A preparação e o modelo ficam juntos em um `Pipeline`, para a API repetir o mesmo tratamento. O script abre o arquivo salvo e confere se as previsões continuam iguais.

O `app.py` usa Flask e carrega esse arquivo quando inicia. Cada chamada só faz a previsão, sem treinar novamente. O `schema.py` guarda os nomes das colunas usadas pelos dois arquivos.

## Como rodar

Abra o terminal na pasta `projeto`. Os dados e o modelo treinado já estão incluídos.

Com Docker instalado e aberto:

```bash
docker compose up -d --build
docker compose ps
docker compose logs
```

Se sua instalação usa o comando antigo, execute `docker-compose up -d --build`. A API fica em `http://localhost:5000`. Para parar, use `docker compose down`.

Para treinar de novo e executar sem Docker, use Python 3.12. No Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe treinamento.py
.\.venv\Scripts\python.exe testar_api.py
.\.venv\Scripts\python.exe app.py
```

No Linux ou macOS, crie o ambiente com `python3 -m venv .venv` e use `.venv/bin/python` nos comandos seguintes. As bibliotecas estão no `requirements.txt`. Se gerar outro modelo e quiser usá-lo no Docker, refaça o comando com `--build`.

## Exemplo de chamada

A rota é `POST /predict`. Ela recebe os dez campos abaixo em JSON. Os cinco primeiros são números inteiros e os demais são textos. Os nomes seguem a base em inglês. Se faltar uma informação, envie `null` no valor, mantendo o nome do campo.

O arquivo `exemplo.json` contém:

```json
{
  "age": 39,
  "education-num": 13,
  "capital-gain": 2174,
  "capital-loss": 0,
  "hours-per-week": 40,
  "workclass": "State-gov",
  "marital-status": "Never-married",
  "occupation": "Adm-clerical",
  "relationship": "Not-in-family",
  "native-country": "United-States"
}
```

Com a API rodando, abra outro terminal na mesma pasta. No Windows:

```powershell
curl.exe -X POST http://localhost:5000/predict -H "Content-Type: application/json" --data-binary "@exemplo.json"
```

No Linux ou macOS, use `curl` no lugar de `curl.exe`.

A resposta desse exemplo foi:

```json
{"classe_prevista": "<=50K", "probabilidade_maior_50k": 0.108}
```

A previsão foi renda de até 50 mil. O valor 0,108 é a estimativa do modelo para a outra faixa, ou seja, 10,8%. Não é uma certeza sobre a pessoa.

A API confere os campos e os tipos enviados. Dados inválidos recebem erro 400, e chamadas sem o tipo `application/json` recebem erro 415. Categorias novas são aceitas, mas o modelo não aprendeu exemplos delas no treino.

## Testes e referências

O treinamento, a leitura do modelo salvo e os testes de `testar_api.py` funcionaram. No Docker, a imagem foi construída e o contêiner ficou ativo na porta 5000. A chamada com `exemplo.json` retornou HTTP 200 e a previsão esperada. Os testes também passaram dentro do contêiner, usando `docker compose exec api python testar_api.py`.

O material de MLOps ajudou na parte de salvar o modelo e usá-lo pelo Flask. Em um uso contínuo, seria necessário acompanhar os erros e as mudanças nos dados. Essa parte de monitoramento não foi implementada.

- Material da disciplina: https://github.com/klaytoncastro/idp-machinelearning/tree/main/mlops
- Pipeline: https://scikit-learn.org/stable/modules/compose.html
- Enunciado: Prova_ML_AV1_2026-2.pdf.