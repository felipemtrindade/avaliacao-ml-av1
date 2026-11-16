# Treina o modelo e salva os resultados
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from schema import NUMERICAS, CATEGORICAS, ATRIBUTOS, COLUNAS

BASE = Path(__file__).resolve().parent

def main():
    partes = []
    for nome in ['adult.data', 'adult.test']:
        tabela = pd.read_csv(
            BASE / 'dataset' / nome,
            names=COLUNAS,
            skipinitialspace=True,
            na_values='?',
            comment='|'
        )
        partes.append(tabela)
    dados = pd.concat(partes, ignore_index=True)

    # Limpeza dos textos e dos números
    for col in dados.select_dtypes(include='object'):
        dados[col] = dados[col].str.strip()
    dados['income'] = dados['income'].str.rstrip('.')
    for col in NUMERICAS + ['fnlwgt']:
        dados[col] = pd.to_numeric(dados[col], errors='coerce')
    resumo = {'registros_originais': len(dados), 'tipos': dados.dtypes.astype(str).to_dict(),
              'ausentes': dados.isna().sum().to_dict(),
              'duplicados_completos': int(dados.duplicated().sum()),
              'estatisticas_numericas': dados[NUMERICAS].describe().to_dict()}
    dados = dados.drop_duplicates().copy()
    # Uma mesma entrada não deve aparecer nos dois lados da avaliação
    repetidos = dados.duplicated(subset=ATRIBUTOS, keep=False)
    resumo['registros_com_entrada_repetida'] = int(repetidos.sum())
    if not dados['income'].isin(['<=50K', '>50K']).all():
        raise ValueError('Alvo inesperado na base.')
    X = dados[ATRIBUTOS]
    y = dados['income'].eq('>50K').astype(int)
    # Linhas com os mesmos atributos recebem o mesmo código de grupo
    grupos = pd.util.hash_pandas_object(X, index=False)
    divisao = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    treino, teste = next(divisao.split(X, y, groups=grupos))
    X_treino = X.iloc[treino]
    X_teste = X.iloc[teste]
    y_treino = y.iloc[treino]
    y_teste = y.iloc[teste]
    assert set(grupos.iloc[treino]).isdisjoint(set(grupos.iloc[teste]))
    resumo.update(registros_utilizados=len(dados), treino=len(X_treino), teste=len(X_teste),
                  classes=y.value_counts().sort_index().to_dict(),
                  classes_treino=y_treino.value_counts().sort_index().to_dict(),
                  classes_teste=y_teste.value_counts().sort_index().to_dict())
    # Preenche os textos que faltam e depois transforma em números
    preparo_categorias = Pipeline([
        ('ausentes', SimpleImputer(strategy='most_frequent')),
        ('codificacao', OneHotEncoder(handle_unknown='ignore'))
    ])
    preparo = ColumnTransformer([
        ('numeros', SimpleImputer(strategy='median'), NUMERICAS),
        ('categorias', preparo_categorias, CATEGORICAS)
    ])
    floresta = RandomForestClassifier(
        n_estimators=150,
        max_depth=18,
        min_samples_leaf=3,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    modelo = Pipeline([
        ('preparo', preparo),
        ('classificador', floresta)
    ])

    # Aprende com o treino e faz as previsões no teste
    modelo.fit(X_treino, y_treino)
    previsto = modelo.predict(X_teste)
    probabilidade = modelo.predict_proba(X_teste)[:, 1]
    referencia = DummyClassifier(strategy='most_frequent')
    referencia.fit(X_treino, y_treino)
    metricas = {'acuracia': accuracy_score(y_teste, previsto),
                'precisao': precision_score(y_teste, previsto, zero_division=0),
                'recall': recall_score(y_teste, previsto), 'f1': f1_score(y_teste, previsto),
                'roc_auc': roc_auc_score(y_teste, probabilidade),
                'matriz_confusao': confusion_matrix(y_teste, previsto).tolist(),
                'acuracia_referencia': accuracy_score(y_teste, referencia.predict(X_teste)),
                'f1_referencia': f1_score(y_teste, referencia.predict(X_teste), zero_division=0)}
    saida = BASE / 'resultados'
    saida.mkdir(exist_ok=True)
    for nome, objeto in [('analise_dados.json', resumo), ('metricas.json', metricas)]:
        (saida / nome).write_text(json.dumps(objeto, indent=2, ensure_ascii=False), encoding='utf-8')
    pd.DataFrame({'indice': X_teste.index, 'real': y_teste, 'previsto': previsto,
                  'probabilidade_maior_50k': probabilidade}).to_csv(saida / 'previsoes_teste.csv', index=False)
    # Salva tudo junto para usar depois na API
    joblib.dump(modelo, BASE / 'modelo.pkl', compress=3)
    recarregado = joblib.load(BASE / 'modelo.pkl')
    assert (recarregado.predict(X_teste) == previsto).all()
    print(json.dumps({'dados': resumo, 'metricas': metricas}, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()