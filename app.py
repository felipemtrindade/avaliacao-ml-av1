# Carrega o modelo salvo e recebe os dados para prever a renda
import math
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from werkzeug.exceptions import BadRequest
from schema import ATRIBUTOS, NUMERICAS, CATEGORICAS

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024
modelo = joblib.load(Path(__file__).resolve().parent / 'modelo.pkl')

@app.get('/')
def inicio():
    return jsonify(status='ok', projeto='Classificação de renda - Adult', endpoint='POST /predict')

@app.post('/predict')
def predict():
    if not request.is_json:
        return jsonify(erro='Envie Content-Type: application/json.'), 415
    try:
        dados = request.get_json()
    except BadRequest:
        return jsonify(erro='JSON inválido.'), 400
    if not isinstance(dados, dict):
        return jsonify(erro='Envie um objeto JSON com os atributos de uma pessoa.'), 400
    faltantes = sorted(set(ATRIBUTOS) - set(dados))
    extras = sorted(set(dados) - set(ATRIBUTOS))
    if faltantes or extras:
        return jsonify(erro='Confira os campos enviados.', faltantes=faltantes, extras=extras), 400
    limites = {'age': (17, 120), 'education-num': (1, 16), 'capital-gain': (0, 1e12),
               'capital-loss': (0, 1e12), 'hours-per-week': (1, 168)}
    entrada = {}
    for campo in NUMERICAS:
        valor = dados[campo]
        if valor is None:
            entrada[campo] = np.nan
            continue
        minimo, maximo = limites[campo]
        if (isinstance(valor, bool) or not isinstance(valor, (int, float))
                or not minimo <= valor <= maximo or not math.isfinite(valor)
                or not float(valor).is_integer()):
            return jsonify(erro=f'{campo}: use um inteiro entre {minimo:g} e {maximo:g}, ou null.'), 400
        entrada[campo] = valor
    for campo in CATEGORICAS:
        valor = dados[campo]
        if valor is not None and (not isinstance(valor, str) or len(valor) > 100):
            return jsonify(erro=f'{campo}: use texto ou null.'), 400
        entrada[campo] = np.nan if valor is None or valor.strip() in ('', '?') else valor.strip()
    X = pd.DataFrame([entrada], columns=ATRIBUTOS)
    classe = int(modelo.predict(X)[0])
    return jsonify(classe_prevista='>50K' if classe else '<=50K',
                   probabilidade_maior_50k=round(float(modelo.predict_proba(X)[0, 1]), 4))

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=5000)