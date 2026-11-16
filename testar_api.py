# Testa a API com o exemplo e com algumas entradas inválidas
import json
from pathlib import Path
import pandas as pd
from app import app, modelo

BASE = Path(__file__).resolve().parent
dados = json.loads((BASE / 'exemplo.json').read_text(encoding='utf-8'))
cliente = app.test_client()
resposta = cliente.post('/predict', json=dados)
assert resposta.status_code == 200
esperado = '>50K' if modelo.predict(pd.DataFrame([dados]))[0] else '<=50K'
assert resposta.json['classe_prevista'] == esperado
assert cliente.post('/predict', json={}).status_code == 400
assert cliente.post('/predict', json=[]).status_code == 400
assert cliente.post('/predict', data='{}', content_type='text/plain').status_code == 415
assert cliente.post('/predict', data='{', content_type='application/json').status_code == 400
for valor in [-1, True, '39', float('inf'), 10**400]:
    assert cliente.post('/predict', json={**dados, 'age': valor}).status_code == 400
assert cliente.post('/predict', json={**dados, 'income': '>50K'}).status_code == 400
assert cliente.post('/predict', json={**dados, 'occupation': None}).status_code == 200
assert cliente.post('/predict', json={**dados, 'age': None}).status_code == 200
assert cliente.post('/predict', json={**dados, 'occupation': 'Categoria-nova'}).status_code == 200
(BASE / 'resultados' / 'resposta_exemplo.json').write_text(
    json.dumps(resposta.json, indent=2, ensure_ascii=False), encoding='utf-8')
print('API PASSOUUU!. Resposta real:', resposta.json)