import pandas as pd
from pymongo import MongoClient
import numpy as np
from datetime import datetime, timedelta

print("Conectando ao Banco de Dados...")
client = MongoClient("mongodb://localhost:27017/")
db = client["ocorrencia_tatico"]

print("Extraindo histórico esparso (com buracos)...")
dados_antigos = list(db.features_historicas.find({}, {"_id": 0}))

if not dados_antigos:
    print("Banco vazio! Nada para densificar.")
    exit()

df = pd.DataFrame(dados_antigos)
df['janela_tempo'] = pd.to_datetime(df['janela_tempo'])

# Descobre o início e o fim do tempo no seu banco
data_inicio = df['janela_tempo'].min().floor('H')
data_fim = df['janela_tempo'].max().ceil('H')
hex_unicos = df['hex_id'].unique()

print(f"Período detetado: {data_inicio} até {data_fim}")
print(f"Hexágonos conhecidos: {len(hex_unicos)}")

# 1. Cria uma linha do tempo contínua perfeita de hora em hora
linha_do_tempo_perfeita = pd.date_range(start=data_inicio, end=data_fim, freq='H')

# 2. Cria o novo DataFrame denso
registros_densos = []
print("Injetando Zeros (Zero-Padding)... Isso pode levar alguns segundos.")

for hex_id in hex_unicos:
    # Pega apenas os dados deste hexágono
    dados_deste_hex = df[df['hex_id'] == hex_id].set_index('janela_tempo')
    
    # MÁGICA: Força o Pandas a adotar a linha do tempo perfeita. 
    # Onde não havia dados, ele cria a linha e coloca 'NaN'
    dados_reindexados = dados_deste_hex.reindex(linha_do_tempo_perfeita)
    
    # Preenche os 'NaN' de risco com ZERO
    dados_reindexados['score_risco_total'] = dados_reindexados['score_risco_total'].fillna(0.0)
    
    # Refaz a matemática do tempo para as linhas novas
    horas = dados_reindexados.index.hour
    dias = dados_reindexados.index.weekday
    
    dados_reindexados['hora_sin'] = np.sin(2 * np.pi * horas / 24)
    dados_reindexados['hora_cos'] = np.cos(2 * np.pi * horas / 24)
    dados_reindexados['dia_sin'] = np.sin(2 * np.pi * dias / 7)
    dados_reindexados['dia_cos'] = np.cos(2 * np.pi * dias / 7)
    
    # Mantém o peso de cobertura ou preenche com 1.0
    dados_reindexados['peso_cobertura'] = dados_reindexados['peso_cobertura'].fillna(1.0)
    dados_reindexados['hex_id'] = hex_id
    
    # Reseta o index para transformar as datas de volta em coluna
    dados_reindexados = dados_reindexados.reset_index().rename(columns={'index': 'janela_tempo'})
    registros_densos.extend(dados_reindexados.to_dict('records'))

print("🗑️ Limpando a coleção antiga...")
db.features_historicas.drop()

print("💾 Salvando o novo histórico denso militar...")
# Salva em blocos para não estourar a memória do Mongo
lote_size = 50000
for i in range(0, len(registros_densos), lote_size):
    lote = registros_densos[i:i + lote_size]
    db.features_historicas.insert_many(lote)

print(f"SUCESSO! A memória da I.A. foi curada. Total de registos: {len(registros_densos)}.")