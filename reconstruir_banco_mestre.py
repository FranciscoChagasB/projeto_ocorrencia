import pandas as pd
import h3
import numpy as np
from pymongo import MongoClient
import os

print("🔌 Conectando ao Banco de Dados...")
client = MongoClient("mongodb://localhost:27017/")
db = client["ocorrencia_tatico"]

# 1. PEGAR A MATÉRIA PRIMA INTACTA
print("📥 Lendo as ocorrências brutas reais do MongoDB...")
crimes_brutos = list(db.ocorrencias_brutas.find({}, {"_id": 0}))

if not crimes_brutos:
    print("❌ ERRO: A coleção 'ocorrencias_brutas' está vazia! Importe seu Excel primeiro.")
    exit()

df_crimes = pd.DataFrame(crimes_brutos)

# 2. CONVERTER GEOGRAFIA E ARREDONDAR O TEMPO (O SEGREDO!)
print("🗺️ Mapeando coordenadas e arredondando os relógios...")
df_crimes['hex_id'] = df_crimes.apply(lambda x: h3.latlng_to_cell(x['latitude'], x['longitude'], 9), axis=1)

# Arredonda a data para baixo (14:32 vira 14:00). Assim o Pandas nunca mais vai perder um crime!
df_crimes['janela_tempo'] = df_crimes['data_hora'].dt.floor('h')

# 3. AGRUPAR OS CRIMES POR HORA E POR QUARTEIRÃO
print("📊 Calculando os riscos reais...")
df_agrupado = df_crimes.groupby(['hex_id', 'janela_tempo']).size().reset_index(name='score_risco_total')
total_crimes = df_agrupado['score_risco_total'].sum()
print(f"🔥 Total de crimes preservados com sucesso: {total_crimes}")

# 4. CRIAR A LINHA DO TEMPO PERFEITA
data_inicio = df_agrupado['janela_tempo'].min()
data_fim = df_agrupado['janela_tempo'].max()
linha_do_tempo_perfeita = pd.date_range(start=data_inicio, end=data_fim, freq='h')
hex_unicos = df_agrupado['hex_id'].unique()

print(f"⏳ Criando linha do tempo contínua para {len(hex_unicos)} hexágonos...")
registros_densos = []

for hex_id in hex_unicos:
    # Filtra só os crimes deste quarteirão
    dados_hex = df_agrupado[df_agrupado['hex_id'] == hex_id].copy()
    dados_hex.set_index('janela_tempo', inplace=True)
    
    # Preenche os buracos de tempo com ZEROS com segurança
    dados_densos = dados_hex.reindex(linha_do_tempo_perfeita)
    dados_densos['hex_id'] = hex_id
    dados_densos['score_risco_total'] = dados_densos['score_risco_total'].fillna(0.0)
    
    # Matemática temporal
    horas = dados_densos.index.hour
    dias = dados_densos.index.weekday
    dados_densos['hora_sin'] = np.sin(2 * np.pi * horas / 24)
    dados_densos['hora_cos'] = np.cos(2 * np.pi * horas / 24)
    dados_densos['dia_sin'] = np.sin(2 * np.pi * dias / 7)
    dados_densos['dia_cos'] = np.cos(2 * np.pi * dias / 7)
    dados_densos['peso_cobertura'] = 1.0
    
    dados_densos = dados_densos.reset_index().rename(columns={'index': 'janela_tempo'})
    registros_densos.extend(dados_densos.to_dict('records'))

# 5. SALVAR NO BANCO DE DADOS
print("🗑️ Limpando a coleção antiga corrompida...")
db.features_historicas.drop()

print("💾 Salvando o novo histórico à prova de falhas...")
lote_size = 50000
for i in range(0, len(registros_densos), lote_size):
    lote = registros_densos[i:i + lote_size]
    db.features_historicas.insert_many(lote)

print(f"✅ PIPELINE CONCLUÍDO! O banco agora tem {len(registros_densos)} registos e os crimes foram salvos!")