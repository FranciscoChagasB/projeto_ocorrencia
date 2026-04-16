import pandas as pd
import numpy as np
import h3
import os
from pymongo import MongoClient, UpdateOne
from tqdm import tqdm

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
NOME_BANCO = "ocorrencia_tatica"
RESOLUCAO_H3 = 9

PESOS_OCORRENCIA = {
    'homicídio': 10.0, 'estupro': 10.0, 'sequestro': 10.0, 'tentativa de homicídio': 9.0,
    'disparo de arma': 8.5, 'grupo criminoso': 8.5, 'maria da penha': 8.5,
    'porte ilegal de arma': 8.0, 'roubo de veículo': 8.0, 'roubo': 7.0,
    'lesão corporal': 6.0, 'tráfico': 6.0, 'tráfico de drogas em eventos esportivos': 6.0,
    'importunação sexual': 6.0, 'violação da monitoração eletrônica de pessoas': 5.0,
    'furto de veículo': 4.0, 'furto': 3.0, 'danos/depredação': 3.0,
    'consumo de entorpecentes': 2.0, 'embriaguez e desordem': 2.0,
    'perturbação ao sossego alheio': 1.0, 'default': 1.0
}

def obter_peso(tipo):
    return PESOS_OCORRENCIA.get(str(tipo).strip().lower(), PESOS_OCORRENCIA['default'])

def reconstruir_mestre():
    print("🚀 INICIANDO RECONSTRUÇÃO MESTRE DO CÉREBRO DA I.A.")
    db = MongoClient(MONGO_URI)[NOME_BANCO]

    # 1. LEITURA DOS DADOS (Agora que o banco está padronizado, é simples assim)
    print("\n📥 Lendo ocorrências brutas do MongoDB...")
    cursor = db.ocorrencias_brutas.find(
        {"latitude": {"$exists": True}, "longitude": {"$exists": True}}, 
        {"_id": 1, "latitude": 1, "longitude": 1, "created_at": 1, "tipo_desc": 1, "hex_id": 1}
    )
    df = pd.DataFrame(list(cursor))
    
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df = df.dropna(subset=['latitude', 'longitude', 'created_at'])
    print(f"   📊 Encontradas {len(df)} ocorrências com dados válidos.")

    # 2. CURA DO BANCO (Garante que todos têm hex_id)
    mask_sem_hex = df.get('hex_id', pd.Series(dtype=str)).isna()
    if mask_sem_hex.any():
        print(f"\n🚑 Gerando hex_id para {mask_sem_hex.sum()} ocorrências...")
        def calc_h3(r):
            try: return h3.latlng_to_cell(r['latitude'], r['longitude'], RESOLUCAO_H3)
            except AttributeError: return h3.geo_to_h3(r['latitude'], r['longitude'], RESOLUCAO_H3)
            
        df.loc[mask_sem_hex, 'hex_id'] = df[mask_sem_hex].apply(calc_h3, axis=1)
        ops = [UpdateOne({'_id': r['_id']}, {'$set': {'hex_id': r['hex_id']}}) for _, r in df[mask_sem_hex].iterrows()]
        
        for i in tqdm(range(0, len(ops), 5000), desc="Salvando hex_ids", unit="lote"):
            db.ocorrencias_brutas.bulk_write(ops[i:i+5000])

    # 3. PESAGEM E AGRUPAMENTO
    df['janela_tempo'] = pd.to_datetime(df['created_at'], utc=True).dt.floor('h')
    df['peso_gravidade'] = df['tipo_desc'].apply(obter_peso)
    df_agrupado = df.groupby(['hex_id', 'janela_tempo'])['peso_gravidade'].sum().reset_index(name='score_risco_total')

    # 4. DENSIFICAÇÃO VETORIAL (O Turbo dos 3 Segundos)
    print("\n⏳ Iniciando Densificação de Tempo...")
    t_inicio, t_fim = df_agrupado['janela_tempo'].min(), df_agrupado['janela_tempo'].max()
    hex_unicos = df_agrupado['hex_id'].unique()

    linha_tempo = pd.date_range(start=t_inicio, end=t_fim, freq='h')
    multi_idx = pd.MultiIndex.from_product([hex_unicos, linha_tempo], names=['hex_id', 'janela_tempo'])
    print(f"   Matriz: {len(hex_unicos)} locais x {len(linha_tempo)} horas = {len(multi_idx)} tensores.")
    
    df_denso = df_agrupado.set_index(['hex_id', 'janela_tempo']).reindex(multi_idx, fill_value=0.0).reset_index()

    # 5. TRIGONOMETRIA
    horas, dias = df_denso['janela_tempo'].dt.hour, df_denso['janela_tempo'].dt.dayofweek
    df_denso['hora_sin'], df_denso['hora_cos'] = np.sin(2 * np.pi * horas / 24), np.cos(2 * np.pi * horas / 24)
    df_denso['dia_sin'], df_denso['dia_cos'] = np.sin(2 * np.pi * dias / 7), np.cos(2 * np.pi * dias / 7)
    df_denso['peso_cobertura'] = 1.0 

    # 6. SALVAR NO MONGO
    print("\n🗑️ Purgando coleção antiga...")
    db.features_historicas.delete_many({})

    print("💾 Injetando a linha do tempo perfeita...")
    registros = df_denso.to_dict('records')
    for i in tqdm(range(0, len(registros), 50000), desc="Salvando", unit="lote"):
        db.features_historicas.insert_many(registros[i:i + 50000])

    db.features_historicas.create_index([("hex_id", 1), ("janela_tempo", 1)], unique=True)
    print(f"\n✅ RECONSTRUÇÃO CONCLUÍDA! {len(registros)} tensores salvos.")

if __name__ == "__main__":
    reconstruir_mestre()