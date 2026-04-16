import pandas as pd
import numpy as np
import h3
import os
from pymongo import MongoClient, UpdateOne
from tqdm import tqdm

# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
NOME_BANCO = "ocorrencia_tatica" 
TAMANHO_DO_LOTE = 10000 
RESOLUCAO_H3 = 9

def reconstruir_features():
    print(f"🚀 Iniciando Reconstrução Total e Cura do Banco de Dados...")
    
    cliente = MongoClient(MONGO_URI)
    db = cliente[NOME_BANCO]
    col_brutas = db['ocorrencias_brutas']
    col_features = db['features_historicas']

    # ==========================================
    # 2. PUXANDO TODA A MEMÓRIA (Mesmo as corrompidas)
    # ==========================================
    print("\n📦 Extraindo todas as ocorrências brutas do MongoDB...")
    
    # Agora puxamos TUDO que tem latitude e longitude, ignorando se tem hex_id ou não
    cursor = col_brutas.find(
        {"latitude": {"$exists": True}, "longitude": {"$exists": True}},
        {"_id": 1, "latitude": 1, "longitude": 1, "created_at": 1, "hex_id": 1}
    )
    
    df = pd.DataFrame(list(cursor))
    
    if len(df) == 0:
        print("❌ ERRO: O banco de ocorrências brutas está vazio ou sem coordenadas.")
        return

    # Garante que as coordenadas antigas estão em formato numérico correto
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df = df.dropna(subset=['latitude', 'longitude', 'created_at'])

    # ==========================================
    # 3. FASE DE CURA (Healing dos Dados Antigos)
    # ==========================================
    # Descobre quem NÃO tem hex_id
    mask_sem_hex = df['hex_id'].isna()
    
    if mask_sem_hex.any():
        qtd_corrigir = mask_sem_hex.sum()
        print(f"\n🚑 Curando banco de dados: Encontradas {qtd_corrigir} ocorrências sem hex_id.")
        
        # Função para calcular o H3 adaptada para as duas versões da biblioteca
        def calcular_h3(row):
            try:
                return h3.latlng_to_cell(row['latitude'], row['longitude'], RESOLUCAO_H3)
            except AttributeError:
                return h3.geo_to_h3(row['latitude'], row['longitude'], RESOLUCAO_H3)

        # Calcula apenas para as que faltam
        df.loc[mask_sem_hex, 'hex_id'] = df[mask_sem_hex].apply(calcular_h3, axis=1)

        # Atualiza a tabela bruta no Mongo para a Timeline funcionar!
        operacoes_correcao = []
        for _, row in df[mask_sem_hex].iterrows():
            operacoes_correcao.append(
                UpdateOne({'_id': row['_id']}, {'$set': {'hex_id': row['hex_id']}})
            )

        print("💾 Salvando correções no banco bruto...")
        for i in tqdm(range(0, len(operacoes_correcao), 5000), desc="Corrigindo BD Bruto", unit="lote"):
            col_brutas.bulk_write(operacoes_correcao[i:i+5000])
            
        print("   ✅ Banco Bruto Curado com sucesso!")
    else:
        print("\n✅ O Banco Bruto já está perfeito. Nenhuma cura necessária.")

    print(f"   📊 Total de ocorrências limpas e mapeadas prontas para a I.A: {len(df)}")

    # ==========================================
    # 4. ZERANDO O MOTOR ANTIGO
    # ==========================================
    print("\n🧹 Limpando o Cérebro antigo da I.A...")
    col_features.delete_many({}) 
    print("   ✅ Tabela de features históricas zerada.")

    # ==========================================
    # 5. ENGENHARIA DE FEATURES TÁTICAS
    # ==========================================
    print("\n⚙️ Calculando tensores espaço-temporais para as 31 mil ocorrências...")
    
    df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
    df['janela_tempo'] = df['created_at'].dt.floor('h')
    df_features = df.groupby(['hex_id', 'janela_tempo']).size().reset_index(name='score_risco_total')

    hora = df_features['janela_tempo'].dt.hour
    dia_semana = df_features['janela_tempo'].dt.dayofweek

    df_features['hora_sin'] = np.sin(2 * np.pi * hora / 24)
    df_features['hora_cos'] = np.cos(2 * np.pi * hora / 24)
    df_features['dia_sin'] = np.sin(2 * np.pi * dia_semana / 7)
    df_features['dia_cos'] = np.cos(2 * np.pi * dia_semana / 7)
    df_features['peso_cobertura'] = 0 

    # ==========================================
    # 6. INJEÇÃO DE ALTA VELOCIDADE
    # ==========================================
    print("\n💾 Injetando o novo Cérebro no MongoDB...")
    
    registros = df_features.to_dict('records')
    
    for i in tqdm(range(0, len(registros), TAMANHO_DO_LOTE), desc="Inserindo Features", unit="lote"):
        lote = registros[i:i + TAMANHO_DO_LOTE]
        col_features.insert_many(lote)

    col_features.create_index([("hex_id", 1), ("janela_tempo", 1)], unique=True)
    
    print(f"\n   ✅ {len(df_features)} blocos de inteligência gerados e inseridos!")
    print("🎯 Reconstrução Concluída. O sistema está 100% atualizado com TODO o histórico de 31k registros.")

if __name__ == "__main__":
    reconstruir_features()