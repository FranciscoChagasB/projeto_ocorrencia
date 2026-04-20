import json
import math
import os
from datetime import datetime, timezone
from pymongo import MongoClient
import pandas as pd
import numpy as np
import h3

# ==========================================
# CONFIGURAÇÕES
# ==========================================
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
NOME_BANCO = "ocorrencia_tatica"
ARQUIVO_JSON = "dados_postgres.json" # <-- Coloque o nome do seu ficheiro JSON aqui

# Tabela de pesos para a I.A. saber a gravidade
def obter_peso_crime(tipo_desc):
    if not tipo_desc: return 3.0
    t = str(tipo_desc).upper()
    if 'HOMIC' in t: return 10.0
    if 'ARMA' in t: return 8.5
    if 'PENHA' in t: return 8.5
    if 'GRUPO CRIMINOSO' in t: return 8.5
    if 'ROUBO' in t: return 7.0
    if 'FURTO' in t: return 5.0
    if 'PERTURBA' in t: return 1.0
    return 3.0

def parse_data(data_str):
    if not data_str: return None
    try:
        # Pega apenas os primeiros 19 caracteres "2026-04-16T10:19:09" e ignora os milissegundos/timezone complexo
        dt_str = str(data_str)[:19] 
        # Converter para datetime e forçar fuso horário UTC para o MongoDB entender perfeitamente
        return datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except Exception as e:
        return None

def importar_dados():
    print(f"1. Conectando ao MongoDB ({NOME_BANCO})...")
    cliente = MongoClient(MONGO_URI)
    db = cliente[NOME_BANCO]
    
    print(f"2. Lendo ficheiro {ARQUIVO_JSON}...")
    try:
        with open(ARQUIVO_JSON, 'r', encoding='utf-8') as f:
            dados_brutos = json.load(f)
    except Exception as e:
        print(f"❌ Erro ao ler JSON: {e}")
        return

    # O PostgreSQL exporta com a query SQL como chave principal. Vamos extrair a lista dinâmica.
    chave_raiz = list(dados_brutos.keys())[0]
    lista_registos = dados_brutos[chave_raiz]
    
    print(f"3. Processando e inserindo {len(lista_registos)} ocorrências brutas...")
    
    operacoes_inseridas = 0
    df_features_raw = []

    for reg in lista_registos:
        try:
            # Gerar o H3 Index
            try:
                hex_id = h3.latlng_to_cell(reg['latitude'], reg['longitude'], 9)
            except AttributeError:
                hex_id = h3.geo_to_h3(reg['latitude'], reg['longitude'], 9)
            
            # Converter datas principais
            created_at_dt = parse_data(reg.get('created_at'))
            if not created_at_dt:
                created_at_dt = parse_data(reg.get('criacao')) # Fallback
                
            # Montar o Documento Rico para o MongoDB
            doc = {
                "_id": reg["ocorrencia"], # Evita duplicação (Upsert)
                "created_at": created_at_dt,
                "latitude": reg["latitude"],
                "longitude": reg["longitude"],
                "tipo_desc": reg.get("tipo_desc", "DESCONHECIDO"),
                "subtipo_desc": reg.get("subtipo_desc"),
                "grupo": reg.get("grupo"), # A nossa futura referência de AIS
                "hex_id": hex_id,
                "loc": {
                    "type": "Point",
                    "coordinates": [reg["longitude"], reg["latitude"]]
                },
                # Novos campos operacionais
                "criacao": parse_data(reg.get("criacao")),
                "despacho": parse_data(reg.get("despacho")),
                "em_rota": parse_data(reg.get("em_rota")),
                "chegada": parse_data(reg.get("chegada")),
                "encerramento": parse_data(reg.get("encerramento")),
                "viaturas_atribuidas": reg.get("viaturas_atribuidas")
            }
            
            # Insere ou Atualiza (Evita duplicação se rodar 2 vezes)
            db.ocorrencias_brutas.update_one({"_id": doc["_id"]}, {"$set": doc}, upsert=True)
            operacoes_inseridas += 1
            
            # Guarda para reconstruir as features
            df_features_raw.append({
                "hex_id": hex_id,
                "janela_tempo": created_at_dt.replace(minute=0, second=0, microsecond=0), # Arredonda para a hora cheia
                "peso": obter_peso_crime(doc["tipo_desc"])
            })
            
        except Exception as e:
            print(f"⚠️ Erro ao processar registro {reg.get('ocorrencia')}: {e}")

    print(f"✅ {operacoes_inseridas} Ocorrências Brutas sincronizadas!")

    print("4. Reconstruindo Malha de Treinamento (Features Históricas)...")
    if not df_features_raw:
        print("❌ Sem dados para reconstruir features.")
        return

    # Agrupa por Local (hex_id) e por Hora (janela_tempo) para somar os pesos
    df = pd.DataFrame(df_features_raw)
    df_agrupado = df.groupby(['hex_id', 'janela_tempo'])['peso'].sum().reset_index()
    df_agrupado.rename(columns={'peso': 'score_risco_total'}, inplace=True)
    
    # Adiciona as features temporais contínuas que a LSTM precisa
    df_agrupado['hora'] = df_agrupado['janela_tempo'].dt.hour
    df_agrupado['dia'] = df_agrupado['janela_tempo'].dt.weekday
    
    df_agrupado['hora_sin'] = np.sin(2 * np.pi * df_agrupado['hora'] / 24)
    df_agrupado['hora_cos'] = np.cos(2 * np.pi * df_agrupado['hora'] / 24)
    df_agrupado['dia_sin'] = np.sin(2 * np.pi * df_agrupado['dia'] / 7)
    df_agrupado['dia_cos'] = np.cos(2 * np.pi * df_agrupado['dia'] / 7)
    df_agrupado['peso_cobertura'] = 1.0 # Assumindo padrão, a API sobrescreve em tempo real

    # Limpa as colunas auxiliares
    df_agrupado = df_agrupado.drop(columns=['hora', 'dia'])

    # Atualiza o MongoDB
    features_inseridas = 0
    for _, row in df_agrupado.iterrows():
        doc_feature = row.to_dict()
        # Busca se já existe aquela hora naquele quarteirão para somar o risco, senão cria novo
        filtro = {"hex_id": doc_feature['hex_id'], "janela_tempo": doc_feature['janela_tempo']}
        update = {
            "$inc": {"score_risco_total": doc_feature['score_risco_total']},
            "$set": {
                "hora_sin": doc_feature['hora_sin'],
                "hora_cos": doc_feature['hora_cos'],
                "dia_sin": doc_feature['dia_sin'],
                "dia_cos": doc_feature['dia_cos'],
                "peso_cobertura": doc_feature['peso_cobertura']
            }
        }
        db.features_historicas.update_one(filtro, update, upsert=True)
        features_inseridas += 1

    print(f"✅ {features_inseridas} blocos espaço-temporais atualizados nas Features Históricas.")
    print("🚀 PROCESSO CONCLUÍDO! O banco está pronto para treinar a I.A.")

if __name__ == "__main__":
    importar_dados()