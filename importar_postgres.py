import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from pymongo import MongoClient
import h3

# ==========================================
# CONFIGURAÇÕES
# ==========================================
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
NOME_BANCO = "ocorrencia_tatica"

ARQUIVOS_JSON = [
    "dados_postgres.json",
    "dados_postgres1.json"
]

# ==========================================
# REGRAS DE NEGÓCIO
# ==========================================
def obter_peso_crime(tipo_desc):
    if not tipo_desc:
        return 3.0

    t = str(tipo_desc).upper()

    if 'HOMIC' in t:
        return 10.0
    if 'ARMA' in t:
        return 8.5
    if 'PENHA' in t:
        return 8.5
    if 'GRUPO CRIMINOSO' in t:
        return 8.5
    if 'ROUBO' in t:
        return 7.0
    if 'FURTO' in t:
        return 5.0
    if 'PERTURBA' in t:
        return 1.0

    return 3.0


def parse_data(data_str):
    if not data_str:
        return None

    try:
        dt_str = str(data_str)[:19]
        return datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def extrair_registros(dados_brutos):
    chave_raiz = list(dados_brutos.keys())[0]
    return dados_brutos[chave_raiz]


# ==========================================
# ETAPA 1 - EXTRACT
# ==========================================
def extrair_dados():
    print("📥 EXTRAÇÃO DOS DADOS")

    registros = []

    for arquivo in ARQUIVOS_JSON:
        print(f"📂 Lendo {arquivo}...")
        try:
            with open(arquivo, 'r', encoding='utf-8') as f:
                dados_brutos = json.load(f)
                registros.extend(extrair_registros(dados_brutos))
        except Exception as e:
            print(f"❌ Erro ao ler {arquivo}: {e}")

    print(f"🔢 Total bruto carregado: {len(registros)}")
    return registros


# ==========================================
# ETAPA 2 - TRANSFORM
# ==========================================
def transformar_dados(registros):
    print("🔄 TRANSFORMAÇÃO DOS DADOS")

    registros_processados = []
    features_raw = []
    ids_processados = set()

    for reg in registros:
        try:
            ocorrencia_id = reg.get("ocorrencia")

            # Filtro 1: somente IDs válidos
            if not ocorrencia_id or not ocorrencia_id.startswith("M"):
                continue

            # Filtro 2: remover duplicados
            if ocorrencia_id in ids_processados:
                continue

            ids_processados.add(ocorrencia_id)

            tipo_desc = reg.get("tipo_desc", "DESCONHECIDO")

            # 🔥 NOVO FILTRO: remover perturbação de sossego
            if tipo_desc and any(p in tipo_desc.upper() for p in ["PERTURBA", "SOSSEGO"]):
                continue

            lat = reg.get("latitude")
            lon = reg.get("longitude")

            # Filtro 3: coordenadas válidas
            if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
                continue

            # H3
            try:
                hex_id = h3.latlng_to_cell(lat, lon, 9)
            except AttributeError:
                hex_id = h3.geo_to_h3(lat, lon, 9)

            # Datas
            created_at = parse_data(reg.get("created_at")) or parse_data(reg.get("criacao"))

            if not created_at:
                continue

            # Documento Mongo
            doc = {
                "_id": ocorrencia_id,
                "created_at": created_at,
                "latitude": lat,
                "longitude": lon,
                "tipo_desc": tipo_desc,
                "subtipo_desc": reg.get("subtipo_desc"),
                "grupo": reg.get("grupo"),
                "hex_id": hex_id,
                "loc": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                },
                "criacao": parse_data(reg.get("criacao")),
                "despacho": parse_data(reg.get("despacho")),
                "em_rota": parse_data(reg.get("em_rota")),
                "chegada": parse_data(reg.get("chegada")),
                "encerramento": parse_data(reg.get("encerramento")),
                "viaturas_atribuidas": reg.get("viaturas_atribuidas")
            }

            registros_processados.append(doc)

            # Feature engineering base
            features_raw.append({
                "hex_id": hex_id,
                "janela_tempo": created_at.replace(minute=0, second=0, microsecond=0),
                "peso": obter_peso_crime(tipo_desc)
            })

        except Exception as e:
            print(f"⚠️ Erro ao processar {reg.get('ocorrencia')}: {e}")

    print(f"✅ Registros válidos: {len(registros_processados)}")
    return registros_processados, features_raw


# ==========================================
# ETAPA 3 - LOAD (MongoDB)
# ==========================================
def carregar_dados(db, registros):
    print("💾 SALVANDO OCORRÊNCIAS")

    total = 0

    for doc in registros:
        db.ocorrencias_brutas.update_one(
            {"_id": doc["_id"]},
            {"$set": doc},
            upsert=True
        )
        total += 1

    print(f"✅ {total} ocorrências salvas/atualizadas")


# ==========================================
# ETAPA 4 - FEATURE ENGINEERING
# ==========================================
def processar_features(db, features_raw):
    print("🧠 PROCESSANDO FEATURES")

    if not features_raw:
        print("❌ Sem dados para features")
        return

    df = pd.DataFrame(features_raw)

    df = df.groupby(['hex_id', 'janela_tempo'])['peso'].sum().reset_index()
    df.rename(columns={'peso': 'score_risco_total'}, inplace=True)

    df['hora'] = df['janela_tempo'].dt.hour
    df['dia'] = df['janela_tempo'].dt.weekday

    df['hora_sin'] = np.sin(2 * np.pi * df['hora'] / 24)
    df['hora_cos'] = np.cos(2 * np.pi * df['hora'] / 24)
    df['dia_sin'] = np.sin(2 * np.pi * df['dia'] / 7)
    df['dia_cos'] = np.cos(2 * np.pi * df['dia'] / 7)
    df['peso_cobertura'] = 1.0

    df = df.drop(columns=['hora', 'dia'])

    total = 0

    for _, row in df.iterrows():
        doc = row.to_dict()

        db.features_historicas.update_one(
            {
                "hex_id": doc["hex_id"],
                "janela_tempo": doc["janela_tempo"]
            },
            {
                "$inc": {"score_risco_total": doc["score_risco_total"]},
                "$set": {
                    "hora_sin": doc["hora_sin"],
                    "hora_cos": doc["hora_cos"],
                    "dia_sin": doc["dia_sin"],
                    "dia_cos": doc["dia_cos"],
                    "peso_cobertura": doc["peso_cobertura"]
                }
            },
            upsert=True
        )

        total += 1

    print(f"✅ {total} features atualizadas")


# ==========================================
# PIPELINE PRINCIPAL
# ==========================================
def executar_pipeline():
    print("🚀 INICIANDO PIPELINE ETL")

    client = MongoClient(MONGO_URI)
    db = client[NOME_BANCO]

    registros = extrair_dados()
    registros_processados, features_raw = transformar_dados(registros)
    carregar_dados(db, registros_processados)
    processar_features(db, features_raw)

    print("🎯 PIPELINE FINALIZADO COM SUCESSO!")


if __name__ == "__main__":
    executar_pipeline()