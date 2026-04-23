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
    "tb_ocorrencias_202604220721.json"
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
    print("🧠 PROCESSANDO FEATURES (Aplicando Smart Padding)")

    if not features_raw:
        print("❌ Sem dados para features")
        return

    df = pd.DataFrame(features_raw)

    # Agrupa as ocorrências da mesma hora/local
    df_agrupado = df.groupby(['hex_id', 'janela_tempo'])['peso'].sum().reset_index()
    df_agrupado.rename(columns={'peso': 'score_risco_total'}, inplace=True)
    
    # ---------------------------------------------------------
    # SMART PADDING (Preenchimento Contínuo de Tempo)
    # ---------------------------------------------------------
    print("   ↳ Criando linha do tempo contínua para os locais ativos...")
    min_tempo = df_agrupado['janela_tempo'].min()
    max_tempo = df_agrupado['janela_tempo'].max()
    
    # Gera uma lista com TODAS as horas entre a data mais antiga e a mais nova
    todas_horas = pd.date_range(start=min_tempo, end=max_tempo, freq='h')
    # Pega apenas os hexágonos que tiveram algum crime
    hex_ativos = df_agrupado['hex_id'].unique()
    
    # Cria uma matriz gigante com TODAS as horas para TODOS os hex_ativos
    index_completo = pd.MultiIndex.from_product([hex_ativos, todas_horas], names=['hex_id', 'janela_tempo'])
    
    # Faz o merge da matriz gigante com os nossos crimes. Onde não houve crime, preenche com 0.0
    df_agrupado = df_agrupado.set_index(['hex_id', 'janela_tempo']).reindex(index_completo).fillna({'score_risco_total': 0.0}).reset_index()
    # ---------------------------------------------------------

    # Engenharias de Features Temporais
    df_agrupado['hora'] = df_agrupado['janela_tempo'].dt.hour
    df_agrupado['dia'] = df_agrupado['janela_tempo'].dt.weekday

    df_agrupado['hora_sin'] = np.sin(2 * np.pi * df_agrupado['hora'] / 24)
    df_agrupado['hora_cos'] = np.cos(2 * np.pi * df_agrupado['hora'] / 24)
    df_agrupado['dia_sin'] = np.sin(2 * np.pi * df_agrupado['dia'] / 7)
    df_agrupado['dia_cos'] = np.cos(2 * np.pi * df_agrupado['dia'] / 7)
    df_agrupado['peso_cobertura'] = 1.0

    df_agrupado = df_agrupado.drop(columns=['hora', 'dia'])

    # Salvar no MongoDB
    # Como recriámos a matriz do zero para garantir consistência temporal,
    # vamos apagar a coleção de features antiga (a brute não é afetada) e inserir em massa.
    print("   ↳ Inserindo matriz espaço-temporal no banco...")
    db.features_historicas.delete_many({})
    
    registos = df_agrupado.to_dict(orient='records')
    total = len(registos)
    
    # Inserção em lotes (Bulk Insert) para não sobrecarregar a RAM do MongoDB
    tamanho_lote = 50000
    for i in range(0, total, tamanho_lote):
        db.features_historicas.insert_many(registos[i : i + tamanho_lote])

    print(f"✅ {total} features (crimes + horas de paz) atualizadas com sucesso!")


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