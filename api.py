from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import py_eureka_client.eureka_client as eureka_client
from sklearn.cluster import KMeans
import requests
import torch
import pandas as pd
import os
import h3
import math
from pymongo import MongoClient
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import numpy as np
import h3
from pymongo import MongoClient
from datetime import datetime, timedelta
from motor_ia import MotorDiagnostico
from collections import defaultdict
from pydantic import BaseModel
from typing import List, Optional
import itertools

from motor_ia import PrevisorOcorrencias, SistemaSugestaoTatica
from analise_avancada import DetetorAnomalias, RastreadorClusters, PosicionamentoEstrategico

MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "ocorrencia_tatica")
PORTA_API_INTERNA = 8080
NOME_APLICACAO = "motor-ia-tatico"
PUBLISH_IP = os.getenv("PUBLISH_IP", "127.0.0.1")
PORTA_EXTERNA = int(os.getenv("PORT", 8080))
EUREKA_SERVER = os.getenv("EUREKA_URL", "http://127.0.0.1:8761/eureka")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MICROSERVICO_CERCAS = os.getenv("API_CERCAS_URL", "http://172.25.132.135:30066")

print("Conectando ao MongoDB...")
try:
    cliente_mongo = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db_global = cliente_mongo[MONGO_DB_NAME]
    cliente_mongo.server_info()
except Exception as e:
    print(f"FATAL: Falha ao conectar no MongoDB ({MONGO_URI}): {e}")

    
def rotina_alimentador():
    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Iniciando Alimentador IA (Modo Denso/Zero-Padding)...")
        
        agora = datetime.now()
        inicio_janela = agora - timedelta(hours=1)
        
        todos_hex_ids = db_global.features_historicas.distinct("hex_id")
        
        crimes = list(db_global.ocorrencias_brutas.find({
            "data_hora": {"$gte": inicio_janela, "$lt": agora}
        }))
        
        volume_total_cidade = len(crimes)
        contagem_crimes = {}

        if volume_total_cidade > 0:
            df = pd.DataFrame(crimes)
            df['hex_id'] = df.apply(lambda x: h3.latlng_to_cell(x['latitude'], x['longitude'], 9), axis=1)
            contagem_crimes = df.groupby('hex_id').size().to_dict()
            
            for hex_novo in contagem_crimes.keys():
                if hex_novo not in todos_hex_ids:
                    todos_hex_ids.append(hex_novo)

        novos_registros = []
        hora = agora.hour
        dia = agora.weekday()
        
        hora_sin = float(np.sin(2 * np.pi * hora / 24))
        hora_cos = float(np.cos(2 * np.pi * hora / 24))
        dia_sin = float(np.sin(2 * np.pi * dia / 7))
        dia_cos = float(np.cos(2 * np.pi * dia / 7))

        for hex_id in todos_hex_ids:
            score_risco = float(contagem_crimes.get(hex_id, 0.0))
            
            novos_registros.append({
                "hex_id": hex_id,
                "janela_tempo": agora,
                "score_risco_total": score_risco,
                "hora_sin": hora_sin,
                "hora_cos": hora_cos,
                "dia_sin": dia_sin,
                "dia_cos": dia_cos,
                "peso_cobertura": 1.0
            })
        
        if novos_registros:
            db_global.features_historicas.insert_many(novos_registros)
            print(f"{len(novos_registros)} registos da malha atualizados. (Crimes: {volume_total_cidade})")

        is_anomalo = motor_anomalias.detetar(hora, dia, volume_total_cidade)
        _salvar_status_anomalia(db_global, agora, hora, dia, volume_total_cidade, bool(is_anomalo))
            
    except Exception as e:
        print(f"Erro no Alimentador: {e}")

def _salvar_status_anomalia(db, timestamp, hora, dia, volume, is_anomalo):
    if is_anomalo:
        status_texto = "ALERTA VERMELHO"
        detalhe = f"Volume de ocorrências ({volume}) criticamente acima da média histórica."
    else:
        status_texto = "TRANQUILO"
        detalhe = f"Volume da cidade ({volume} registros) operando dentro da normalidade."

    documento = {
        "tipo_status": "sensor_anomalia",
        "timestamp": timestamp,
        "hora_analisada": hora,
        "volume_registrado": volume,
        "alerta_critico": is_anomalo,
        "rotulo_frontend": status_texto,
        "detalhamento_frontend": detalhe
    }
    
    db.status_sistema.update_one(
        {"tipo_status": "sensor_anomalia"},
        {"$set": documento},
        upsert=True
    )

async def loop_alimentador():
    while True:
        await asyncio.to_thread(rotina_alimentador)
        await asyncio.sleep(3600)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Iniciando {NOME_APLICACAO} internamente na porta {PORTA_API_INTERNA}...")
    
    if os.getenv("USE_EUREKA", "false").lower() == "true":
        print(f"Registrando no Eureka Server ({EUREKA_SERVER})...")
        print(f"Avisando a rede que estou disponível em: {PUBLISH_IP}:{PORTA_EXTERNA}")
        
        await eureka_client.init_async(
            eureka_server=EUREKA_SERVER,
            app_name=NOME_APLICACAO,
            instance_host=PUBLISH_IP,      
            instance_port=PORTA_EXTERNA    
        )
    
    yield
    
    if os.getenv("USE_EUREKA", "false").lower() == "true":
        print("Desconectando do Eureka...")
        await eureka_client.stop_async()

app = FastAPI(title="API de gerenciamento inteligente de ocorrências", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("1. Inicializando Motores de Inteligência...")
diretorio_base = os.path.dirname(os.path.abspath(__file__))

print("2. Carregando memória dos últimos 7 dias e Infraestrutura...")
diretorio_base = os.path.dirname(os.path.abspath(__file__))
caminho_csv = os.path.join(diretorio_base, "dados", "features_treinamento.csv")
caminho_cameras = os.path.join(diretorio_base, "dados", "cameras.xlsx")

mapa_cobertura = {}
try:
    df_cameras = pd.read_excel(caminho_cameras)
    for _, row in df_cameras.iterrows():
        try:
            h_id = h3.latlng_to_cell(row['latitude'], row['longitude'], 9)
        except AttributeError:
            h_id = h3.geo_to_h3(row['latitude'], row['longitude'], 9)
            
        mapa_cobertura[h_id] = mapa_cobertura.get(h_id, 0) + 1
    print(f"Infraestrutura: {sum(mapa_cobertura.values())} câmeras ativas mapeadas em {len(mapa_cobertura)} hexágonos.")
except Exception as e:
    print(f"Aviso: Não foi possível carregar cameras.xlsx. Usando cobertura zero. Motivo: {e}")

try:
    limite_tempo = datetime.now() - timedelta(days=14)
    cursor = db_global.features_historicas.find({"janela_tempo": {"$gte": limite_tempo}}, {"_id": 0})
    lista_dados = list(cursor)
    
    if len(lista_dados) == 0:
        raise ValueError("Banco de dados vazio!")

    df_real = pd.DataFrame(lista_dados)
    df_real['janela_tempo'] = pd.to_datetime(df_real['janela_tempo'])
    print(f"Memória carregada: {len(df_real)} registros do MongoDB.")
except Exception as e:
    print(f"Usando backup offline. Motivo: {e}")

print("3. Fatiando o tempo e mapeando a cidade...")

agora = datetime.now()
limite_tempo = datetime.now() - timedelta(days=14)
cursor = db_global.features_historicas.find({"janela_tempo": {"$gte": limite_tempo}}, {"_id": 0})
lista_dados = list(cursor)
if len(lista_dados) == 0:
        raise ValueError("Banco de dados vazio!")
df_real = pd.DataFrame(lista_dados)
df_24h = df_real[df_real['janela_tempo'] >= (agora - timedelta(hours=24))]
df_48h = df_real[df_real['janela_tempo'] >= (agora - timedelta(hours=48))]
df_7d  = df_real[df_real['janela_tempo'] >= (agora - timedelta(days=7))]

dict_14d = df_real.groupby('hex_id')['score_risco_total'].sum().to_dict()
dict_7d  = df_7d.groupby('hex_id')['score_risco_total'].sum().to_dict()
dict_48h = df_48h.groupby('hex_id')['score_risco_total'].sum().to_dict()
dict_24h = df_24h.groupby('hex_id')['score_risco_total'].sum().to_dict()

estado_atual_tensores = []
estado_atual_hex_ids = []
estado_atual_coberturas = []

lista_14d, lista_7d, lista_48h, lista_24h = [], [], [], []

colunas_features = ['score_risco_total', 'hora_sin', 'hora_cos', 'dia_sin', 'dia_cos', 'peso_cobertura']

for hex_id, dados_hex in df_real.groupby('hex_id'):
    dados_hex = dados_hex.sort_values('janela_tempo')
    ultimas_janelas = dados_hex.tail(3)[colunas_features].values
    
    if len(ultimas_janelas) < 3:
        pad = np.zeros((3 - len(ultimas_janelas), 6))
        pad[:, 5] = 1.0  
        ultimas_janelas = np.vstack((pad, ultimas_janelas))
        
    estado_atual_tensores.append(torch.FloatTensor(ultimas_janelas))
    estado_atual_hex_ids.append(hex_id)
    
    estado_atual_coberturas.append(float(mapa_cobertura.get(hex_id, 0.0)))
    
    lista_14d.append(dict_14d.get(hex_id, 0.0))
    lista_7d.append(dict_7d.get(hex_id, 0.0))
    lista_48h.append(dict_48h.get(hex_id, 0.0))
    lista_24h.append(dict_24h.get(hex_id, 0.0))

print(f"{len(estado_atual_hex_ids)} hexágonos embalados para a I.A.")

print("4. Acordando as Inteligências Artificiais...")
caminho_modelo = os.path.join(diretorio_base, "modelo_tatico_lstm.pth")
modelo_lstm = PrevisorOcorrencias(input_size=6)

if os.path.exists(caminho_modelo):
    modelo_lstm.load_state_dict(torch.load(caminho_modelo, weights_only=True))
    modelo_lstm.eval()
else:
    print("Aviso: modelo_tatico_lstm.pth não encontrado!")

sistema_tatico = SistemaSugestaoTatica(modelo_lstm)
motor_anomalias = DetetorAnomalias()
motor_diagnostico = MotorDiagnostico()
motor_diagnostico.treinar(df_real)

df_cidade = df_real.groupby('janela_tempo')['score_risco_total'].sum().reset_index(name='contagem_bruta')
df_cidade['hora'] = df_cidade['janela_tempo'].dt.hour
df_cidade['dia_semana'] = df_cidade['janela_tempo'].dt.weekday

motor_anomalias.treinar(df_cidade)
motor_clusters = RastreadorClusters(raio_km=1.5, min_ocorrencias=3)
motor_posicionamento = PosicionamentoEstrategico()

print("API online e aguardando comandos.")

class OcorrenciaInput(BaseModel):
    id: str
    latitude: float
    longitude: float
    tipo: str

class ViaturaConfig(BaseModel):
    id_manual: str
    tipo_crime_foco: str 
    ais_alocada: str 

class RequisicaoPatrulha(BaseModel):
    viaturas: List[ViaturaConfig]
    distancia_max_km: Optional[float] = 2.0

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dLat, dLon = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dLat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def otimizar_rota_tsp(pontos):
    if len(pontos) <= 2:
        return pontos, sum(haversine_km(pontos[i]['lat'], pontos[i]['lon'], pontos[i+1]['lat'], pontos[i+1]['lon']) for i in range(len(pontos)-1))
    
    if len(pontos) <= 7:
        menor_distancia = float('inf')
        melhor_rota = None
        
        for permutacao in itertools.permutations(pontos):
            dist = 0
            for i in range(len(permutacao) - 1):
                dist += haversine_km(permutacao[i]['lat'], permutacao[i]['lon'], permutacao[i+1]['lat'], permutacao[i+1]['lon'])
            
            if dist < menor_distancia:
                menor_distancia = dist
                melhor_rota = list(permutacao)
                
        return melhor_rota, menor_distancia
    else:
        return pontos, 0 # Fallback

def buscar_poligono_da_sua_api(ais_nome: str):

    url = f"http://172.25.132.135:30066/rotas/cerca/geometria/system?nome={ais_nome}"

    try:
        resposta = requests.get(url, timeout=5)
        if resposta.status_code == 200:
            dados = resposta.json()
            
            pontos = dados.get("geom", {}).get("points", [])
            
            coords = [[p["y"], p["x"]] for p in pontos]
            return coords
            
    except Exception as e:
        print(f"Erro de comunicação com o microserviço de cercas para {ais_nome}: {e}")
        
    # Se falhar ou não existir, retorna vazio
    return []

def ponto_dentro_poligono(lat, lon, poligono):
    x, y = lat, lon
    inside = False
    n = len(poligono)
    if n == 0: return False
    p1x, p1y = poligono[0]
    for i in range(1, n + 1):
        p2x, p2y = poligono[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

# ENDPOINTS DA API
@app.get("/")
def health_check():
    return {"status": "online", "motores_ativos": 4, "hexagonos_monitorados": len(estado_atual_hex_ids)}

@app.get("/api/tatico/sugestoes-cameras")
def obter_malha_tatica(ais: list[str] = Query(default=[])):
    if not ais:
        return []

    try:
        # 1. Obter Polígonos das AIS Selecionadas
        poligonos_ativos = []
        bbox_ativos = []
        for ais_nome in ais:
            coords = buscar_poligono_da_sua_api(ais_nome)
            if coords:
                poligonos_ativos.append(coords)
                lats = [p[0] for p in coords]
                lons = [p[1] for p in coords]
                bbox_ativos.append((min(lats), max(lats), min(lons), max(lons)))
                
        if not poligonos_ativos:
            return []

        # 2. Puxar Dados Históricos Dinâmicos do Mongo
        # Vamos usar o relógio real para fatiar o histórico recente
        agora = datetime.now(timezone.utc)
        d24h = agora - timedelta(days=1)
        d48h = agora - timedelta(days=2)
        d7d = agora - timedelta(days=7)
        d14d = agora - timedelta(days=14)

        pipeline = [
            {"$group": {
                "_id": "$hex_id", 
                "peso_historico": {"$sum": "$score_risco_total"},
                # Se a janela de tempo do crime for maior que ontem, soma o risco em hist_24h, senão soma 0
                "hist_24h": {"$sum": {"$cond": [{"$and": [{"$gte": ["$janela_tempo", d24h]}, {"$gt": ["$score_risco_total", 0]}]}, 1, 0]}},
                "hist_48h": {"$sum": {"$cond": [{"$and": [{"$gte": ["$janela_tempo", d48h]}, {"$gt": ["$score_risco_total", 0]}]}, 1, 0]}},
                "hist_7d":  {"$sum": {"$cond": [{"$and": [{"$gte": ["$janela_tempo", d7d]}, {"$gt": ["$score_risco_total", 0]}]}, 1, 0]}},
                "hist_14d": {"$sum": {"$cond": [{"$and": [{"$gte": ["$janela_tempo", d14d]}, {"$gt": ["$score_risco_total", 0]}]}, 1, 0]}}
            }}
        ]
        
        locais = list(db_global['features_historicas'].aggregate(pipeline))
        
        # 3. Correção do Outlier (O fim do "Mapa Tudo Verde")
        # Em vez de pegar no valor absoluto máximo, pegamos no topo dos 5% piores locais (Percentil 95)
        pesos_ordenados = sorted([loc['peso_historico'] for loc in locais])
        if not pesos_ordenados:
            return []
            
        indice_p95 = int(len(pesos_ordenados) * 0.95)
        teto_peso = pesos_ordenados[indice_p95] if indice_p95 < len(pesos_ordenados) and pesos_ordenados[indice_p95] > 0 else 1
        
        hora_atual = agora.hour
        malha_resposta = []
        
        # 4. Cruzamento Geográfico e Cálculo Final
        for loc in locais:
            h_id = loc['_id']
            
            # Filtro de AIS (Ponto dentro do Polígono)
            try:
                lat, lon = h3.cell_to_latlng(h_id)
            except AttributeError:
                lat, lon = h3.h3_to_geo(h_id)
                
            dentro_da_ais = False
            for i, poli in enumerate(poligonos_ativos):
                min_lat, max_lat, min_lon, max_lon = bbox_ativos[i]
                # Passa pelo BBox primeiro (rápido), e se bater, passa pelo Ray Casting (preciso)
                if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                    if ponto_dentro_poligono(lat, lon, poli):
                        dentro_da_ais = True
                        break
            
            if not dentro_da_ais:
                continue

            # A MÁGICA DOS CÁLCULOS
            peso = loc['peso_historico']
            
            # Vulnerabilidade (Estática, baseada no Teto P95)
            vulnerabilidade = (peso / teto_peso) * 100
            vulnerabilidade = min(99.9, max(1.0, vulnerabilidade))
            
            # Risco Atual (Dinâmico) - A I.A. suavizada para não desligar o mapa de manhã
            # O modificador agora flutua entre 0.4 e 1.0 (nunca zera)
            modificador_hora = (math.sin(math.pi * (hora_atual - 6) / 12) * 0.3) + 0.7 
            
            # Bónus Tático: Se a rua teve crime nas últimas 24h, a I.A. dispara o Risco Atual
            pico_recente = 1.25 if loc['hist_24h'] > 0 else 1.0
            
            risco = vulnerabilidade * modificador_hora * pico_recente
            risco = min(99.9, max(1.0, risco))

            malha_resposta.append({
                "hex_id": row['hex_id'],
                "peso_cobertura": row['peso_cobertura'],
                "nivel_prioridade": row['nivel_prioridade'],
                
                # Valores distintos baseados no novo motor de regressão
                "vulnerabilidade_atual": row['vulnerabilidade_atual'],
                "risco_atual": row['risco_atual'],
                
                # A nova métrica: Quantidade de Crimes
                "previsao_ocorrencias_48h": row['previsao_qtd_48h'], 
                
                "hist_24h": int(row['hist_24h']),
                "hist_48h": int(row['hist_48h']),
                "hist_7d": int(row['hist_7d']),
                "hist_14d": int(row['hist_14d'])
            })
            
        return malha_resposta
    except Exception as e:
        print("Erro na malha tática:", e)
        return []

@app.get("/api/tatico/mapa-calor-historico")
def endpoint_mapa_calor_historico(ais: list[str] = Query(default=[])):
    if not ais:
        return []

    poligonos_ativos = []
    bbox_ativos = []
    
    for ais_nome in ais:
        coords = buscar_poligono_da_sua_api(ais_nome)
        
        if not coords:
            continue 
            
        poligonos_ativos.append(coords)
        
        lats = [p[0] for p in coords]
        lons = [p[1] for p in coords]
        bbox_ativos.append((min(lats), max(lats), min(lons), max(lons)))

    ocorrencias_brutas = list(db_global.ocorrencias_brutas.find({}, {"latitude": 1, "longitude": 1, "_id": 0}))
    
    pontos_filtrados = []
    
    for o in ocorrencias_brutas:
        lat = o.get("latitude")
        lon = o.get("longitude")
        
        if lat is None or lon is None: 
            continue
            
        dentro_da_ais = False
        for i, poli in enumerate(poligonos_ativos):
            min_lat, max_lat, min_lon, max_lon = bbox_ativos[i]
            if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                if ponto_dentro_poligono(lat, lon, poli):
                    dentro_da_ais = True
                    break
                    
        if dentro_da_ais:
            pontos_filtrados.append({"lat": lat, "lon": lon})

    return pontos_filtrados

@app.post("/api/tatico/pontos-base")
async def gerar_rotas_patrulha(req: RequisicaoPatrulha):
    try:
        frota_final = []
        DISTANCIA_MIN_KM = 0.0 
        LIMITE_PONTOS_POR_ROTA = 6

        for v in req.viaturas:
            poligono_ais = buscar_poligono_da_sua_api(v.ais_alocada)
            if not poligono_ais:
                continue
            
            foco = v.tipo_crime_foco.upper()
            
            pipeline = [
                {
                    "$match": {
                        "$or": [
                            {"tipo_desc": {"$regex": foco, "$options": "i"}},
                            {"tipo": {"$regex": foco, "$options": "i"}}
                        ]
                    }
                },
                {
                    "$group": {
                        "_id": "$hex_id",
                        "peso": {"$sum": 1},
                        "lat": {"$first": "$latitude"},
                        "lon": {"$first": "$longitude"}
                    }
                },
                {"$sort": {"peso": -1}}
            ]
            
            candidatos = list(db_global['ocorrencias_brutas'].aggregate(pipeline))
            
            pontos_validos = []
            for c in candidatos:
                lat, lon = c['lat'], c['lon']
                if ponto_dentro_poligono(lat, lon, poligono_ais):
                    pontos_validos.append({
                        "lat": lat, "lon": lon, "peso": c['peso']
                    })

            if not pontos_validos:
                continue

            pontos_selecionados = []
            atual = max(pontos_validos, key=lambda x: x['peso'])
            pontos_validos.remove(atual)
            pontos_selecionados.append(atual)

            while pontos_validos and len(pontos_selecionados) < LIMITE_PONTOS_POR_ROTA:
                vizinhos = []
                for p in pontos_validos:
                    d = haversine_km(atual['lat'], atual['lon'], p['lat'], p['lon'])
                    if d <= req.distancia_max_km and d > DISTANCIA_MIN_KM:
                        vizinhos.append((p, d))
                
                if not vizinhos: break 
                
                proximo, dist_segmento = min(vizinhos, key=lambda x: x[1])
                pontos_validos.remove(proximo)
                pontos_selecionados.append(proximo)
                atual = proximo

            melhor_ordem, distancia_total = otimizar_rota_tsp(pontos_selecionados)
            
            rota_formatada = [[p['lon'], p['lat']] for p in melhor_ordem]

            frota_final.append({
                "viatura_id": v.id_manual,
                "foco_especializado": foco,
                "ais": v.ais_alocada,
                "rota": rota_formatada,
                "qtd_pontos": len(rota_formatada),
                "distancia_total_km": round(distancia_total, 2)
            })

        return {"frota": frota_final}
        
    except Exception as e:
        print("Erro nas rotas:", e)
        return {"erro": str(e), "frota": []}

@app.get("/api/tatico/diagnostico-local/{hex_id}")
def diagnostico_local(hex_id: str):
    try:
        cursor = db_global['ocorrencias_brutas'].find({"hex_id": hex_id})
        crimes = list(cursor)
        
        if not crimes:
            return {"tipologia_sugerida": "Área de Monitoramento", "ameacas_especificas": []}

        contagem_tipos = defaultdict(list)
        for c in crimes:
            tipo = c.get('tipo_desc', 'OUTROS').upper()
            dt = c.get('created_at')
            h = int(dt[11:13]) if isinstance(dt, str) else dt.hour
            contagem_tipos[tipo].append(h)

        ameacas = []
        for tipo, horas in contagem_tipos.items():
            perc = (len(horas) / len(crimes)) * 100
            if perc > 10: # Filtra crimes relevantes
                hora_pico = max(set(horas), key=horas.count)
                inicio, fim = (hora_pico - 2) % 24, (hora_pico + 2) % 24
                faixa = f"{inicio:02d}:00 - {fim:02d}:00"
                
                ameacas.append({
                    "crime": tipo,
                    "vulnerabilidade": round(perc, 1),
                    "faixa_horario": faixa,
                    "qtd": len(horas)
                })

        ameacas = sorted(ameacas, key=lambda x: x['vulnerabilidade'], reverse=True)
        return {
            "tipologia_sugerida": f"Foco: {ameacas[0]['crime']}" if ameacas else "Risco Geral",
            "ameacas_especificas": ameacas
        }
    except Exception as e:
        return {"erro": str(e)}

@app.get("/api/tatico/cerca/{ais_nome}")
def obter_cerca_virtual(ais_nome: str):
    try:
        url = f"{MICROSERVICO_CERCAS}/rotas/cerca/geometria/system?nome={ais_nome}"
        resposta = requests.get(url, timeout=5)
        
        if resposta.status_code == 200:
            return resposta.json()
        return {"geom": {"points": []}}
    except Exception as e:
        print(f"Erro de comunicação com o serviço de cercas Java: {e}")
        return {"geom": {"points": []}}

def listar_ocorrencias_local(hex_id: str):
    try:
        limite_dt = datetime.now(timezone.utc) - timedelta(days=14)
        limite_str = limite_dt.strftime("%Y-%m-%d") 
        
        query = {
            "hex_id": hex_id,
            "$or": [
                {"created_at": {"$gte": limite_dt}},
                {"created_at": {"$gte": limite_str}}
            ]
        }
        
        cursor = db_global['ocorrencias_brutas'].find(
            query, 
            {"_id": 1, "tipo_desc": 1, "created_at": 1}
        ).sort("created_at", -1).limit(50)
        
        lista = []
        for doc in cursor:
            data_raw = doc.get("created_at")
            
            if isinstance(data_raw, str):
                data_formatada = data_raw[8:10] + "/" + data_raw[5:7] + " " + data_raw[11:16]
            else:
                data_formatada = data_raw.strftime("%d/%m %H:%M")
                
            lista.append({
                "id": str(doc["_id"]),
                "tipo": doc.get("tipo_desc", "Desconhecido"),
                "data": data_formatada
            })
        return lista
    except Exception as e:
        print(f"Erro ao listar ocorrências: {e}")
        return []

@app.get("/api/tatico/alerta-anomalia")
def endpoint_alerta_anomalia():
    db = cliente_mongo["ocorrencia_tatico"]
    
    status = db.status_sistema.find_one({"tipo_status": "sensor_anomalia"}, {"_id": 0})
    
    if not status:
        return {
            "alerta_critico": False,
            "rotulo_frontend": "INICIALIZANDO",
            "detalhamento_frontend": "Aguardando a primeira leitura dos sensores da cidade.",
            "hora_analisada": 0,
            "volume_registrado": 0,
            "ultima_atualizacao": "N/A"
        }

    ultima_att = status["timestamp"].strftime("%H:%M")

    return {
        "alerta_critico": status["alerta_critico"],
        "rotulo_frontend": status["rotulo_frontend"],
        "detalhamento_frontend": status["detalhamento_frontend"],
        "hora_analisada": status["hora_analisada"],
        "volume_registrado": status["volume_registrado"],
        "ultima_atualizacao": ultima_att
    }