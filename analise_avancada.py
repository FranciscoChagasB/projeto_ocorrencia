import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

# 1. MOTOR DE ANOMALIAS (Isolation Forest)
class DetetorAnomalias:
    def __init__(self):
        # O Isolation Forest isola pontos que estão muito fora do padrão estatístico
        self.modelo = IsolationForest(contamination=0.05, random_state=42)
        self.treinado = False

    def treinar(self, df_historico):
        """
        Aprende o que é 'normal' na cidade cruzando a hora, o dia e o volume de crimes.
        """
        features = df_historico[['hora', 'dia_semana', 'contagem_bruta']].values
        self.modelo.fit(features)
        self.treinado = True

    def detetar(self, hora, dia_semana, contagem_atual):
        """
        Retorna True se o volume atual for uma anomalia (ex: protesto, arrastão).
        """
        if not self.treinado: return False
        
        # O modelo retorna -1 para anomalia e 1 para normal
        previsao = self.modelo.predict([[hora, dia_semana, contagem_atual]])
        return previsao[0] == -1

# 2. MOTOR DE CLUSTERIZAÇÃO (DBSCAN Tático)
class RastreadorClusters:
    def __init__(self, raio_km=1.0, min_ocorrencias=3):
        # Converte o raio de KM para graus (aproximado) para o algoritmo entender
        raio_graus = raio_km / 111.32 
        # O DBSCAN agrupa pontos espacialmente densos
        self.modelo = DBSCAN(eps=raio_graus, min_samples=min_ocorrencias)

    def rastrear_manchas(self, df_ocorrencias_recentes):
        """
        Recebe as ocorrências das últimas X horas.
        Retorna os IDs das ocorrências que fazem parte de um 'ataque em série' ou mancha criminal.
        """
        if len(df_ocorrencias_recentes) < 3:
            return [] # Não há dados suficientes para formar um cluster

        coordenadas = df_ocorrencias_recentes[['latitude', 'longitude']].values
        clusters = self.modelo.fit_predict(coordenadas)
        
        # Ignora os ruídos (clusters rotulados como -1)
        df_ocorrencias_recentes['cluster_id'] = clusters
        manchas = df_ocorrencias_recentes[df_ocorrencias_recentes['cluster_id'] != -1]
        
        return manchas.to_dict(orient='records')

# 3. MOTOR PRESCRITIVO (Posicionamento Dinâmico de Viaturas)
class PosicionamentoEstrategico:
    def __init__(self):
        pass

    def calcular_pontos_base(self, dados_totais, qtd_viaturas=3):
        """
        Calcula as bases usando K-Means ponderado offline.
        Rodará em milissegundos, independente de serem 3 ou 500 viaturas.
        """
        if len(dados_totais) == 0: return []
        if len(dados_totais) < qtd_viaturas: qtd_viaturas = len(dados_totais)

        import h3
        pontos = []
        for d in dados_totais:
            lat, lon = h3.cell_to_latlng(d['hex_id'])
            pontos.append([lat, lon, d['score_vulnerabilidade']])
            
        df_pontos = pd.DataFrame(pontos, columns=['lat', 'lon', 'risco'])
        coordenadas = df_pontos[['lat', 'lon']].values

        pesos_risco = df_pontos['risco'].values + 1.0 

        # K-Means offline
        kmeans = KMeans(n_clusters=qtd_viaturas, random_state=42, n_init='auto')
        df_pontos['zona'] = kmeans.fit_predict(coordenadas, sample_weight=pesos_risco)

        posicionamentos = []
        
        for zona in range(qtd_viaturas):
            pontos_zona = df_pontos[df_pontos['zona'] == zona]
            
            # Encontra o ponto exato mais perigoso dentro da zona
            ponto_focal = pontos_zona.loc[pontos_zona['risco'].idxmax()]
            lat_focal, lon_focal = ponto_focal['lat'], ponto_focal['lon']

            posicionamentos.append({
                "viatura_id": f"VTR-{zona + 1}",
                "rua_recomendada": "Centro Tático (Baseado na Malha)",
                "lat": lat_focal,
                "lon": lon_focal,
                "pontos_cobertos": len(pontos_zona),
                "cor_tática": self._gerar_cor_viatura(zona)
            })
            
        return posicionamentos

    def _gerar_cor_viatura(self, indice):
        cores = [[59, 130, 246], [168, 85, 247], [236, 72, 153], [249, 115, 22]]
        return cores[indice % len(cores)]