import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.ensemble import RandomForestClassifier

# 1. PREPARAÇÃO DOS DADOS (JANELA DESLIZANTE)
class OcorrenciasDataset(Dataset):
    def __init__(self, df_features, passos_historico=7, previsao_futura=1):
        """
        Transforma o DataFrame em matrizes 3D para a rede neural ler.
        passos_historico: Quantas janelas no passado a IA vai olhar (ex: 7).
        """
        self.X = []
        self.y = []
        self.hex_ids = []
        
        # Agrupamos por região (hexágono)
        grupos = df_features.groupby('hex_id')
        
        # As 6 colunas que definimos no features.py
        colunas_features = ['score_risco_total', 'hora_sin', 'hora_cos', 'dia_sin', 'dia_cos', 'peso_cobertura']
        
        for hex_id, dados_hex in grupos:
            dados_hex = dados_hex.sort_values('janela_tempo')
            valores = dados_hex[colunas_features].values
            
            # Cria o histórico contínuo (Janela Deslizante)
            for i in range(len(valores) - passos_historico - previsao_futura + 1):
                janela_x = valores[i : i + passos_historico]
                
                # Pegamos o vetor com as 48 horas futuras e SOMAMOS (np.sum)
                total_crimes_futuros = np.sum(valores[i + passos_historico : i + passos_historico + previsao_futura, 0])
                
                alvo_y = [total_crimes_futuros]
                
                self.X.append(janela_x)
                self.y.append(alvo_y)
                self.hex_ids.append(hex_id)
                
        # Converte para tensores do PyTorch
        self.X = torch.FloatTensor(np.array(self.X))
        self.y = torch.FloatTensor(np.array(self.y))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.hex_ids[idx]

# 2. A REDE NEURAL ESPAÇO-TEMPORAL (LSTM)
class PrevisorOcorrencias(nn.Module):
    def __init__(self, input_size=6, hidden_layer_size=64, output_size=1):
        """
        input_size: 6 (pois temos 6 features entrando)
        output_size: 1 (pois queremos prever 1 número: o risco futuro)
        """
        super(PrevisorOcorrencias, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        
        # A LSTM processa a linha do tempo 
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        
        # Camadas densas para refinar a decisão
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(hidden_layer_size, 32)
        self.linear2 = nn.Linear(32, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        
        # Pega apenas a saída do último momento de tempo para fazer a previsão
        ultimo_passo = lstm_out[:, -1, :] 
        
        x = self.relu(self.linear1(ultimo_passo))
        previsao_risco = self.linear2(x)
        return previsao_risco

# 3. O SISTEMA DE SUGESTÃO TÁTICA
class SistemaSugestaoTatica:
    def __init__(self, modelo_ia):
        self.modelo = modelo_ia
        self.modelo.eval()

    def calcular_metricas(self, previsao_ia, h_24, h_48, h_7, h_14, cobertura, horizonte):
        import math
        
        # 1. DECAIMENTO TEMPORAL (O Desempate)
        # Dá pesos diferentes consoante a urgência do histórico. 
        # Um crime ontem vale muito mais do que um crime na semana passada.
        peso_criminal = (h_24 * 4.0) + (h_48 * 2.0) + (h_7 * 1.0) + (h_14 * 0.25)
        impacto_historico = math.sqrt(peso_criminal) if peso_criminal > 0 else 0.0
        
        # 2. Risco Base Puro
        risco_base = previsao_ia + impacto_historico
        
        # 3. Fator de Projeção Futura (Curva de Potência Suave)
        # Em vez de explodir para o infinito, a semana que vem (168h) multiplica o risco por apenas ~2.7x
        fator_tempo = math.pow(horizonte / 6.0, 0.3) if horizonte >= 6 else 1.0
        risco_projetado = risco_base * fator_tempo
        
        # 4. A CURVA ASSINTÓTICA DE SATURAÇÃO (O Segredo)
        # Transforma os números num Índice de 0.0 a 99.9 orgânico.
        constante_suavizacao = 0.35 
        score_risco = 100.0 * (1.0 - math.exp(-constante_suavizacao * risco_projetado))
        
        # 5. Vulnerabilidade (Abatimento das Câmeras - até 75% max)
        fator_reducao_escudo = min(0.75, cobertura * 0.15)
        score_vuln = score_risco * (1.0 - fator_reducao_escudo)
        
        return round(score_risco, 1), round(score_vuln, 1)

    def gerar_malha_universal(self, tensores, hex_ids, coberturas, hist_14d, hist_7d, hist_48h, hist_24h):
        sugestoes = []
        if len(tensores) == 0: return pd.DataFrame()
        
        with torch.no_grad():
            lote_historico = torch.stack(tensores)
            todas_previsoes = self.modelo(lote_historico)
            
            for i in range(len(tensores)):
                # ----------------------------------------------------
                # A MÁGICA DA REGRESSÃO (Qtd. de Ocorrências Previstas)
                # ----------------------------------------------------
                # A rede agora devolve um valor bruto que representa a quantidade esperada.
                # Se for negativo (a rede às vezes erra para baixo), zeramos.
                previsao_qtd_bruta = max(0.0, todas_previsoes[i].item())
                
                cobertura = coberturas[i]
                
                # O Histórico é a Vulnerabilidade (O DNA do crime)
                peso_historico = (hist_24h[i] * 4.0) + (hist_48h[i] * 2.0) + (hist_7d[i] * 1.0) + (hist_14d[i] * 0.25)
                
                # 1. A VULNERABILIDADE (Estática)
                # Usamos uma curva suave para transformar o histórico pesado numa percentagem 0-100%
                vulnerabilidade = 100.0 * (1.0 - math.exp(-0.35 * peso_historico))
                
                # 2. O RISCO (Dinâmico + A I.A.)
                # O Risco sobe agressivamente se a I.A. disser que vai haver > 1 ocorrência
                multiplicador_ia = 1.0 + (previsao_qtd_bruta * 0.5) 
                risco_atual = vulnerabilidade * multiplicador_ia
                
                # Limitamos os limites (0.0 a 99.9)
                vulnerabilidade = min(99.9, vulnerabilidade)
                risco_atual = min(99.9, max(1.0, risco_atual))
                
                # Níveis táticos 
                nivel = 1
                if risco_atual >= 70: nivel = 5
                elif risco_atual >= 50: nivel = 4
                elif risco_atual >= 30: nivel = 3
                elif risco_atual >= 15: nivel = 2
                
                sugestoes.append({
                    'hex_id': hex_ids[i],
                    'peso_cobertura': cobertura,
                    'nivel_prioridade': nivel,
                    'risco_atual': round(risco_atual, 1),
                    'vulnerabilidade_atual': round(vulnerabilidade, 1),
                    'previsao_qtd_48h': round(previsao_qtd_bruta, 1), # <-- O CAMPO NOVO QUE VOCÊ PEDIU
                    'risco_1w': round(risco_atual * 1.1, 1), # Apenas como fallback
                    'vulnerabilidade_1w': round(vulnerabilidade, 1),
                    'hist_24h': hist_24h[i],
                    'hist_48h': hist_48h[i],
                    'hist_7d': hist_7d[i],
                    'hist_14d': hist_14d[i]
                })
        
        df_sugestoes = pd.DataFrame(sugestoes)
        df_sugestoes = df_sugestoes.sort_values(by='risco_atual', ascending=False)
        return df_sugestoes

class MotorDiagnostico:
    def __init__(self):
        # Um modelo rápido, leve e altamente explicável
        self.modelo = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.nomes_features = [
            'Câmeras de Segurança', 
            'Histórico 24h', 
            'Histórico 7 Dias', 
            'Histórico 14 Dias',
            'Fator Madrugada/Noite',
            'Fator Fim de Semana'
        ]
        self.treinado = False

    def treinar(self, df_historico):
        """
        Treina o detetive usando o histórico passado para prever o presente,
        garantindo que não há "Data Leakage" (Vazamento de respostas).
        """
        try:
            print("Treinando o Motor de Diagnóstico (Random Forest)...")
            df = df_historico.copy()
            
            # Extraímos o contexto de tempo
            df['hora'] = df['janela_tempo'].dt.hour
            df['dia_semana'] = df['janela_tempo'].dt.weekday
            df['fator_noite'] = df['hora'].apply(lambda h: 1 if h >= 18 or h <= 5 else 0)
            df['fator_fds'] = df['dia_semana'].apply(lambda d: 1 if d >= 5 else 0)
            
            # Ordenamos cronologicamente para garantir que o passado fica atrás do presente
            df = df.sort_values(by=['hex_id', 'janela_tempo'])
            
            # ==========================================
            # A CORREÇÃO DO DATA LEAKAGE:
            # Usamos o shift(1) para dizer à I.A. para olhar para a linha ANTERIOR
            # e tentar adivinhar a linha ATUAL.
            # ==========================================
            df['hist_imediato'] = df.groupby('hex_id')['score_risco_total'].shift(1).fillna(0)
            df['hist_medio']    = df.groupby('hex_id')['score_risco_total'].shift(2).fillna(0) + df['hist_imediato']
            df['hist_antigo']   = df.groupby('hex_id')['score_risco_total'].shift(3).fillna(0) + df['hist_medio']
            
            # A variável alvo continua a ser o que aconteceu HOJE
            df['teve_crime'] = df['score_risco_total'].apply(lambda x: 1 if x > 0 else 0)
            
            if df['teve_crime'].sum() == 0:
                print("Sem crimes suficientes para treinar o detetive.")
                return

            # Agora sim! A I.A. não tem a resposta. Tem de analisar o contexto.
            X = df[['peso_cobertura', 'hist_imediato', 'hist_medio', 'hist_antigo', 'fator_noite', 'fator_fds']]
            y = df['teve_crime']
            
            self.modelo.fit(X, y)
            self.treinado = True
            print("Motor de Diagnóstico treinado com sucesso e sem vícios!")
        except Exception as e:
            print(f"Erro ao treinar Motor de Diagnóstico: {e}")

    def explicar_local(self, cobertura, hist_24h, hist_7d, hist_14d):
        """
        Recebe a foto atual de um hexágono e devolve um relatório em português do porquê do risco.
        """
        if not self.treinado:
            return {"erro": "Modelo não treinado"}

        from datetime import datetime
        agora = datetime.now()
        fator_noite = 1 if agora.hour >= 18 or agora.hour <= 5 else 0
        fator_fds = 1 if agora.weekday() >= 5 else 0

        # 1. TIPOLOGIA SUGERIDA
        tipologia = "Atividade Suspeita Geral"
        if fator_noite == 1 and cobertura == 0:
            tipologia = "Alto Risco: Roubo a Transeunte/Veículo (Ponto Cego Noturno)"
        elif hist_24h > 0:
            tipologia = "Risco Iminente: Repetição de Padrão Criminal (Retorno do Infrator)"
        elif hist_14d > 2 and cobertura > 0:
            tipologia = "Foco de Furtos (Área visada apesar do monitoramento)"

        # 2. FEATURE IMPORTANCE (O que está puxando o risco para cima AQUI?)
        importancias = self.modelo.feature_importances_
        
        # Calculamos o peso local multiplicando a importância global pela realidade da rua
        peso_local = [
            importancias[0] * (1.0 if cobertura == 0 else 0.0), # Só agrava se NÃO tiver câmera
            importancias[1] * (hist_24h * 4.0), 
            importancias[2] * (hist_7d * 2.0), 
            importancias[3] * (hist_14d * 1.0),
            importancias[4] * fator_noite,
            importancias[5] * fator_fds
        ]
        
        soma_pesos = sum(peso_local)
        relatorio_fatores = []
        
        # Se a soma_pesos for maior que zero, significa que há agravantes reais
        if soma_pesos > 0:
            for i, nome in enumerate(self.nomes_features):
                percentual = (peso_local[i] / soma_pesos) * 100.0
                if percentual > 2.0: # Baixamos o filtro para 2% para ser mais detalhista
                    relatorio_fatores.append({
                        "fator": nome,
                        "peso_percentual": round(percentual, 1)
                    })
        else:
            # === A CORREÇÃO DOS HEXÁGONOS VAZIOS ===
            # Se a rua for um "paraíso" (de dia, sem histórico, com câmera), a IA explica isso!
            tipologia = "Monitoramento de Rotina (Risco Baixo)"
            relatorio_fatores.append({
                "fator": "Risco Residual Base (Sem agravantes agudos detetados)",
                "peso_percentual": 100.0
            })
                
        # Ordena do maior causador do problema para o menor
        relatorio_fatores = sorted(relatorio_fatores, key=lambda x: x['peso_percentual'], reverse=True)

        return {
            "tipologia_sugerida": tipologia,
            "fatores_de_risco": relatorio_fatores
        }

# TESTE DE INTEGRIDADE DO ARQUIVO
if __name__ == "__main__":
    print("Testando motor_ia.py...")
    # Criando um modelo zerado apenas para ver se instancia corretamente
    modelo_teste = PrevisorOcorrencias(input_size=6)
    sistema_teste = SistemaSugestaoTatica(modelo_teste)
    print("Motor IA instanciado com sucesso! As classes estão prontas para receber dados.")