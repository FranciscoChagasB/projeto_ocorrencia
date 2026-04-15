import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pymongo import MongoClient
from datetime import datetime, timedelta
import os

# Importa as classes do seu motor
from motor_ia import PrevisorOcorrencias, OcorrenciasDataset

print("🔌 Conectando ao Banco de Dados...")
client = MongoClient("mongodb://localhost:27017/")
db = client["ocorrencia_tatico"]

print("📥 Extraindo histórico recente para treino...")
# Pega os últimos 14 dias para não sobrecarregar a RAM (ajuste se tiver muita RAM)
limite = datetime.now() - timedelta(days=14)
cursor = db.features_historicas.find({"janela_tempo": {"$gte": limite}}, {"_id": 0})
df = pd.DataFrame(list(cursor))

if df.empty:
    print("❌ Sem dados para treinar!")
    exit()

print(f"📊 Dados originais extraídos: {len(df)} registros.")

# ==========================================
# O TRUQUE ANTI-PREGUIÇA (Undersampling)
# ==========================================
print("⚖️ Equilibrando a balança (Escondendo o excesso de paz)...")

# Separa as horas com crime das horas pacíficas
df_crimes = df[df['score_risco_total'] > 0]
df_paz = df[df['score_risco_total'] == 0]

# O "Pulo do Gato": Para cada crime, mantemos apenas 10 horas de paz.
# O resto, jogamos fora apenas no momento do treino!
limite_paz = len(df_crimes) * 10 

if len(df_paz) > limite_paz:
    df_paz = df_paz.sample(n=limite_paz, random_state=42)

# Junta tudo e baralha
df_equilibrado = pd.concat([df_crimes, df_paz]).sample(frac=1).reset_index(drop=True)
df_equilibrado = df_equilibrado.sort_values(by=['hex_id', 'janela_tempo'])

print(f"⚖️ Dados após o corte: {len(df_equilibrado)} registros (Crimes: {len(df_crimes)} | Paz: {len(df_paz)})")

# ==========================================
# INÍCIO DO TREINAMENTO
# ==========================================
print("🧠 Preparando os dados Espaço-Temporais...")
dataset = OcorrenciasDataset(df_equilibrado, passos_historico=7, previsao_futura=1)

# Se ainda ficar pesado, use DataLoader com batch_size
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

modelo = PrevisorOcorrencias(input_size=6)
criterio = nn.MSELoss()
otimizador = optim.Adam(modelo.parameters(), lr=0.001)

epocas = 20
print(f"🚀 Iniciando retreino tático por {epocas} épocas...")

for epoca in range(epocas):
    erro_total = 0
    modelo.train()
    
    for lote_x, lote_y, _ in dataloader:
        otimizador.zero_grad()
        
        # Faz a previsão
        previsoes = modelo(lote_x)
        
        # Calcula o erro em relação à realidade
        loss = criterio(previsoes, lote_y)
        
        # Ajusta os "neurónios"
        loss.backward()
        otimizador.step()
        
        erro_total += loss.item()
        
    print(f"   -> Época [{epoca+1}/{epocas}] | Erro Médio (Loss): {erro_total/len(dataloader):.4f}")

# Salva o novo cérebro
diretorio_base = os.path.dirname(os.path.abspath(__file__))
caminho_modelo = os.path.join(diretorio_base, "modelo_tatico_lstm.pth")
torch.save(modelo.state_dict(), caminho_modelo)

print("✅ SUCESSO! O novo cérebro foi salvo.")
print("Reinicie a API para carregar a nova inteligência agressiva!")