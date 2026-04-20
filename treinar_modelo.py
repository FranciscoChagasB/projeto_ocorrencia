import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
from pymongo import MongoClient

# Importando as classes do seu motor (Certifique-se que o motor_ia.py está na mesma pasta)
from motor_ia import OcorrenciasDataset, PrevisorOcorrencias

# ==========================================
# CONFIGURAÇÕES DE CONEXÃO
# ==========================================
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
NOME_BANCO = "ocorrencia_tatica"

def treinar_ia(epocas=100, batch_size=32, learning_rate=0.001):
    print("1. Conectando ao MongoDB e extraindo Features...")
    
    cliente = MongoClient(MONGO_URI)
    db = cliente[NOME_BANCO]
    col_features = db['features_historicas']

    # Busca todas as features calculadas pelo pipeline
    cursor = col_features.find({}, {"_id": 0})
    df_features = pd.DataFrame(list(cursor))

    if df_features.empty:
        print("❌ ERRO: A coleção 'features_historicas' está vazia! Rode o pipeline/reconstrução antes.")
        return

    # IMPORTANTE PARA LSTM: Ordenar por local e tempo para a rede entender a sequência
    df_features = df_features.sort_values(['hex_id', 'janela_tempo'])

    print(f"   ✅ {len(df_features)} blocos de tempo carregados para treinamento.")

    # 2. Criar o Dataset e o DataLoader
    print("2. Fatiando a linha do tempo (Janela Deslizante)...")
    # passos_historico=7 (Olha 1 semana para trás) 
    # previsao_futura=48 (Prevê o somatório ou a curva das próximas 48 horas)
    dataset = OcorrenciasDataset(df_features, passos_historico=7, previsao_futura=48)
    
    if len(dataset) == 0:
        print("❌ ERRO: Dados insuficientes para criar janelas de treino. Insira mais ocorrências.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3. Inicializar o Modelo
    # input_size=6 (score_risco, hora_sin, hora_cos, dia_sin, dia_cos, peso_cobertura)
    print("3. Instanciando a Rede Neural LSTM...")
    modelo = PrevisorOcorrencias(input_size=6)
    
    criterio_erro = nn.MSELoss()
    otimizador = optim.Adam(modelo.parameters(), lr=learning_rate)

    # 4. O Loop de Treinamento
    print(f"\n🚀 Iniciando Treinamento Tático ({epocas} épocas)...")
    modelo.train()
    
    for epoca in range(epocas):
        erro_total_epoca = 0.0
        
        for batch_X, batch_y, _ in dataloader:
            otimizador.zero_grad()
            previsoes = modelo(batch_X)
            
            # Garante que as dimensões batem [Batch, 1]
            erro = criterio_erro(previsoes, batch_y.view(-1, 1))
            
            erro.backward()
            otimizador.step()
            
            erro_total_epoca += erro.item()
            
        erro_medio = erro_total_epoca / len(dataloader)
        
        if (epoca + 1) % 10 == 0 or epoca == 0:
            print(f"Época [{epoca+1}/{epocas}] | Perda (Loss): {erro_medio:.6f}")

    # 5. Salvar o Cérebro Atualizado
    caminho_modelo = "modelo_tatico_lstm.pth"
    torch.save(modelo.state_dict(), caminho_modelo)
    print(f"\n✅ Treinamento concluído com sucesso!")
    print(f"🎯 O arquivo '{caminho_modelo}' agora reflete os 31k registros reais.")

if __name__ == "__main__":
    # Como agora os dados são reais e limpos, 200 a 500 épocas costumam ser suficientes
    treinar_ia(epocas=500, batch_size=32)