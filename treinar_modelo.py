import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

# Importando as classes do nosso motor
from motor_ia import OcorrenciasDataset, PrevisorOcorrencias

def treinar_ia(caminho_dados_csv, epocas=50, batch_size=32, learning_rate=0.001):
    print("1. Carregando e preparando os dados...")
    try:
        df_features = pd.read_csv(caminho_dados_csv)
    except FileNotFoundError:
        print("⚠️ Arquivo CSV não encontrado! Criando dados simulados para teste do motor de treino...")
        # Mock de dados com as 6 colunas necessárias + hex_id e janela_tempo
        df_features = pd.DataFrame({
            'hex_id': ['885f046531fffff'] * 100, # Simulando 100 janelas de tempo em 1 hexágono
            'janela_tempo': pd.date_range(start='2025-01-01', periods=100, freq='4h'),
            'score_risco_total': np.random.uniform(0, 10, 100),
            'hora_sin': np.random.uniform(-1, 1, 100),
            'hora_cos': np.random.uniform(-1, 1, 100),
            'dia_sin': np.random.uniform(-1, 1, 100),
            'dia_cos': np.random.uniform(-1, 1, 100),
            'peso_cobertura': np.random.uniform(0, 2, 100)
        })

    # 2. Criar o Dataset e o DataLoader
    print("2. Fatiando a linha do tempo (Janela Deslizante)...")
    dataset = OcorrenciasDataset(df_features, passos_historico=4, previsao_futura=1)
    
    # O DataLoader agrupa as janelas em "Lotes" (Batches) para a placa de vídeo / CPU processar mais rápido
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3. Inicializar o Modelo, Função de Erro e Otimizador
    print("3. Instanciando a Rede Neural LSTM...")
    modelo = PrevisorOcorrencias(input_size=6)
    
    # MSELoss (Mean Squared Error): Punição padrão para modelos que preveem números contínuos
    criterio_erro = nn.MSELoss()
    
    # Adam: O otimizador mais eficiente atualmente para ajustar os pesos da rede
    otimizador = optim.Adam(modelo.parameters(), lr=learning_rate)

    # 4. O Loop de Treinamento (A "Escola" da I.A.)
    print("\n🚀 Iniciando o Treinamento...")
    modelo.train() # Coloca o modelo em modo de aprendizado
    
    for epoca in range(epocas):
        erro_total_epoca = 0.0
        
        for batch_X, batch_y, _ in dataloader:
            # Passo A: Zerar os gradientes do lote anterior
            otimizador.zero_grad()
            
            # Passo B: Fazer a previsão (Forward pass)
            previsoes = modelo(batch_X)
            
            # Passo C: Calcular o quanto a I.A. errou
            # batch_y vem no formato [tamanho_lote, 1], previsoes também precisa bater o formato
            erro = criterio_erro(previsoes, batch_y)
            
            # Passo D: Aprender com o erro (Backward pass / Backpropagation)
            erro.backward()
            
            # Passo E: Atualizar os "pesos" da rede neural
            otimizador.step()
            
            erro_total_epoca += erro.item()
            
        # Calcula a média do erro nesta época
        erro_medio = erro_total_epoca / len(dataloader)
        
        # Mostra o progresso a cada 10 épocas
        if (epoca + 1) % 10 == 0 or epoca == 0:
            print(f"Época [{epoca+1}/{epocas}] | Erro Médio (Loss): {erro_medio:.4f}")

    # 5. Salvar o Cérebro Treinado
    caminho_modelo = "modelo_tatico_lstm.pth"
    torch.save(modelo.state_dict(), caminho_modelo)
    print(f"\n✅ Treinamento concluído! Pesos da rede salvos em: '{caminho_modelo}'")

if __name__ == "__main__":
    # Aponte para o CSV gerado pelo seu pipeline de features
    treinar_ia(caminho_dados_csv="dados/features_treinamento.csv", epocas=1000)