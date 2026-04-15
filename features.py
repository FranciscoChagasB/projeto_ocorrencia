import pandas as pd
import numpy as np
import h3

# 1. MATRIZ DE GRAVIDADE TÁTICA (PESOS ATUALIZADOS)
PESOS_OCORRENCIA = {
    # NÍVEL 5: Risco à Vida, Integridade e Liberdade (Prioridade Máxima)
    'homicídio': 10.0,
    'estupro': 10.0,
    'sequestro': 10.0,
    'tentativa de homicídio': 9.0,
    
    # NÍVEL 4: Armas, Facções e Violência Grave
    'disparo de arma': 8.5,
    'grupo criminoso': 8.5,
    'maria da penha': 8.5, # Alta urgência de despacho
    'porte ilegal de arma': 8.0,
    'roubo de veículo': 8.0,
    'roubo': 7.0,

    # NÍVEL 3: Tráfico, Lesões e Quebra de Condicional
    'lesão corporal': 6.0,
    'tráfico': 6.0,
    'tráfico de drogas em eventos esportivos': 6.0,
    'importunação sexual': 6.0,
    'violação da monitoração eletrônica de pessoas': 5.0, # Risco de reincidência

    # NÍVEL 2: Crimes contra o Patrimônio (Sem violência direta)
    'furto de veículo': 4.0,
    'furto': 3.0,
    'danos/depredação': 3.0,

    # NÍVEL 1: Desordem e Baixo Potencial Ofensivo
    'consumo de entorpecentes': 2.0,
    'embriaguez e desordem': 2.0,
    'perturbação ao sossego alheio': 1.0,

    # TIPO NÃO MAPEADO (Gatilho de Segurança)
    'default': 1.0
}

def obter_peso(tipo):
    # Padroniza a string do Excel (remove espaços sobrando e joga para minúsculo)
    tipo_formatado = str(tipo).strip().lower()
    return PESOS_OCORRENCIA.get(tipo_formatado, PESOS_OCORRENCIA['default'])

# 2. ENGENHARIA DE FEATURES AVANÇADAS
def enriquecer_dados_ia(df_ocorrencias, df_cameras, resolucao_h3=9): # Ajustado para 9 (maior precisão tática de rua)
    print("Aplicando Matriz de Gravidade...")
    
    # 1. Aplicar os pesos nas ocorrências
    df_ocorrencias['peso_gravidade'] = df_ocorrencias['tipo_ocorrencia'].apply(obter_peso)
    
    # 2. Agregação Temporal com Score de Risco
    tensor_risco = df_ocorrencias.groupby(['hex_id', 'janela_tempo']).agg(
        contagem_bruta=('id', 'count'),
        score_risco_total=('peso_gravidade', 'sum') 
    ).reset_index()

    # 3. Codificação Cíclica do Tempo (O "Relógio" da I.A.)
    print("Gerando Embeddings Temporais Cíclicos...")
    tensor_risco['hora'] = tensor_risco['janela_tempo'].dt.hour
    tensor_risco['dia_semana'] = tensor_risco['janela_tempo'].dt.dayofweek
    
    tensor_risco['hora_sin'] = np.sin(2 * np.pi * tensor_risco['hora'] / 24)
    tensor_risco['hora_cos'] = np.cos(2 * np.pi * tensor_risco['hora'] / 24)
    tensor_risco['dia_sin'] = np.sin(2 * np.pi * tensor_risco['dia_semana'] / 7)
    tensor_risco['dia_cos'] = np.cos(2 * np.pi * tensor_risco['dia_semana'] / 7)

    # 4. Raio de Influência das Câmeras (Suavização Espacial)
    print("Calculando área de influência tática das câmeras...")
    cobertura_expandida = []
    
    for _, row in df_cameras.iterrows():
        hex_central = row['hex_id']
        cobertura_expandida.append({'hex_id': hex_central, 'peso_cobertura': 1.0})
        
        vizinhos = h3.grid_disk(hex_central, 1)
        for vizinho in vizinhos:
            if vizinho != hex_central:
                cobertura_expandida.append({'hex_id': vizinho, 'peso_cobertura': 0.3})
                
    df_cobertura = pd.DataFrame(cobertura_expandida)
    df_cobertura_agrupada = df_cobertura.groupby('hex_id')['peso_cobertura'].sum().reset_index()

    # 5. Merge Final
    df_final = pd.merge(tensor_risco, df_cobertura_agrupada, on='hex_id', how='left')
    df_final['peso_cobertura'] = df_final['peso_cobertura'].fillna(0)

    return df_final