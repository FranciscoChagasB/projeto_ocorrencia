import pandas as pd
import h3
import glob
import os

# Importa a engenharia de features que criamos
from features import enriquecer_dados_ia 

def corrigir_coordenada(valor, tipo='lat'):
    """
    Força qualquer lixo do Excel a virar uma coordenada válida do Ceará.
    """
    if pd.isna(valor): return valor
    try:
        # 1. Converte pra string, troca vírgula por ponto e lê como Float (resolve o "E+15")
        v_str = str(valor).replace(',', '.').upper()
        v_float = float(v_str)
        
        if v_float == 0: return 0.0

        # 2. Divide o número gigante até a vírgula cair no lugar certo
        if tipo == 'lat':
            # Latitude do Ceará fica entre -2 e -8. Divide por 10 até chegar lá.
            while v_float <= -10 or v_float >= 10:
                v_float /= 10.0
        elif tipo == 'lon':
            # Longitude do Ceará fica entre -37 e -42.
            while v_float <= -100 or v_float >= 100:
                v_float /= 10.0
            # Se a longitude encolheu demais (ex: virou -3.8 em vez de -38)
            while v_float > -10 and v_float < 10:
                v_float *= 10.0
                
        return float(v_float)
    except Exception as e:
        return valor

def carregar_dados_excel(caminho_cameras, pasta_ocorrencias):
    print("1. Carregando câmeras...")
    df_cameras = pd.read_excel(caminho_cameras)
    
    print(f"2. Buscando arquivos de ocorrências em: {pasta_ocorrencias}")
    arquivos_excel = glob.glob(os.path.join(pasta_ocorrencias, "*.xlsx"))
    
    if not arquivos_excel:
        raise FileNotFoundError(f"Nenhum arquivo .xlsx encontrado na pasta {pasta_ocorrencias}")
    
    lista_dfs_ocorrencias = []
    for arquivo in arquivos_excel:
        print(f"   Lendo: {os.path.basename(arquivo)}")
        lista_dfs_ocorrencias.append(pd.read_excel(arquivo))
    
    df_ocorrencias = pd.concat(lista_dfs_ocorrencias, ignore_index=True)
    
    # APLICAÇÃO DO ESCUDO NAS COLUNAS DE COORDENADAS
    print(" -> Limpando e formatando anomalias geográficas (vírgulas e notação científica)...")
    df_ocorrencias['latitude'] = df_ocorrencias['latitude'].apply(lambda x: corrigir_coordenada(x, 'lat'))
    df_ocorrencias['longitude'] = df_ocorrencias['longitude'].apply(lambda x: corrigir_coordenada(x, 'lon'))
    
    df_cameras['latitude'] = df_cameras['latitude'].apply(lambda x: corrigir_coordenada(x, 'lat'))
    df_cameras['longitude'] = df_cameras['longitude'].apply(lambda x: corrigir_coordenada(x, 'lon'))
    
    print(f"   Total de ocorrências carregadas e corrigidas: {len(df_ocorrencias)}")
    
    return df_ocorrencias, df_cameras

def processar_dados_ia(df_ocorrencias, df_cameras, resolucao_h3=9, janela_horas=6):
    print("3. Convertendo coordenadas para Hexágonos H3 e fatiando o tempo...")
    
    df_ocorrencias['hex_id'] = df_ocorrencias.apply(
        lambda row: h3.latlng_to_cell(row['latitude'], row['longitude'], resolucao_h3), axis=1
    )
    df_cameras['hex_id'] = df_cameras.apply(
        lambda row: h3.latlng_to_cell(row['latitude'], row['longitude'], resolucao_h3), axis=1
    )

    df_ocorrencias['data_hora'] = pd.to_datetime(df_ocorrencias['data_hora'])
    df_ocorrencias['janela_tempo'] = df_ocorrencias['data_hora'].dt.floor(f'{janela_horas}h')

    # Em vez de apenas contar, passamos para o features.py aplicar a Gravidade e o K-Ring
    print("4. Aplicando Matrizes de Risco e Tempo Cíclico...")
    df_final_enriquecido = enriquecer_dados_ia(df_ocorrencias, df_cameras, resolucao_h3)

    return df_final_enriquecido

if __name__ == "__main__":
    CAMINHO_CAMERAS = r"projeto_ocorrencia\dados\cameras.xlsx"
    PASTA_OCORRENCIAS = r"projeto_ocorrencia\dados\ocorrencias_diarias"
    
    try:
        df_oco, df_cam = carregar_dados_excel(CAMINHO_CAMERAS, PASTA_OCORRENCIAS)
        df_pronto_para_ia = processar_dados_ia(df_oco, df_cam)
        
        # Salva exatamente com o nome que o treinar_modelo.py está esperando
        caminho_saida = r"dados\features_treinamento.csv"
        df_pronto_para_ia.to_csv(caminho_saida, index=False)
        print(f"\n✅ Pipeline concluído com sucesso! Arquivo '{caminho_saida}' gerado.")
        print("Agora você já pode rodar o script 'treinar_modelo.py'!")
        
    except Exception as e:
        print(f"\n❌ Erro na execução: {e}")