# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 18:34:32 2025

@author: julio
"""

import pandas as pd 
import numpy as np 

df = pd.read_excel('df_residencial.xlsx')

# limpando linhas maiores que > 0 
colunas_booleanas = ['E_COLI', 'RECOLETA', 'RECLAMACAO', 'COLIFORMES_TOTAIS']



df_anomalia = df 

df_anomalia = df_anomalia[(df_anomalia[colunas_booleanas] == 0).all(axis=1)]

# criar sentido de seres temporais sazonalidade 

def processar_coletas(df_anomalia):
    """Processa independente do número inicial"""
    
    # 1. Encontrar o menor frasco de CADA DIA
    df_anomalia['MENOR_FRASCO_DIA'] = df_anomalia.groupby('DATA')['N_FRASCO'].transform('min')
    
    # 2. Calcular DESVIO do menor (isso é universal!)
    df_anomalia['DESVIO_DO_INICIO'] = df_anomalia['N_FRASCO'] - df_anomalia['MENOR_FRASCO_DIA']
    
    # 3. Posição ABSOLUTA (1ª, 2ª, 3ª coleta)
    df_anomalia['POSICAO_ABSOLUTA'] =df_anomalia['DESVIO_DO_INICIO'] + 1
    
    # Exemplo:
    # Se menor frasco = 3 → frasco 3 = posição 1
    # Se menor frasco = 1 → frasco 1 = posição 1
    # SEMPRE funciona!
    
    return df_anomalia

# Aplicar
df_anomalia = processar_coletas(df_anomalia)


# podemos criar alertas de desvio 
df_anomalia['INICIO_TARDIO'] = (df_anomalia['DESVIO_DO_INICIO'] > 2).astype(int)