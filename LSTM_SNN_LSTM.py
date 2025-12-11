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

