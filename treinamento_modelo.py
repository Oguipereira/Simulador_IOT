import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import os
 
print("Treinamento modelo")
 
 
print("CARREGANDO DADOS")
 
df = pd.read_parquet('dados/telemetria.parquet')
print(f"Dados carregados: {len(df):,} linhas")

print("Preparando os dados para a parada rolar")