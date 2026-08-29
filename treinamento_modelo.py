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

# PASSO 2: FEATURE ENGINEERING (preparar dados pra treinamento do modelo)

print("\n\n prerando os dados)")

print("\nCriando novas features...")

df_features = df.copy()

# Feature 1: Hora do dia (padrão de operação muda por hora)
df_features['hora'] = df_features['timestamp'].dt.hour

# Feature 2: Dia da semana (segunda=0, domingo=6)
df_features['dia_semana'] = df_features['timestamp'].dt.dayofweek

# Feature 3: Semana do mês (primeiras semanas vs últimas)
df_features['semana_mes'] = (df_features['timestamp'].dt.day // 7) + 1

# Feature 4: Razão carga/velocidade (quanto peso por velocidade)
# Evita divisão por zero
df_features['razao_carga_vel'] = df_features['peso_carga_kg'] / (df_features['velocidade_kmh'] + 1)

# Feature 5: Indicador de movimento (1 se movendo, 0 se parado)
df_features['em_movimento'] = (df_features['velocidade_kmh'] > 0).astype(int)

# Feature 6: Temperatura normalizada (0-100 escala)
df_features['temp_normalizada'] = np.clip(
    (df_features['temperatura_motor_celsius'] - 60) / 40 * 100,
    0, 100
)

# Feature 7: Combustível já consumido até agora (variação do dia)
df_features['combustivel_consumido'] = 100 - df_features['combustivel_pct']

print(f" Features criadas!")
print(f"  Total de features: 15")
print(f"\n  Originais:")
print(f"    - empilhadeira_id")
print(f"    - operador_id")
print(f"    - velocidade_kmh")
print(f"    - peso_carga_kg")
print(f"    - temperatura_motor_celsius")
print(f"\n  Novas features criadas:")
print(f"    - hora")
print(f"    - dia_semana")
print(f"    - semana_mes")
print(f"    - razao_carga_vel")
print(f"    - em_movimento")
print(f"    - temp_normalizada")
print(f"    - combustivel_consumido")

# Target: o que queremos prever (combustível consumido)
# Usamos "combustível consumido" em vez de "combustível restante"
# Porque é mais direto: quanto o motor gastou?

y = df_features['combustivel_consumido']

# Features: o que usamos pra prever
X = df_features[[
    'empilhadeira_id',
    'operador_id',
    'velocidade_kmh',
    'peso_carga_kg',
    'temperatura_motor_celsius',
    'hora',
    'dia_semana',
    'semana_mes',
    'razao_carga_vel',
    'em_movimento',
    'temp_normalizada'
]]

print(f"Target (y): combustivel_consumido")
print(f"  Tamanho: {len(y):,} amostras")
print(f"  Média: {y.mean():.2f}%")
print(f"  Min/Max: {y.min():.2f}% / {y.max():.2f}%")

print(f"\n Features (X): {X.shape[1]} colunas")
print(f"  {list(X.columns)}")

# Dividir em treino 80% e teste 20%
# Treino = modelo aprende
# Teste = validamos se aprendeu mesmo
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,  # 20% pra teste
    random_state=42  # reprodutibilidade
)

print(f"\n Divisão treino/teste:")
print(f"  Treino: {len(X_train):,} amostras (80%)")
print(f"  Teste: {len(X_test):,} amostras (20%)")

# treinando o modelo...

print("\n\n PASSO 4: TREINANDO MODELO XGBOOST")

print("\n Configurando")
print("  Parâmetros:")
print("    - n_estimators=100 - 100 árvores")
print("    - max_depth=7  - profundidade das árvores")
print("    - random_state=42 reprodutibilidade")

modelo = XGBRegressor(
    n_estimators=100,      # Número de árvores
    max_depth=7,           # Profundidade de cada árvore
    learning_rate=0.1,     # Velocidade de aprendizado
    random_state=42,
    verbosity=0
)

print("\n Treinando modelo")
modelo.fit(X_train, y_train)
print("Modelo treinado!")

# Avaliação de modelo 

# Previsões no conjunto de treino
y_train_pred = modelo.predict(X_train)

# Previsões no conjunto de teste
y_test_pred = modelo.predict(X_test)

# Métricas de erro
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"\n TREINO:")
print(f"  MAE (Mean Absolute Error): {mae_train:.4f}%")
print(f"  RMSE (Root Mean Squared Error): {rmse_train:.4f}%")
print(f"  R^2 Score: {r2_train:.4f}")

print(f"\n TESTE:")
print(f"  MAE: {mae_test:.4f}%")
print(f"  RMSE: {rmse_test:.4f}%")
print(f"  R^2 Score: {r2_test:.4f}")

print(f"\n INTERPRETAÇÃO:")
print(f"  • R^2 = {r2_test:.4f} significa que o modelo explica {r2_test*100:.1f}% da variação")
print(f"  • MAE = {mae_test:.4f}% significa erro médio de {mae_test:.2f} pontos percentuais")
print(f"  • Se motor consumi 50%, modelo erra em ±{mae_test:.2f}%")

# Verifica overfitting
diferenca = abs(r2_train - r2_test)
if diferenca > 0.1:
    print(f"\n Possível overfitting (treino: {r2_train:.4f}, teste: {r2_test:.4f})")
else:
    print(f"\n Modelo generaliza bem (diferença R^2: {diferenca:.4f})")


# Obter importância de cada feature
importancias = modelo.feature_importances_
nomes_features = X.columns

# Ordenar por importância
indices_ordenados = np.argsort(importancias)[::-1]

print("\n Features mais importantes para prever consumo:")
for i, idx in enumerate(indices_ordenados[:10], 1):
    importancia_pct = importancias[idx] * 100
    print(f"  {i}. {nomes_features[idx]:25s} {importancia_pct:5.1f}%")

# Amostras de previsão 

print("\n\n amostras de previsões")

# Pega 5 exemplos aleatórios do conjunto de teste
indices_aleatorios = np.random.choice(len(X_test), size=5, replace=False)

print("\n temos:")
for idx in indices_aleatorios:
    consumo_real = y_test.iloc[idx]
    consumo_previsto = y_test_pred[idx]
    erro = abs(consumo_real - consumo_previsto)
    erro_pct = (erro / consumo_real * 100) if consumo_real > 0 else 0

    print(f"\n  Real: {consumo_real:.2f}% | Previsto: {consumo_previsto:.2f}% | Erro: {erro:.2f}% ({erro_pct:.1f}%)")

# salvando modelo de analise 

# Cria pasta de modelos
if not os.path.exists('modelos'):
    os.makedirs('modelos')

# Salva o modelo treinado
caminho_modelo = 'modelos/xgboost_consumo.pkl'
with open(caminho_modelo, 'wb') as f:
    pickle.dump(modelo, f)

print(f"Modelo salvo em: {caminho_modelo}")

# Salva também os nomes das features (pra usar depois)
caminho_features = 'modelos/feature_names.pkl'
with open(caminho_features, 'wb') as f:
    pickle.dump(nomes_features.tolist(), f)

print(f"Feature names salvos em: {caminho_features}")

# Resumão da parada 

print("Trampo executado ")
print(" ")


print(f"""
Resumão :
   • Modelo: XGBoost
   • Target: Consumo de combustível (%)
   • Features: 11 variáveis
   • Amostras treino: {len(X_train):,}
   • Amostras teste: {len(X_test):,}

Performance:
   • R^2 Score (teste): {r2_test:.4f} ({r2_test*100:.1f}%)
   • MAE: +- {mae_test:.2f}%
   • RMSE: {rmse_test:.2f}%

Interpretando a previsão:
   O modelo consegue prever consumo de combustível
   com {mae_test:.2f} pontos percentuais de erro em média.

   Exemplos:
   • Se prevê 50%, real pode ser 48-52%
   • Se prevê 70%, real pode ser 68-72%

todos os arquivos que foram gerado:
   • modelos/xgboost_consumo.pkl (modelo treinado)
   • modelos/feature_names.pkl (nomes das features)

Dashboard
   Vamos visualizar as previsões
""")