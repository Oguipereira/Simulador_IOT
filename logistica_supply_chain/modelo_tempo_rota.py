"""
MODELO: PREVISAO DE TEMPO DE ROTA

Treina um XGBoost pra prever quanto tempo uma empilhadeira carregada vai
levar pra ir da origem ate o destino, usando so features que realmente
tem relacao causal com o tempo de rota na simulacao:

  - distancia_percorrida_m  (rota real, ja considerando desvio de obstaculo)
  - peso_kg                 (carga mais pesada = mais devagar)
  - congestionamento_medio  (trafego no caminho)
  - hora_do_dia / dia_semana (pega os picos de congestionamento)
  - zona_origem / zona_destino / tipo (geometria/rota tipica)

De proposito NAO usamos empilhadeira_id nem operador_id como feature:
no simulador eles nao tem nenhum efeito causal sobre o tempo de rota, e
usa-los deixaria o modelo "aprender" ruido especifico de cada ID (ver
`3_exploracao_supply_chain.py` e a decisao tomada no projeto anterior de
telemetria de combustivel).

Comparamos o modelo treinado contra a "estimativa ingenua" que ja vem
nos dados (distancia / velocidade fixa) pra provar que o modelo agrega
valor de verdade.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

PASTA_ATUAL = os.path.dirname(__file__)
PASTA_DADOS = os.path.join(PASTA_ATUAL, "dados")
PASTA_MODELOS = os.path.join(PASTA_ATUAL, "modelos")

FEATURES_NUMERICAS = ["distancia_percorrida_m", "peso_kg", "congestionamento_medio", "hora_do_dia", "dia_semana"]
FEATURES_CATEGORICAS = ["zona_origem", "zona_destino", "tipo", "prioridade"]
ALVO = "tempo_real_min"


def carregar_dados():
    return pd.read_csv(os.path.join(PASTA_DADOS, "demandas.csv"))


def preparar_features(df):
    df = df.copy()
    for col in FEATURES_CATEGORICAS:
        df[col] = df[col].astype("category")
    X = df[FEATURES_NUMERICAS + FEATURES_CATEGORICAS]
    y = df[ALVO]
    return X, y


def split_temporal(df, dias_teste=4):
    """Validacao temporal: treina nos dias mais antigos, testa nos mais
    recentes -- mais realista que embaralhar aleatoriamente, porque no
    mundo real voce sempre preve o futuro a partir do passado."""
    ultimo_dia_treino = df["dia"].max() - dias_teste
    treino = df[df["dia"] <= ultimo_dia_treino]
    teste = df[df["dia"] > ultimo_dia_treino]
    return treino, teste


def treinar_modelo(X_treino, y_treino):
    modelo = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        enable_categorical=True,
        random_state=42,
    )
    modelo.fit(X_treino, y_treino)
    return modelo


def avaliar(nome, y_real, y_previsto):
    mae = mean_absolute_error(y_real, y_previsto)
    rmse = np.sqrt(mean_squared_error(y_real, y_previsto))
    r2 = r2_score(y_real, y_previsto)
    print(f"  {nome:22s} MAE={mae:.4f} min | RMSE={rmse:.4f} min | R2={r2:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


if __name__ == "__main__":
    print("=" * 80)
    print("MODELO DE PREVISAO DE TEMPO DE ROTA")
    print("=" * 80)

    df = carregar_dados()
    treino, teste = split_temporal(df, dias_teste=4)
    print(f"\nTreino: {len(treino):,} demandas (dias 0-{treino['dia'].max()})")
    print(f"Teste:  {len(teste):,} demandas (dias {teste['dia'].min()}-{teste['dia'].max()})")

    X_treino, y_treino = preparar_features(treino)
    X_teste, y_teste = preparar_features(teste)

    print("\nTreinando XGBoost...")
    modelo = treinar_modelo(X_treino, y_treino)

    y_previsto = modelo.predict(X_teste)

    print("\nDesempenho no conjunto de teste (dias nunca vistos no treino):")
    metricas_modelo = avaliar("XGBoost", y_teste, y_previsto)
    metricas_baseline = avaliar("Estimativa ingenua", y_teste, teste["tempo_estimado_min"])

    melhoria_mae = 100 * (metricas_baseline["mae"] - metricas_modelo["mae"]) / metricas_baseline["mae"]
    print(f"\nO modelo reduz o erro medio (MAE) em {melhoria_mae:.1f}% frente a estimativa ingenua.")

    print("\nImportancia das features:")
    importancias = pd.Series(modelo.feature_importances_, index=X_treino.columns).sort_values(ascending=False)
    for feat, imp in importancias.items():
        print(f"  {feat:25s} {imp:.3f}")

    os.makedirs(PASTA_MODELOS, exist_ok=True)
    with open(os.path.join(PASTA_MODELOS, "xgboost_tempo_rota.pkl"), "wb") as f:
        pickle.dump(modelo, f)
    with open(os.path.join(PASTA_MODELOS, "feature_names_tempo_rota.pkl"), "wb") as f:
        pickle.dump(list(X_treino.columns), f)

    print(f"\nModelo salvo em: {PASTA_MODELOS}/xgboost_tempo_rota.pkl")
