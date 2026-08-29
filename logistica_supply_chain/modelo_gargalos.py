"""
DETECCAO DE GARGALOS (zonas de congestionamento + horarios de pico)

Duas perguntas de negocio:
  1. ONDE a fabrica congestiona? -> clustering (DBSCAN) nos pontos de
     telemetria marcados como "congestionado", agrupando-os em zonas
     criticas (em vez de so olhar celula por celula).
  2. QUANDO isso acontece? -> agregacao por hora do turno.

Saidas:
  dados/gargalos_detectados.csv -> um cluster por linha (centro, tamanho, nivel)
  dados/horarios_pico.csv       -> nivel de congestionamento medio por hora
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

PASTA_ATUAL = os.path.dirname(__file__)
PASTA_DADOS = os.path.join(PASTA_ATUAL, "dados")

LIMIAR_CONGESTIONAMENTO = 0.6
DBSCAN_EPS_M = 3.0
DBSCAN_MIN_AMOSTRAS = 80
MAX_PONTOS_CLUSTERING = 40_000  # amostra pra manter o DBSCAN rapido


def carregar_telemetria():
    return pd.read_parquet(os.path.join(PASTA_DADOS, "telemetria_rotas.parquet"))


def detectar_zonas_criticas(telemetria):
    congestionados = telemetria[telemetria["congestionado"]]
    print(f"Pontos congestionados (nivel > {LIMIAR_CONGESTIONAMENTO}): "
          f"{len(congestionados):,} de {len(telemetria):,} "
          f"({100 * len(congestionados) / len(telemetria):.1f}%)")

    amostra = congestionados
    if len(amostra) > MAX_PONTOS_CLUSTERING:
        amostra = amostra.sample(MAX_PONTOS_CLUSTERING, random_state=42)
        print(f"Amostrando {MAX_PONTOS_CLUSTERING:,} pontos pra rodar o DBSCAN mais rapido")

    coordenadas = amostra[["x", "y"]].to_numpy()
    clustering = DBSCAN(eps=DBSCAN_EPS_M, min_samples=DBSCAN_MIN_AMOSTRAS).fit(coordenadas)
    amostra = amostra.copy()
    amostra["cluster"] = clustering.labels_

    ruido = (amostra["cluster"] == -1).sum()
    num_clusters = amostra["cluster"].nunique() - (1 if ruido > 0 else 0)
    print(f"DBSCAN encontrou {num_clusters} zonas criticas "
          f"({ruido:,} pontos isolados descartados como ruido)")

    zonas_criticas = amostra[amostra["cluster"] != -1].groupby("cluster").agg(
        x_centro=("x", "mean"),
        y_centro=("y", "mean"),
        num_eventos=("cluster", "count"),
        nivel_congestionamento_medio=("nivel_congestionamento", "mean"),
        zona_predominante=("zona_atual", lambda s: s.mode().iat[0]),
    ).round(3).sort_values("num_eventos", ascending=False)

    return zonas_criticas.reset_index(drop=True)


def horarios_de_pico(telemetria):
    telemetria = telemetria.copy()
    telemetria["hora"] = pd.to_datetime(telemetria["timestamp"]).dt.hour
    por_hora = telemetria.groupby("hora").agg(
        congestionamento_medio=("nivel_congestionamento", "mean"),
        pct_pontos_congestionados=("congestionado", "mean"),
        num_eventos=("congestionado", "count"),
    ).round(3)
    por_hora["pct_pontos_congestionados"] = (por_hora["pct_pontos_congestionados"] * 100).round(1)
    return por_hora.sort_values("congestionamento_medio", ascending=False)


if __name__ == "__main__":
    print("=" * 80)
    print("DETECCAO DE GARGALOS")
    print("=" * 80)

    telemetria = carregar_telemetria()

    print("\n--- ONDE: zonas criticas (clustering espacial) ---")
    zonas_criticas = detectar_zonas_criticas(telemetria)
    print(zonas_criticas.head(10).to_string(index=False))

    print("\n--- QUANDO: congestionamento por hora do turno ---")
    picos = horarios_de_pico(telemetria)
    print(picos.to_string())

    top_horarios = picos.head(3)
    print(f"\nHorarios mais criticos: {list(top_horarios.index)}h "
          f"(congestionamento medio ate {top_horarios['congestionamento_medio'].max():.2f})")

    zonas_criticas.to_csv(os.path.join(PASTA_DADOS, "gargalos_detectados.csv"), index=False)
    picos.to_csv(os.path.join(PASTA_DADOS, "horarios_pico.csv"))
    print(f"\nSalvo: dados/gargalos_detectados.csv ({len(zonas_criticas)} zonas)")
    print("Salvo: dados/horarios_pico.csv")
