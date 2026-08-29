"""
EXPLORACAO DE DADOS - SUPPLY CHAIN 2D

Analisa a telemetria de posicao (x, y) e o resumo de demandas gerados
pelo 2_gerador_telemetria_v2.py:
  - Onde as empilhadeiras mais passam (mapa de calor)
  - Distribuicao de tempo de rota (real vs estimativa ingenua)
  - Congestionamento por horario do dia
  - Relacao distancia x tempo x congestionamento
  - Volume de demandas por zona e tipo
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

PASTA_ATUAL = os.path.dirname(__file__)
PASTA_DADOS = os.path.join(PASTA_ATUAL, "dados")
PASTA_GRAFICOS = os.path.join(PASTA_ATUAL, "graficos_supply_chain")


def carregar_dados():
    telemetria = pd.read_parquet(os.path.join(PASTA_DADOS, "telemetria_rotas.parquet"))
    demandas = pd.read_csv(os.path.join(PASTA_DADOS, "demandas.csv"), parse_dates=["timestamp_inicio"])
    return telemetria, demandas


def resumo_geral(telemetria, demandas):
    print("=" * 80)
    print("RESUMO GERAL")
    print("=" * 80)
    print(f"Pontos de telemetria: {len(telemetria):,}")
    print(f"Demandas concluidas:  {len(demandas):,}")
    print(f"Empilhadeiras:        {telemetria['empilhadeira_id'].nunique()}")
    print(f"Operadores:           {telemetria['operador_id'].nunique()}")
    print(f"Dias simulados:       {demandas['dia'].nunique()}")
    print(f"\nTempo de rota (min):  media={demandas['tempo_real_min'].mean():.3f}  "
          f"p90={demandas['tempo_real_min'].quantile(0.9):.3f}  max={demandas['tempo_real_min'].max():.3f}")
    print(f"Congestionamento medio das rotas: {demandas['congestionamento_medio'].mean():.3f}")
    gap = (demandas['tempo_real_min'] - demandas['tempo_estimado_min'])
    print(f"Gap tempo real - estimativa ingenua: media={gap.mean():.3f} min "
          f"({100 * gap.mean() / demandas['tempo_estimado_min'].mean():.1f}% acima do estimado)")


def grafico_mapa_calor_posicoes(telemetria):
    """Onde as empilhadeiras passam mais tempo -- revela os corredores/zonas
    mais usados de verdade (nao so no papel do layout)."""
    fig, ax = plt.subplots(figsize=(12, 7))
    h = ax.hist2d(telemetria["x"], telemetria["y"], bins=[40, 24], cmap="inferno")
    fig.colorbar(h[3], ax=ax, label="Pontos de telemetria (densidade de trafego)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Mapa de Calor: Onde as Empilhadeiras Circulam Mais")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_GRAFICOS, "01_mapa_calor_posicoes.png"), dpi=100)
    plt.close()
    print("Salvo: 01_mapa_calor_posicoes.png")


def grafico_tempo_real_vs_estimado(demandas):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(demandas["tempo_estimado_min"], bins=40, alpha=0.6, label="Estimativa ingenua", color="steelblue")
    axes[0].hist(demandas["tempo_real_min"], bins=40, alpha=0.6, label="Tempo real (simulado)", color="orangered")
    axes[0].set_xlabel("Tempo de rota (min)")
    axes[0].set_ylabel("Frequencia")
    axes[0].set_title("Tempo Real vs Estimativa Ingenua")
    axes[0].legend()

    axes[1].scatter(demandas["tempo_estimado_min"], demandas["tempo_real_min"],
                     alpha=0.1, s=8, color="darkorange")
    lim = max(demandas["tempo_estimado_min"].max(), demandas["tempo_real_min"].max())
    axes[1].plot([0, lim], [0, lim], "k--", linewidth=1, label="Estimativa perfeita")
    axes[1].set_xlabel("Tempo estimado (min)")
    axes[1].set_ylabel("Tempo real (min)")
    axes[1].set_title("Estimativa vs Realidade (acima da linha = congestionamento atrasou a rota)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_GRAFICOS, "02_tempo_real_vs_estimado.png"), dpi=100)
    plt.close()
    print("Salvo: 02_tempo_real_vs_estimado.png")


def grafico_congestionamento_por_hora(demandas):
    congest_por_hora = demandas.groupby("hora_do_dia")["congestionamento_medio"].mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(congest_por_hora.index, congest_por_hora.values, marker="o", color="crimson", linewidth=2)
    ax.fill_between(congest_por_hora.index, congest_por_hora.values, alpha=0.2, color="crimson")
    ax.set_xlabel("Hora do dia")
    ax.set_ylabel("Congestionamento medio")
    ax.set_title("Congestionamento ao Longo do Turno (picos = janelas criticas de operacao)")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_GRAFICOS, "03_congestionamento_por_hora.png"), dpi=100)
    plt.close()
    print("Salvo: 03_congestionamento_por_hora.png")


def grafico_distancia_tempo_congestionamento(demandas):
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(demandas["distancia_percorrida_m"], demandas["tempo_real_min"],
                     c=demandas["congestionamento_medio"], cmap="viridis", alpha=0.4, s=10)
    fig.colorbar(sc, ax=ax, label="Congestionamento medio")
    ax.set_xlabel("Distancia percorrida (m)")
    ax.set_ylabel("Tempo real (min)")
    ax.set_title("Distancia x Tempo, colorido por Congestionamento")
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_GRAFICOS, "04_distancia_tempo_congestionamento.png"), dpi=100)
    plt.close()
    print("Salvo: 04_distancia_tempo_congestionamento.png")


def grafico_volume_por_zona_tipo(demandas):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    demandas["zona_origem"].value_counts().plot(kind="bar", ax=axes[0], color="steelblue", edgecolor="black")
    axes[0].set_title("Demandas por Zona de Origem")
    axes[0].set_ylabel("Numero de demandas")
    axes[0].tick_params(axis="x", rotation=0)

    demandas["tipo"].value_counts().plot(kind="bar", ax=axes[1], color="seagreen", edgecolor="black")
    axes[1].set_title("Demandas por Tipo de Movimentacao")
    axes[1].set_ylabel("Numero de demandas")
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_GRAFICOS, "05_volume_zona_tipo.png"), dpi=100)
    plt.close()
    print("Salvo: 05_volume_zona_tipo.png")


def top_zonas_criticas(demandas):
    print("\n" + "=" * 80)
    print("ZONAS COM MAIS TEMPO PERDIDO (real - estimado)")
    print("=" * 80)
    demandas = demandas.copy()
    demandas["atraso_min"] = demandas["tempo_real_min"] - demandas["tempo_estimado_min"]
    por_zona = demandas.groupby("zona_origem").agg(
        num_demandas=("demanda_id", "count"),
        atraso_medio_min=("atraso_min", "mean"),
        congestionamento_medio=("congestionamento_medio", "mean"),
    ).round(3).sort_values("atraso_medio_min", ascending=False)
    print(por_zona)


if __name__ == "__main__":
    os.makedirs(PASTA_GRAFICOS, exist_ok=True)

    telemetria, demandas = carregar_dados()

    resumo_geral(telemetria, demandas)

    print("\n" + "=" * 80)
    print("GERANDO GRAFICOS")
    print("=" * 80)
    grafico_mapa_calor_posicoes(telemetria)
    grafico_tempo_real_vs_estimado(demandas)
    grafico_congestionamento_por_hora(demandas)
    grafico_distancia_tempo_congestionamento(demandas)
    grafico_volume_por_zona_tipo(demandas)

    top_zonas_criticas(demandas)

    print(f"\nGraficos salvos em: {PASTA_GRAFICOS}")
