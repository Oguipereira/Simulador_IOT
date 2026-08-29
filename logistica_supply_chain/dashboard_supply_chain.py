"""
DASHBOARD - SUPPLY CHAIN 2D

Junta tudo que os arquivos anteriores geraram:
  simulador_fabrica.py       -> layout_fabrica.json
  gerador_telemetria_v2.py   -> telemetria_rotas.parquet, demandas.csv
  modelo_tempo_rota.py       -> xgboost_tempo_rota.pkl
  modelo_gargalos.py         -> gargalos_detectados.csv, horarios_pico.csv
  otimizacao_rotas.py        -> otimizacao_resultados.csv

Rodar com: streamlit run dashboard_supply_chain.py
"""

import json
import os
import pickle
import sys

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

PASTA_ATUAL = os.path.dirname(__file__)
PASTA_DADOS = os.path.join(PASTA_ATUAL, "dados")
PASTA_MODELOS = os.path.join(PASTA_ATUAL, "modelos")
if PASTA_ATUAL not in sys.path:
    sys.path.insert(0, PASTA_ATUAL)

import otimizacao_rotas as otimizador

st.set_page_config(page_title="Supply Chain Inteligente", layout="wide", initial_sidebar_state="expanded")
sns.set_theme(style="whitegrid")


@st.cache_data
def carregar_layout():
    with open(os.path.join(PASTA_DADOS, "layout_fabrica.json"), "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def carregar_telemetria():
    return pd.read_parquet(os.path.join(PASTA_DADOS, "telemetria_rotas.parquet"))


@st.cache_data
def carregar_demandas():
    return pd.read_csv(os.path.join(PASTA_DADOS, "demandas.csv"))


@st.cache_data
def carregar_gargalos():
    return pd.read_csv(os.path.join(PASTA_DADOS, "gargalos_detectados.csv"))


@st.cache_data
def carregar_horarios_pico():
    return pd.read_csv(os.path.join(PASTA_DADOS, "horarios_pico.csv"))


@st.cache_data
def carregar_otimizacao():
    return pd.read_csv(os.path.join(PASTA_DADOS, "otimizacao_resultados.csv"))


@st.cache_resource
def carregar_modelo_tempo_rota():
    with open(os.path.join(PASTA_MODELOS, "xgboost_tempo_rota.pkl"), "rb") as f:
        modelo = pickle.load(f)
    with open(os.path.join(PASTA_MODELOS, "feature_names_tempo_rota.pkl"), "rb") as f:
        features = pickle.load(f)
    return modelo, features


def desenhar_layout_fabrica(ax, layout):
    cores_zona = {"recebimento": "#cde4ff", "armazenagem": "#fff3cd", "expedicao": "#d4f5dd"}
    for nome, zona in layout["zonas"].items():
        largura = zona["x_max"] - zona["x_min"]
        altura = zona["y_max"] - zona["y_min"]
        ax.add_patch(patches.Rectangle(
            (zona["x_min"], zona["y_min"]), largura, altura,
            facecolor=cores_zona.get(nome, "#eeeeee"), edgecolor="gray", linewidth=1.5, zorder=0,
        ))
        ax.text(zona["x_min"] + largura / 2, zona["y_max"] - 3, nome.upper(),
                ha="center", fontsize=9, color="dimgray", weight="bold")

    for obs in layout["obstaculos"]:
        ax.add_patch(patches.Rectangle(
            (obs["x_min"], obs["y_min"]), obs["x_max"] - obs["x_min"], obs["y_max"] - obs["y_min"],
            facecolor="dimgray", zorder=1,
        ))

    racks_x = [r["x"] for r in layout["racks"]]
    racks_y = [r["y"] for r in layout["racks"]]
    ax.scatter(racks_x, racks_y, s=10, color="saddlebrown", marker="s", zorder=2, label="Racks")

    for estacao in layout["estacoes_carregamento"]:
        ax.scatter(estacao["x"], estacao["y"], s=120, color="gold", marker="*",
                   edgecolor="black", zorder=3, label="_nolegend_")

    ax.set_xlim(0, layout["largura"])
    ax.set_ylim(0, layout["altura"])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.invert_yaxis()


layout = carregar_layout()
telemetria = carregar_telemetria()
demandas = carregar_demandas()
gargalos = carregar_gargalos()
horarios_pico = carregar_horarios_pico()
otimizacao = carregar_otimizacao()
modelo_tempo, features_tempo = carregar_modelo_tempo_rota()

st.title("Supply Chain Inteligente - Centro de Distribuicao")
st.markdown("Simulacao 2D + Machine Learning para otimizar a operacao de um centro de distribuicao.")
st.markdown("---")

pagina = st.sidebar.radio("Navegacao", [
    "Mapa da Fabrica",
    "Historico de Rotas",
    "Gargalos e Horarios de Pico",
    "Preditor de Tempo de Rota",
    "Otimizacao e Economia",
])

if pagina == "Mapa da Fabrica":
    st.header("Mapa da Fabrica")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Empilhadeiras", telemetria["empilhadeira_id"].nunique())
    col2.metric("Demandas atendidas", f"{len(demandas):,}")
    col3.metric("Tempo medio de rota", f"{demandas['tempo_real_min'].mean() * 60:.1f} s")
    col4.metric("Congestionamento medio", f"{demandas['congestionamento_medio'].mean():.2f}")

    st.markdown("---")
    col_layout, col_heat = st.columns(2)

    with col_layout:
        st.subheader("Layout (zonas, racks, obstaculos, estacoes)")
        fig, ax = plt.subplots(figsize=(9, 6))
        desenhar_layout_fabrica(ax, layout)
        ax.set_title("Zonas: azul=recebimento, amarelo=armazenagem, verde=expedicao")
        st.pyplot(fig)
        plt.close(fig)

    with col_heat:
        st.subheader("Onde as empilhadeiras circulam mais")
        fig, ax = plt.subplots(figsize=(9, 6))
        h = ax.hist2d(telemetria["x"], telemetria["y"], bins=[40, 24], cmap="inferno")
        fig.colorbar(h[3], ax=ax, label="Densidade de trafego")
        ax.invert_yaxis()
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        st.pyplot(fig)
        plt.close(fig)

elif pagina == "Historico de Rotas":
    st.header("Historico de Rotas por Empilhadeira")

    col1, col2 = st.columns(2)
    with col1:
        empilhadeira_sel = st.selectbox("Empilhadeira", sorted(telemetria["empilhadeira_id"].unique()))
    with col2:
        dias_disponiveis = sorted(demandas["dia"].unique())
        dia_sel = st.selectbox("Dia simulado", dias_disponiveis)

    telemetria["dia"] = (pd.to_datetime(telemetria["timestamp"]).dt.normalize()
                          - pd.to_datetime(telemetria["timestamp"]).dt.normalize().min()).dt.days

    filtro = telemetria[(telemetria["empilhadeira_id"] == empilhadeira_sel) & (telemetria["dia"] == dia_sel)]

    if len(filtro) == 0:
        st.warning("Sem telemetria pra essa combinacao (a empilhadeira pode nao ter recebido demandas nesse dia).")
    else:
        st.metric("Pontos de telemetria no dia", f"{len(filtro):,}")

        fig, ax = plt.subplots(figsize=(11, 7))
        desenhar_layout_fabrica(ax, layout)
        sc = ax.scatter(filtro["x"], filtro["y"], c=filtro["velocidade_kmh"], cmap="cool", s=6, zorder=4)
        fig.colorbar(sc, ax=ax, label="Velocidade (km/h)")
        ax.set_title(f"Caminho percorrido - Empilhadeira #{empilhadeira_sel} - Dia {dia_sel}")
        st.pyplot(fig)
        plt.close(fig)

elif pagina == "Gargalos e Horarios de Pico":
    st.header("Gargalos e Horarios de Pico")
    st.caption(
        "Gargalos = zonas onde o nivel de congestionamento simulado passou de 0.6 com frequencia, "
        "agrupadas por proximidade (DBSCAN). Horarios de pico = media de congestionamento por hora do turno."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Zonas criticas detectadas")
        st.dataframe(gargalos, use_container_width=True)

        fig, ax = plt.subplots(figsize=(9, 6))
        desenhar_layout_fabrica(ax, layout)
        ax.scatter(gargalos["x_centro"], gargalos["y_centro"],
                   s=gargalos["num_eventos"] / gargalos["num_eventos"].max() * 800 + 50,
                   c="red", alpha=0.5, zorder=5, label="Zona critica")
        ax.set_title("Centros das zonas de congestionamento (tamanho = frequencia)")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Congestionamento por hora do turno")
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(horarios_pico["hora"], horarios_pico["congestionamento_medio"], color="crimson", alpha=0.8)
        ax.set_xlabel("Hora do dia")
        ax.set_ylabel("Congestionamento medio")
        ax.set_ylim(0, 1)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("% de eventos congestionados por hora")
        st.dataframe(horarios_pico, use_container_width=True)

elif pagina == "Preditor de Tempo de Rota":
    st.header("Preditor de Tempo de Rota")
    st.markdown("Preve quanto tempo uma viagem carregada deve levar, com base no modelo XGBoost treinado.")

    col1, col2 = st.columns(2)
    zonas_validas = sorted(demandas["zona_origem"].unique())
    tipos_validos = sorted(demandas["tipo"].unique())
    prioridades_validas = sorted(demandas["prioridade"].unique())

    with col1:
        zona_origem = st.selectbox("Zona de origem", zonas_validas)
        zona_destino = st.selectbox("Zona de destino", zonas_validas, index=min(1, len(zonas_validas) - 1))
        tipo = st.selectbox("Tipo de movimentacao", tipos_validos)
        prioridade = st.selectbox("Prioridade", prioridades_validas)

    with col2:
        distancia_percorrida = st.slider("Distancia a percorrer (m)", 5.0, 100.0, 40.0, step=1.0)
        peso_kg = st.slider("Peso da carga (kg)", 0, 2500, 800, step=50)
        congestionamento = st.slider("Congestionamento esperado (0-1)", 0.0, 1.0, 0.5, step=0.05)
        hora_do_dia = st.slider("Hora do dia", 8, 17, 10)
        dia_semana = st.selectbox("Dia da semana", [0, 1, 2, 3, 4],
                                   format_func=lambda x: ["Segunda", "Terca", "Quarta", "Quinta", "Sexta"][x])

    if st.button("Prever tempo de rota", use_container_width=True):
        entrada = pd.DataFrame([{
            "distancia_percorrida_m": distancia_percorrida,
            "peso_kg": peso_kg,
            "congestionamento_medio": congestionamento,
            "hora_do_dia": hora_do_dia,
            "dia_semana": dia_semana,
            "zona_origem": zona_origem,
            "zona_destino": zona_destino,
            "tipo": tipo,
            "prioridade": prioridade,
        }])
        for col in ["zona_origem", "zona_destino", "tipo", "prioridade"]:
            entrada[col] = pd.Categorical(entrada[col], categories=demandas[col].astype("category").cat.categories)
        entrada = entrada[features_tempo]

        tempo_previsto_min = float(modelo_tempo.predict(entrada)[0])

        col1, col2 = st.columns(2)
        col1.metric("Tempo previsto", f"{tempo_previsto_min * 60:.1f} s")
        col2.metric("Tempo previsto", f"{tempo_previsto_min:.3f} min")

elif pagina == "Otimizacao e Economia":
    st.header("Otimizacao de Sequencia de Demandas")
    st.caption(
        "Compara a ordem em que as demandas seriam atendidas sem planejamento (FIFO, por ordem de chegada) "
        "contra uma sequencia otimizada (nearest-neighbor com penalidade de congestionamento)."
    )

    col1, col2, col3 = st.columns(3)
    economia_media_pct = otimizacao["economia_pct"].mean()
    num_empilhadeiras = demandas["empilhadeira_id"].nunique()
    economia_m_media = otimizacao["economia_m"].mean()
    economia_min_media = otimizador.metros_para_minutos(economia_m_media)
    economia_horas_mes = (economia_min_media * num_empilhadeiras * otimizador.DIAS_UTEIS_POR_MES) / 60

    col1.metric("Economia media de distancia", f"{economia_media_pct:.1f}%")
    col2.metric("Economia media por turno", f"{economia_min_media:.2f} min/empilhadeira")
    col3.metric("Economia estimada da frota", f"{economia_horas_mes:.1f} h/mes")

    st.markdown("---")
    st.subheader("Distribuicao da economia entre os lotes simulados (1 empilhadeira x 1 dia)")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(otimizacao["economia_pct"], bins=30, color="seagreen", edgecolor="black", alpha=0.8)
    ax.axvline(economia_media_pct, color="red", linestyle="--", label=f"Media: {economia_media_pct:.1f}%")
    ax.set_xlabel("Economia de distancia (%)")
    ax.set_ylabel("Numero de lotes")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.subheader("Exemplo: antes x depois pra um lote especifico")
    col1, col2 = st.columns(2)
    with col1:
        dia_escolhido = st.selectbox("Dia", sorted(demandas["dia"].unique()))
    with col2:
        empilhadeira_escolhida = st.selectbox("Empilhadeira", sorted(demandas["empilhadeira_id"].unique()))

    lote = demandas[(demandas["dia"] == dia_escolhido) & (demandas["empilhadeira_id"] == empilhadeira_escolhida)]
    resultado = otimizador.otimizar_lote(lote, gargalos)

    if resultado is None:
        st.warning("Esse lote nao tem demandas suficientes pra comparar.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Distancia sem otimizar (FIFO)", f"{resultado['distancia_fifo_m']:.0f} m")
        col2.metric("Distancia otimizada", f"{resultado['distancia_otimizada_m']:.0f} m")
        col3.metric("Economia", f"{resultado['economia_pct']:.1f}%")


st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 12px;'>"
    "Supply Chain Inteligente v1.0 | Simulacao 2D + XGBoost + Clustering + Otimizacao"
    "</div>",
    unsafe_allow_html=True,
)
