"""
OTIMIZACAO DE SEQUENCIA DE DEMANDAS (TSP simplificado + penalidade de congestionamento)

Problema: dada a fila de demandas de uma empilhadeira num turno, em que
ORDEM ela deve atende-las pra rodar o menor caminho possivel?

O trecho "carregado" de cada demanda (origem -> destino) e fixo, nao
importa a ordem. O que muda com a ordem e o deslocamento VAZIO entre o
destino de uma demanda e a origem da proxima -- isso e uma variante do
Traveling Salesman Problem (pickup-and-delivery sequencing).

Estrategia usada (heuristica gulosa, nao garante o otimo global mas e
rapida e da resultado bom o suficiente pra operacao real):
  1. Nearest neighbor: sempre ir pra origem mais perto da posicao atual.
  2. Penalidade de congestionamento: se o trajeto passa perto de uma
     zona critica (ver 5_modelo_gargalos.py), o "custo" dele e inflado,
     fazendo o algoritmo preferir rotas mais longas porem mais rapidas
     na pratica.

Comparamos contra a ordem "ingenua" (FIFO -- atender na ordem em que a
demanda foi criada, que e o que a operacao faz hoje sem otimizacao) pra
estimar a economia de distancia/tempo.
"""

import os
import numpy as np
import pandas as pd

PASTA_ATUAL = os.path.dirname(__file__)
PASTA_DADOS = os.path.join(PASTA_ATUAL, "dados")

VELOCIDADE_REFERENCIA_KMH = 8.5
RAIO_INFLUENCIA_GARGALO_M = 8.0
DIAS_UTEIS_POR_MES = 21


def carregar_dados():
    demandas = pd.read_csv(os.path.join(PASTA_DADOS, "demandas.csv"))
    caminho_gargalos = os.path.join(PASTA_DADOS, "gargalos_detectados.csv")
    gargalos = pd.read_csv(caminho_gargalos) if os.path.exists(caminho_gargalos) else pd.DataFrame(
        columns=["x_centro", "y_centro", "nivel_congestionamento_medio"]
    )
    return demandas, gargalos


def distancia(p1, p2):
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def penalidade_congestionamento(ponto, gargalos, raio=RAIO_INFLUENCIA_GARGALO_M):
    """Se o ponto esta perto do centro de alguma zona critica, infla o
    custo proporcional ao nivel de congestionamento medio daquela zona."""
    penalidade = 1.0
    for _, zona in gargalos.iterrows():
        d = distancia(ponto, (zona["x_centro"], zona["y_centro"]))
        if d < raio:
            penalidade = max(penalidade, 1.0 + zona["nivel_congestionamento_medio"])
    return penalidade


def distancia_efetiva(p1, p2, gargalos):
    """Distancia usada SO pra decidir a ordem -- ja incorpora a penalidade
    de congestionamento do ponto de chegada."""
    return distancia(p1, p2) * penalidade_congestionamento(p2, gargalos)


def distancia_real_total(demandas_ordenadas, pos_inicial):
    """Distancia de verdade (sem penalidade) percorrida seguindo essa ordem
    -- e o numero que interessa pra medir economia real."""
    pos_atual = pos_inicial
    total = 0.0
    for d in demandas_ordenadas:
        origem = (d["x_origem"], d["y_origem"])
        destino = (d["x_destino"], d["y_destino"])
        total += distancia(pos_atual, origem)  # deslocamento vazio
        total += distancia(origem, destino)     # deslocamento carregado (fixo)
        pos_atual = destino
    return total


def sequenciar_nearest_neighbor(demandas_batch, pos_inicial, gargalos):
    restantes = demandas_batch.copy()
    pos_atual = pos_inicial
    ordem = []

    while restantes:
        custos = [
            distancia_efetiva(pos_atual, (d["x_origem"], d["y_origem"]), gargalos)
            for d in restantes
        ]
        idx_melhor = int(np.argmin(custos))
        proxima = restantes.pop(idx_melhor)
        ordem.append(proxima)
        pos_atual = (proxima["x_destino"], proxima["y_destino"])

    return ordem


def otimizar_lote(demandas_df, gargalos):
    """demandas_df: as demandas de UMA empilhadeira em UM dia, ja ordenadas
    por minuto_criacao (== ordem FIFO que a operacao seguiria sem otimizar)."""
    ordem_fifo = demandas_df.sort_values("hora_do_dia").to_dict("records")
    if len(ordem_fifo) < 2:
        return None

    pos_inicial = (ordem_fifo[0]["x_origem"], ordem_fifo[0]["y_origem"])

    dist_fifo = distancia_real_total(ordem_fifo, pos_inicial)

    ordem_otimizada = sequenciar_nearest_neighbor(ordem_fifo, pos_inicial, gargalos)
    dist_otimizada = distancia_real_total(ordem_otimizada, pos_inicial)

    return {
        "num_demandas": len(ordem_fifo),
        "distancia_fifo_m": dist_fifo,
        "distancia_otimizada_m": dist_otimizada,
        "economia_m": dist_fifo - dist_otimizada,
        "economia_pct": 100 * (dist_fifo - dist_otimizada) / dist_fifo if dist_fifo > 0 else 0.0,
    }


def metros_para_minutos(metros, velocidade_kmh=VELOCIDADE_REFERENCIA_KMH):
    return (metros / 1000) / velocidade_kmh * 60


if __name__ == "__main__":
    print("=" * 80)
    print("OTIMIZACAO DE SEQUENCIA DE ROTAS")
    print("=" * 80)

    demandas, gargalos = carregar_dados()
    print(f"\nZonas criticas carregadas: {len(gargalos)}")

    resultados = []
    for (dia, empilhadeira_id), grupo in demandas.groupby(["dia", "empilhadeira_id"]):
        resultado = otimizar_lote(grupo, gargalos)
        if resultado:
            resultado["dia"] = dia
            resultado["empilhadeira_id"] = empilhadeira_id
            resultados.append(resultado)

    df_resultados = pd.DataFrame(resultados)

    print(f"\nLotes otimizados (1 empilhadeira x 1 dia cada): {len(df_resultados)}")
    print("\nResumo da economia de distancia por lote:")
    print(df_resultados[["distancia_fifo_m", "distancia_otimizada_m", "economia_pct"]].describe().round(2))

    economia_media_pct = df_resultados["economia_pct"].mean()
    economia_media_m_por_lote = df_resultados["economia_m"].mean()
    economia_min_por_lote = metros_para_minutos(economia_media_m_por_lote)

    num_empilhadeiras = demandas["empilhadeira_id"].nunique()
    economia_horas_mes = (economia_min_por_lote * num_empilhadeiras * DIAS_UTEIS_POR_MES) / 60

    print(f"\nEconomia media de distancia por lote: {economia_media_pct:.1f}% "
          f"({economia_media_m_por_lote:.1f} m -> {economia_min_por_lote:.2f} min)")
    print(f"Extrapolando pra frota inteira ({num_empilhadeiras} empilhadeiras, "
          f"{DIAS_UTEIS_POR_MES} dias uteis/mes):")
    print(f"  Economia estimada: {economia_horas_mes:.1f} horas de deslocamento vazio por mes")

    print("\n--- Exemplo: 1 lote antes/depois ---")
    exemplo = demandas[(demandas["dia"] == demandas["dia"].min()) &
                        (demandas["empilhadeira_id"] == demandas["empilhadeira_id"].min())]
    r = otimizar_lote(exemplo, gargalos)
    if r:
        print(f"  Demandas no lote: {r['num_demandas']}")
        print(f"  Distancia FIFO (sem otimizar):  {r['distancia_fifo_m']:.1f} m")
        print(f"  Distancia otimizada:            {r['distancia_otimizada_m']:.1f} m")
        print(f"  Economia:                       {r['economia_pct']:.1f}%")

    caminho_saida = os.path.join(PASTA_DADOS, "otimizacao_resultados.csv")
    df_resultados.to_csv(caminho_saida, index=False)
    print(f"\nSalvo: {caminho_saida}")
