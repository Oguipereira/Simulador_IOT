"""
GERADOR DE TELEMETRIA V2 (com posicoes 2D reais)

Usa o layout 2D do 1_simulador_fabrica.py pra simular empilhadeiras
atendendo demandas de movimentacao de carga dentro da fabrica.

Diferenca pro gerador antigo (gerador_dados.py, projeto de telemetria de
combustivel): aqui a posicao (x, y) e a ROTA sao o dado principal, e o
"tempo de rota" e "congestionamento" nascem da geometria da fabrica, nao
de uma formula solta.

SIMPLIFICACAO ASSUMIDA (documentada de proposito):
  Simular N empilhadeiras se movendo ao mesmo tempo e "brigando" pelo
  mesmo corredor exigiria um simulador de eventos discretos completo
  (fila, deteccao de colisao, replanejamento). Pra manter o projeto
  tratavel, o congestionamento e modelado como um mapa de densidade de
  trafego (quantas rotas historicamente passam por cada celula da
  fabrica) multiplicado por um fator de horario de pico. Isso gera
  padroes espaciais e temporais coerentes (uteis pra clustering e ML),
  mas nao e uma simulacao fisica de colisoes entre empilhadeiras.

Saidas:
  dados/telemetria_rotas.parquet -> 1 linha por "tick" de posicao
  dados/demandas.csv             -> 1 linha por demanda concluida (resumo)
"""

import os
import sys
import numpy as np
import pandas as pd

np.random.seed(7)

PASTA_ATUAL = os.path.dirname(__file__)
if PASTA_ATUAL not in sys.path:
    sys.path.insert(0, PASTA_ATUAL)

import simulador_fabrica as sim_fabrica

TAMANHO_CELULA = 5.0  # metros
NUM_EMPILHADEIRAS = 15
NUM_DIAS = 20
DEMANDAS_POR_EMPILHADEIRA_DIA = 25
DURACAO_TURNO_MIN = 540  # 9h (08:00-17:00)
DT_MIN = 1 / 60          # 1 tick por segundo simulado
TEMPO_HANDLING_MIN = 0.5  # tempo fixo de pega/larga carga


def coordenada_para_celula(x, y):
    return (int(x // TAMANHO_CELULA), int(y // TAMANHO_CELULA))


def segmento_intersecta_retangulo(p1, p2, rect, margem=1.0):
    for t in np.linspace(0, 1, 20):
        x = p1[0] + t * (p2[0] - p1[0])
        y = p1[1] + t * (p2[1] - p1[1])
        if (rect["x_min"] - margem <= x <= rect["x_max"] + margem and
                rect["y_min"] - margem <= y <= rect["y_max"] + margem):
            return True
    return False


def calcular_waypoints(origem, destino, obstaculos, margem=2.0):
    """Rota reta, exceto quando cruza um obstaculo -- nesse caso desvia
    por um unico ponto (por cima ou por baixo do pilar, o que for mais perto)."""
    for obs in obstaculos:
        if segmento_intersecta_retangulo(origem, destino, obs, margem):
            y_topo = obs["y_max"] + margem
            y_base = obs["y_min"] - margem
            y_meio_rota = (origem[1] + destino[1]) / 2
            y_desvio = y_topo if abs(y_topo - y_meio_rota) < abs(y_base - y_meio_rota) else y_base
            x_desvio = (obs["x_min"] + obs["x_max"]) / 2
            return [origem, [x_desvio, y_desvio], destino]
    return [origem, destino]


def comprimento_rota(waypoints):
    return sum(sim_fabrica.distancia(waypoints[i], waypoints[i + 1]) for i in range(len(waypoints) - 1))


def construir_mapa_densidade_trafego(layout, num_amostras=3000):
    """Traca N demandas de exemplo e conta quantas vezes cada celula da
    fabrica e cruzada. Isso vira o mapa base de congestionamento."""
    nx = int(np.ceil(layout["largura"] / TAMANHO_CELULA))
    ny = int(np.ceil(layout["altura"] / TAMANHO_CELULA))
    contagem = np.zeros((nx, ny))

    demandas_amostra = sim_fabrica.gerar_demandas(layout, num_amostras, seed=99)
    for d in demandas_amostra:
        origem = [d["x_origem"], d["y_origem"]]
        destino = [d["x_destino"], d["y_destino"]]
        waypoints = calcular_waypoints(origem, destino, layout["obstaculos"])
        for i in range(len(waypoints) - 1):
            p1, p2 = waypoints[i], waypoints[i + 1]
            dist = sim_fabrica.distancia(p1, p2)
            passos = max(2, int(dist / (TAMANHO_CELULA / 2)))
            for t in np.linspace(0, 1, passos):
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                cx, cy = coordenada_para_celula(x, y)
                cx = min(max(cx, 0), nx - 1)
                cy = min(max(cy, 0), ny - 1)
                contagem[cx, cy] += 1

    densidade = contagem / contagem.max()
    return densidade


def multiplicador_horario_pico(minuto_do_turno):
    """Dois picos de movimento no turno (meio da manha e inicio da tarde),
    parecido com pico de expedicao/recebimento numa operacao real."""
    pico_manha = np.exp(-((minuto_do_turno - 90) ** 2) / (2 * 60 ** 2))
    pico_tarde = np.exp(-((minuto_do_turno - 330) ** 2) / (2 * 60 ** 2))
    return 1.0 + 1.2 * max(pico_manha, pico_tarde)


def nivel_congestionamento(densidade_mapa, x, y, minuto_do_turno):
    nx, ny = densidade_mapa.shape
    cx, cy = coordenada_para_celula(x, y)
    cx = min(max(cx, 0), nx - 1)
    cy = min(max(cy, 0), ny - 1)
    base = densidade_mapa[cx, cy]
    nivel = base * multiplicador_horario_pico(minuto_do_turno)
    return float(np.clip(nivel, 0, 1))


def velocidade_efetiva_kmh(peso_kg, nivel_congest):
    """Empilhadeira carregada anda mais devagar, e trafego reduz ainda mais."""
    velocidade_base = 10.0 - (peso_kg / 3000.0) * 3.0  # 7 a 10 km/h
    fator_congestionamento = 1.0 - 0.5 * nivel_congest
    return max(2.0, velocidade_base * fator_congestionamento)


def simular_trecho(waypoints, peso_kg, densidade_mapa, timestamp_inicio, minuto_turno_inicio,
                    empilhadeira_id, operador_id, demanda_id, zona_atual_fn):
    """Percorre os waypoints em ticks de tempo fixo, retornando as linhas de
    telemetria e o tempo total gasto (minutos)."""
    linhas = []
    tempo_acumulado_min = 0.0
    minuto_turno = minuto_turno_inicio
    congestionamentos = []

    for i in range(len(waypoints) - 1):
        p1, p2 = np.array(waypoints[i]), np.array(waypoints[i + 1])
        dist_trecho = float(np.linalg.norm(p2 - p1))
        if dist_trecho == 0:
            continue
        direcao = (p2 - p1) / dist_trecho

        pos_atual = p1.copy()
        dist_percorrida = 0.0
        while dist_percorrida < dist_trecho:
            nivel = nivel_congestionamento(densidade_mapa, pos_atual[0], pos_atual[1], minuto_turno)
            vel_kmh = velocidade_efetiva_kmh(peso_kg, nivel)
            vel_m_min = vel_kmh * 1000 / 60

            timestamp_atual = timestamp_inicio + pd.Timedelta(minutes=tempo_acumulado_min)
            linhas.append({
                "empilhadeira_id": empilhadeira_id,
                "operador_id": operador_id,
                "demanda_id": demanda_id,
                "timestamp": timestamp_atual,
                "x": float(pos_atual[0]),
                "y": float(pos_atual[1]),
                "velocidade_kmh": round(vel_kmh, 2),
                "peso_carga_kg": round(peso_kg, 1),
                "zona_atual": zona_atual_fn(pos_atual[0], pos_atual[1]),
                "nivel_congestionamento": round(nivel, 3),
                "congestionado": nivel > 0.6,
            })
            congestionamentos.append(nivel)

            passo_m = vel_m_min * DT_MIN
            dist_percorrida += passo_m
            tempo_acumulado_min += DT_MIN
            minuto_turno += DT_MIN
            pos_atual = p1 + direcao * min(dist_percorrida, dist_trecho)

    congest_medio = float(np.mean(congestionamentos)) if congestionamentos else 0.0
    return linhas, tempo_acumulado_min, congest_medio


def simular_dia(layout, densidade_mapa, dia_idx, data_base, empilhadeiras_info):
    """Simula um dia de operacao pra todas as empilhadeiras."""
    demandas_do_dia = sim_fabrica.gerar_demandas(
        layout,
        NUM_EMPILHADEIRAS * DEMANDAS_POR_EMPILHADEIRA_DIA,
        seed=1000 + dia_idx,
    )
    np.random.shuffle(demandas_do_dia)

    # cada empilhadeira recebe sua propria fatia do dia -- se fosse uma fila
    # global compartilhada, a primeira empilhadeira (viagens curtas, ~1min)
    # consumiria as ~375 demandas do dia sozinha antes das outras 14 sequer
    # comecarem a trabalhar.
    demandas_por_empilhadeira = np.array_split(demandas_do_dia, len(empilhadeiras_info))
    # dentro da fatia de cada empilhadeira, atende na ordem em que as
    # demandas "chegaram" (minuto_criacao) -- sem isso, todo mundo processa
    # sua fila inteira nos primeiros ~50 minutos do turno e o resto das 9h
    # fica sem nenhuma atividade simulada.
    demandas_por_empilhadeira = [
        sorted(fila, key=lambda d: d["minuto_criacao"]) for fila in demandas_por_empilhadeira
    ]

    zona_atual_fn = lambda x, y: sim_fabrica.zona_do_ponto(layout, x, y)

    todas_linhas_telemetria = []
    todas_demandas_resumo = []

    for emp, fila_demandas in zip(empilhadeiras_info, demandas_por_empilhadeira):
        pos_atual = list(emp["posicao_inicial"])
        tempo_gasto_min = 0.0
        num_operadores = emp["num_operadores"]
        operadores = emp["operadores"]

        for demanda in fila_demandas:
            if tempo_gasto_min >= DURACAO_TURNO_MIN:
                break

            # a empilhadeira so pode comecar a demanda depois que ela foi
            # criada -- se estiver livre antes disso, fica esperando
            tempo_gasto_min = max(tempo_gasto_min, float(demanda["minuto_criacao"]))
            if tempo_gasto_min >= DURACAO_TURNO_MIN:
                break

            idx_operador = int(tempo_gasto_min // 180) % num_operadores
            operador_id = operadores[idx_operador]

            timestamp_inicio_dia = data_base + pd.Timedelta(hours=8)
            timestamp_atual = timestamp_inicio_dia + pd.Timedelta(minutes=tempo_gasto_min)

            origem = [demanda["x_origem"], demanda["y_origem"]]
            destino = [demanda["x_destino"], demanda["y_destino"]]

            # 1) deslocamento vazio ate a origem da demanda
            waypoints_vazio = calcular_waypoints(pos_atual, origem, layout["obstaculos"])
            linhas_vazio, tempo_vazio, _ = simular_trecho(
                waypoints_vazio, 0.0, densidade_mapa, timestamp_atual, tempo_gasto_min,
                emp["empilhadeira_id"], operador_id, demanda["demanda_id"], zona_atual_fn,
            )
            tempo_gasto_min += tempo_vazio + TEMPO_HANDLING_MIN

            # 2) deslocamento carregado ate o destino (isso e o que vira o dado de ML)
            timestamp_carregado = timestamp_inicio_dia + pd.Timedelta(minutes=tempo_gasto_min)
            waypoints_carregado = calcular_waypoints(origem, destino, layout["obstaculos"])
            dist_planejada = comprimento_rota(waypoints_carregado)
            linhas_carregado, tempo_carregado, congest_medio = simular_trecho(
                waypoints_carregado, demanda["peso_kg"], densidade_mapa, timestamp_carregado,
                tempo_gasto_min, emp["empilhadeira_id"], operador_id, demanda["demanda_id"], zona_atual_fn,
            )
            tempo_gasto_min += tempo_carregado + TEMPO_HANDLING_MIN

            todas_linhas_telemetria.extend(linhas_vazio)
            todas_linhas_telemetria.extend(linhas_carregado)

            velocidade_media_planejamento = 8.5  # km/h, usada so pra estimativa "ingenua"
            tempo_estimado_min = (dist_planejada / 1000) / velocidade_media_planejamento * 60

            todas_demandas_resumo.append({
                "demanda_id": demanda["demanda_id"],
                "dia": dia_idx,
                "empilhadeira_id": emp["empilhadeira_id"],
                "operador_id": operador_id,
                "tipo": demanda["tipo"],
                "zona_origem": demanda["zona_origem"],
                "zona_destino": demanda["zona_destino"],
                "x_origem": demanda["x_origem"], "y_origem": demanda["y_origem"],
                "x_destino": demanda["x_destino"], "y_destino": demanda["y_destino"],
                "distancia_euclidiana_m": demanda["distancia_euclidiana_m"],
                "distancia_percorrida_m": round(dist_planejada, 1),
                "peso_kg": round(demanda["peso_kg"], 1),
                "prioridade": demanda["prioridade"],
                "sku": demanda["sku"],
                "timestamp_inicio": timestamp_carregado,
                "hora_do_dia": timestamp_carregado.hour,
                "dia_semana": timestamp_carregado.dayofweek,
                "tempo_real_min": round(tempo_carregado, 3),
                "tempo_estimado_min": round(tempo_estimado_min, 3),
                "congestionamento_medio": round(congest_medio, 3),
            })

            pos_atual = destino

    return todas_linhas_telemetria, todas_demandas_resumo


def montar_frota(layout, num_empilhadeiras):
    estacoes = layout["estacoes_carregamento"]
    frota = []
    for i in range(num_empilhadeiras):
        num_operadores = int(np.random.randint(1, 4))
        operadores = [int(np.random.randint(1000, 9999)) for _ in range(num_operadores)]
        estacao = estacoes[i % len(estacoes)]
        frota.append({
            "empilhadeira_id": i,
            "posicao_inicial": [estacao["x"], estacao["y"]],
            "num_operadores": num_operadores,
            "operadores": operadores,
        })
    return frota


if __name__ == "__main__":
    print("=" * 80)
    print("GERADOR DE TELEMETRIA V2 - SUPPLY CHAIN 2D")
    print("=" * 80)

    caminho_layout = os.path.join(PASTA_ATUAL, "dados", "layout_fabrica.json")
    if not os.path.exists(caminho_layout):
        print("Layout nao encontrado, gerando um novo...")
        layout = sim_fabrica.montar_layout()
        sim_fabrica.salvar_layout(layout, caminho_layout)
    else:
        import json
        with open(caminho_layout, "r", encoding="utf-8") as f:
            layout = json.load(f)

    print("\nConstruindo mapa de densidade de trafego (baseline de congestionamento)...")
    densidade_mapa = construir_mapa_densidade_trafego(layout)
    print(f"  Mapa: {densidade_mapa.shape[0]}x{densidade_mapa.shape[1]} celulas de {TAMANHO_CELULA}m")

    frota = montar_frota(layout, NUM_EMPILHADEIRAS)
    print(f"\nFrota: {NUM_EMPILHADEIRAS} empilhadeiras, {NUM_DIAS} dias uteis simulados")

    data_inicio = pd.Timestamp("2026-01-05")  # segunda-feira
    todas_telemetrias = []
    todas_demandas = []
    dia_idx = 0
    dias_simulados = 0
    while dias_simulados < NUM_DIAS:
        data_atual = data_inicio + pd.Timedelta(days=dia_idx)
        dia_idx += 1
        if data_atual.dayofweek >= 5:
            continue

        linhas, resumo = simular_dia(layout, densidade_mapa, dias_simulados, data_atual, frota)
        todas_telemetrias.extend(linhas)
        todas_demandas.extend(resumo)
        dias_simulados += 1

        if dias_simulados % 5 == 0:
            print(f"  Dia {dias_simulados}/{NUM_DIAS} simulado "
                  f"({len(todas_telemetrias):,} pontos de telemetria ate agora)")

    df_telemetria = pd.DataFrame(todas_telemetrias)
    df_demandas = pd.DataFrame(todas_demandas)

    pasta_dados = os.path.join(PASTA_ATUAL, "dados")
    os.makedirs(pasta_dados, exist_ok=True)
    df_telemetria.to_parquet(os.path.join(pasta_dados, "telemetria_rotas.parquet"), index=False)
    df_demandas.to_csv(os.path.join(pasta_dados, "demandas.csv"), index=False)

    print(f"\nTelemetria: {len(df_telemetria):,} pontos -> dados/telemetria_rotas.parquet")
    print(f"Demandas concluidas: {len(df_demandas):,} -> dados/demandas.csv")
    print(f"\nResumo tempo de rota (min): media={df_demandas['tempo_real_min'].mean():.2f} "
          f"min={df_demandas['tempo_real_min'].min():.2f} max={df_demandas['tempo_real_min'].max():.2f}")
    print(f"Congestionamento medio das rotas: {df_demandas['congestionamento_medio'].mean():.3f}")
