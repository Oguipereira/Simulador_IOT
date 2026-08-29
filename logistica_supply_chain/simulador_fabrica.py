"""
SIMULADOR DE FABRICA 2D

Modela o layout de um centro de distribuicao como um espaco 2D:
  - Zonas (Recebimento, Armazenagem, Expedicao)
  - Posicoes de rack (onde o estoque fica parado)
  - Obstaculos (pilares/estruturas que as empilhadeiras precisam contornar)
  - Estacoes de carregamento

Tambem gera as DEMANDAS: pedidos de movimentacao de carga de um ponto
(origem) para outro (destino) dentro da fabrica.

Este arquivo so descreve o "mundo". Quem move as empilhadeiras dentro
dele e o 2_gerador_telemetria_v2.py.
"""

import json
import os
import numpy as np

np.random.seed(42)

LARGURA_FABRICA = 100.0  # metros (eixo x)
ALTURA_FABRICA = 60.0    # metros (eixo y)


def distancia(p1, p2):
    """Distancia euclidiana entre dois pontos (x, y)."""
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def criar_zonas():
    """Divide a fabrica em 3 zonas lado a lado no eixo x.

    Recebimento (0-20) -> Armazenagem (20-80) -> Expedicao (80-100)
    Cada zona tem uma "doca": o ponto de acesso usado como origem/destino
    quando a movimentacao e de/para a zona como um todo (nao um rack especifico).
    """
    zonas = {
        "recebimento": {
            "tipo": "recebimento",
            "x_min": 0.0, "x_max": 20.0,
            "y_min": 0.0, "y_max": ALTURA_FABRICA,
            "doca": [10.0, ALTURA_FABRICA / 2],
        },
        "armazenagem": {
            "tipo": "armazenagem",
            "x_min": 20.0, "x_max": 80.0,
            "y_min": 0.0, "y_max": ALTURA_FABRICA,
            "doca": [50.0, ALTURA_FABRICA / 2],
        },
        "expedicao": {
            "tipo": "expedicao",
            "x_min": 80.0, "x_max": LARGURA_FABRICA,
            "y_min": 0.0, "y_max": ALTURA_FABRICA,
            "doca": [90.0, ALTURA_FABRICA / 2],
        },
    }
    return zonas


def criar_racks(zonas, num_corredores=6, posicoes_por_corredor=10):
    """Cria os pontos de rack dentro da zona de armazenagem.

    Layout tipo "espinha de peixe": corredores verticais igualmente
    espacados, com posicoes de picking ao longo de cada corredor.
    """
    zona_arm = zonas["armazenagem"]
    margem = 4.0
    x_inicio = zona_arm["x_min"] + margem
    x_fim = zona_arm["x_max"] - margem
    y_inicio = zona_arm["y_min"] + margem
    y_fim = zona_arm["y_max"] - margem

    corredores_x = np.linspace(x_inicio, x_fim, num_corredores)
    posicoes_y = np.linspace(y_inicio, y_fim, posicoes_por_corredor)

    racks = []
    rack_id = 0
    for cx in corredores_x:
        for py in posicoes_y:
            racks.append({
                "rack_id": rack_id,
                "corredor": float(cx),
                "x": float(cx),
                "y": float(py),
                "sku": f"SKU-{np.random.randint(1000, 9999)}",
                "capacidade_kg": int(np.random.choice([500, 1000, 1500, 2000])),
            })
            rack_id += 1
    return racks


def criar_obstaculos(zonas):
    """Pilares fixos dentro do armazem que as rotas devem contornar.

    Representados como retangulos (x_min, y_min, x_max, y_max).
    """
    zona_arm = zonas["armazenagem"]
    obstaculos = []
    pilares_x = np.linspace(zona_arm["x_min"] + 12, zona_arm["x_max"] - 12, 3)
    pilares_y = np.linspace(zona_arm["y_min"] + 15, zona_arm["y_max"] - 15, 2)
    for px in pilares_x:
        for py in pilares_y:
            obstaculos.append({
                "x_min": float(px - 1.5), "x_max": float(px + 1.5),
                "y_min": float(py - 1.5), "y_max": float(py + 1.5),
            })
    return obstaculos


def criar_estacoes_carregamento():
    """Pontos onde as empilhadeiras recarregam/abastecem, nos cantos da fabrica."""
    return [
        {"estacao_id": 0, "x": 2.0, "y": 2.0},
        {"estacao_id": 1, "x": 2.0, "y": ALTURA_FABRICA - 2.0},
        {"estacao_id": 2, "x": LARGURA_FABRICA - 2.0, "y": 2.0},
        {"estacao_id": 3, "x": LARGURA_FABRICA - 2.0, "y": ALTURA_FABRICA - 2.0},
    ]


def montar_layout():
    """Monta o layout completo da fabrica."""
    zonas = criar_zonas()
    racks = criar_racks(zonas)
    obstaculos = criar_obstaculos(zonas)
    estacoes = criar_estacoes_carregamento()

    layout = {
        "largura": LARGURA_FABRICA,
        "altura": ALTURA_FABRICA,
        "zonas": zonas,
        "racks": racks,
        "obstaculos": obstaculos,
        "estacoes_carregamento": estacoes,
    }
    return layout


def zona_do_ponto(layout, x, y):
    """Descobre em qual zona um ponto (x, y) esta."""
    for nome, zona in layout["zonas"].items():
        if zona["x_min"] <= x <= zona["x_max"] and zona["y_min"] <= y <= zona["y_max"]:
            return nome
    return "desconhecida"


def gerar_demandas(layout, num_demandas, seed=None):
    """Gera pedidos de movimentacao de carga (origem -> destino).

    Tipos de demanda:
      - recebimento_para_rack: descarregar caminhao e guardar no estoque
      - rack_para_expedicao: separar pedido e levar pra doca de saida
      - rack_para_rack: reorganizacao interna de estoque

    Cada demanda tem peso, sku, prioridade e horario de criacao dentro do turno.
    """
    if seed is not None:
        np.random.seed(seed)

    racks = layout["racks"]
    doca_receb = layout["zonas"]["recebimento"]["doca"]
    doca_exp = layout["zonas"]["expedicao"]["doca"]

    tipos = ["recebimento_para_rack", "rack_para_expedicao", "rack_para_rack"]
    pesos_tipos = [0.3, 0.5, 0.2]

    demandas = []
    for i in range(num_demandas):
        tipo = np.random.choice(tipos, p=pesos_tipos)
        rack_origem = racks[np.random.randint(0, len(racks))]
        rack_destino = racks[np.random.randint(0, len(racks))]

        if tipo == "recebimento_para_rack":
            origem = doca_receb
            destino = [rack_destino["x"], rack_destino["y"]]
        elif tipo == "rack_para_expedicao":
            origem = [rack_origem["x"], rack_origem["y"]]
            destino = doca_exp
        else:
            origem = [rack_origem["x"], rack_origem["y"]]
            destino = [rack_destino["x"], rack_destino["y"]]

        demandas.append({
            "demanda_id": i,
            "tipo": tipo,
            "x_origem": origem[0], "y_origem": origem[1],
            "x_destino": destino[0], "y_destino": destino[1],
            "zona_origem": zona_do_ponto(layout, *origem),
            "zona_destino": zona_do_ponto(layout, *destino),
            "peso_kg": float(np.clip(np.random.normal(800, 400), 20, 2500)),
            "sku": rack_origem["sku"],
            "prioridade": np.random.choice(["baixa", "normal", "alta"], p=[0.2, 0.6, 0.2]),
            "distancia_euclidiana_m": distancia(origem, destino),
            "minuto_criacao": int(np.random.uniform(0, 540)),  # dentro de um turno de 9h
        })

    return demandas


def salvar_layout(layout, caminho="dados/layout_fabrica.json"):
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(layout, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    print("=" * 80)
    print("SIMULADOR DE FABRICA 2D")
    print("=" * 80)

    layout = montar_layout()
    print(f"\nFabrica: {layout['largura']}m x {layout['altura']}m")
    for nome, zona in layout["zonas"].items():
        print(f"  Zona '{nome}': x[{zona['x_min']}, {zona['x_max']}] "
              f"y[{zona['y_min']}, {zona['y_max']}] doca={zona['doca']}")
    print(f"  Racks de estoque: {len(layout['racks'])}")
    print(f"  Obstaculos (pilares): {len(layout['obstaculos'])}")
    print(f"  Estacoes de carregamento: {len(layout['estacoes_carregamento'])}")

    caminho_saida = os.path.join(os.path.dirname(__file__), "dados", "layout_fabrica.json")
    salvar_layout(layout, caminho_saida)
    print(f"\nLayout salvo em: {caminho_saida}")

    print("\n--- Exemplo de demandas geradas ---")
    demandas_exemplo = gerar_demandas(layout, num_demandas=5, seed=1)
    for d in demandas_exemplo:
        print(f"  #{d['demanda_id']} [{d['tipo']}] {d['zona_origem']} -> {d['zona_destino']} "
              f"| {d['distancia_euclidiana_m']:.1f}m | {d['peso_kg']:.0f}kg | {d['prioridade']}")
