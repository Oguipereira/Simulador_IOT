import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os 

np.random.seed(42)

def gerar_timestamp_trabalho(data_base, empilhadeira_id):
    """ Gera timestamps realistas de trabalho.
    Lógica:
     Cada empilhadeira trabalha entre 8h e 17h
     Gera eventos a cada 1 minuto
      Alguns eventos são "parada" (GPS igual, velocidade = 0)
 
    Configurações bases:
        data_base: data inicial (datetime)
        empilhadeira_id: ID da empilhadeira (afeta variação de horário)
 
    Retornos:
        Lista de timestamps durante o turno
    """
    hora_inicio = 8 + (empilhadeira_id % 3)*0.5 
    hora_fim = 17 + (empilhadeira_id % 2)*0.5
    inicio = data_base.replace(hour=int(hora_inicio), minute=int((hora_inicio % 1) * 60), second=0)
    fim =   data_base.replace(hour=int(hora_fim), minute=int((hora_fim % 1) * 60), second=0)

    timestamps = pd.date_range(start=inicio, end=fim, freq='1min')
    return timestamps

def gerar_rotas_simuladas(num_eventos):
    """
    A ideia é simular uma rota realista com coordenadas que variam gradualmente.
 
    Lógica:
      - Começa em um ponto aleatório (dentro do galpão)
      - Se move gradualmente (velocidade realista de empilhadeira: 5 á 20 km/h)
      - Volta ao ponto de partida (rotina)
 
    Configurações bases:
        num_eventos: quantos pontos GPS gerar
        Armazém simulado: -23.55 a -23.56 lat, -46.63 a -46.64 long (tipo São Paulo)
        vou considerar que a empilhadeira poarte do ponto 0 = velocidade 0, e vai aumentando até 20 km/h, depois volta a 0.
 
    Retornos:
        Arrays de latitude, longitude, velocidade
    """
    lat_inicio = np.random.uniform(-23.556, -23.550)
    lng_inicio = np.random.uniform(-46.638, -46.632)  # Exemplo: São Paulo

    latitudes = [lat_inicio]
    longitudes = [lng_inicio]
    velocidades = [0] 

    for i in range(num_eventos - 1):
        delta_lat = np.random.uniform(0, 0.0001)
        delta_lng = np.random.uniform(0, 0.0001)

        lat_novo = latitudes[-1] + delta_lat
        lng_novo = longitudes[-1] + delta_lng

        latitudes.append(lat_novo)
        longitudes.append(lng_novo)

        """ Pegando uma velociade realista, 80% ela esta em movimento, considero de 5 á 15km/h, e 20% parada (0 km/h) """
        if  np.random.rand() < 0.8:
            velocidade = np.random.uniform(5, 15) 
        else:
            velocidade = 0 

        velocidades.append(velocidade)
    return np.array(latitudes), np.array(longitudes), np.array(velocidades)

def gerar_combustivel(velocidades, carga_transportada):
    """""Calcular consumo de combustivel baseado na velocidade e carga transportada.
     Lógica:
      Empilhadeira parada: consome combustível mínimo
      Velocidade alta + carga pesada: máximo consumo
      Fórmula: base + (velocidade * 0.1) + (carga * 0.01)
 
    Configurações bases:
        velocidades: array de velocidades 
        carga_transportada: peso da carga 
        o nosso tanque começa cheio 
        a unica regra é o tanque nunca zerar
    
 
    Retornos:
        Array de níveis de combustível (0-100)"""

    combustivel = 100.0
    historico_combustivel = [combustivel]

    for i in range(len(velocidades)):
        consumo_base = 0.05
        consumo_dinamico = (velocidades[i] * 0.1) + (carga_transportada[i] * 0.01)
        consumo_minuto = consumo_base + consumo_dinamico

        # pensando em um consumo real com aleatoriedade
        consumo_real = consumo_minuto * np.random.uniform(0.8, 1.2)
        combustivel -= consumo_real

        combustivel = max(0, combustivel)
        historico_combustivel.append(combustivel)
    return np.array(historico_combustivel[:-1])  

def gerar_temperatura (combustivel, velocidades):
    """"Simula temperatura do motor baseada em combustível e uso.
 
    Lógica:
      - Temperatura base: 60°C (parado, motor ligado)
      - Em movimento: aumenta com velocidade
      - Motor quente consome mais combustível (realista!)
 
    Configurações bases:
        combustivel: array de níveis de combustível
        velocidades: array de velocidades
 
    Retornos:
        Array de temperaturas em graus Celsius"""
    temperaturas = []

    for vel, comb in zip(velocidades, combustivel):
        temp_base = 60
        aumento_velocidade = vel * 0.5
        aumento_combustivel_baixo = max(0,(30 - comb) * 0.1) if comb < 30 else 0
        temperatura = temp_base + aumento_velocidade + aumento_combustivel_baixo
        temperaturas.append(temperatura)
    return np.array(temperaturas)

def gerar_dados_completos(num_empilhadeiras =300, num_dias=30):
    """Função principal que junta tudo.
 
    Cria um DataFrame com:
      300 empilhadeiras
      30 dias de simulação
      Eventos a cada 1 minuto durante turno (8-17h)
      menos fds
 
    Configurações bases:
        num_empilhadeiras: quantas empilhadeiras
        num_dias: quantos dias de simulação[]
        cada empilhadeira tem 1 a 4 operadores revezantes
        Operador que está trabalhando (alterna a cada 4 horas)

 
    Retornos:
        DataFrame com colunas:
          empilhadeira_id, operador_id, timestamp, latitude, longitude,
          velocidade, combustivel, peso_carga, temperatura_motor, tempo_parada"""

    dados_lista = []
    data_inicio = datetime(2026, 1, 1)
    print(f"Gerando dados para {num_empilhadeiras} empilhadeiras durante {num_dias} dias...")
    for empilhadeira_id in range(num_empilhadeiras):
        if empilhadeira_id % 50 == 0:
            print(f"Empilhadeira{empilhadeira_id}/{num_empilhadeiras}")
        num_operadores = np.random.randint(1, 4)
        operadores = [np.random.randint(1000, 9999) for _ in range(num_operadores)]

        for dia in range(num_dias):
            data_atual = data_inicio + timedelta(days=dia)

            if data_atual.weekday() >= 5:
                continue
            timestamps = gerar_timestamp_trabalho(data_atual, empilhadeira_id)
            num_eventos = len(timestamps)

            lats, lngs, velocidades = gerar_rotas_simuladas(num_eventos)
            carga_media = np.random.uniform(500, 2000)
            cargas = np.random.normal(carga_media, 300,num_eventos)
            cargas = np.clip(cargas, 0, 3000)
            combustivel = gerar_combustivel(velocidades, cargas)
            temperatura = gerar_temperatura(combustivel, velocidades)
 
            # Tempo parado: quanto tempo a empilhadeira fica sem se mover
            # (indicador de eficiência)
            tempo_parado = np.where(velocidades == 0, 1, 0)
            tempo_parado_acumulado = np.cumsum(tempo_parado)  # Minutos parado no dia
 
            operador_id_eventos = []
            for i in range(num_eventos):
                idx_operador = (i // 240) % len(operadores)
                operador_id_eventos.append(operadores[idx_operador])
 
            # Monta o DataFrame do dia
            df_dia = pd.DataFrame({
                'empilhadeira_id': empilhadeira_id,
                'operador_id': operador_id_eventos,
                'timestamp': timestamps,
                'latitude': lats,
                'longitude': lngs,
                'velocidade_kmh': velocidades,
                'combustivel_pct': combustivel,
                'peso_carga_kg': cargas,
                'temperatura_motor_celsius': temperatura,
                'tempo_parado_minutos': tempo_parado_acumulado,
            })
 
            dados_lista.append(df_dia)
 
    # Junta tudo em um único DataFrame
    df_completo = pd.concat(dados_lista, ignore_index=True)
 
    print(f"✓ Dados gerados: {len(df_completo):,} eventos")
    print(f"  Datas: {df_completo['timestamp'].min()} a {df_completo['timestamp'].max()}")
    print(f"  Empilhadeiras: {df_completo['empilhadeira_id'].nunique()}")
    print(f"  Operadores: {df_completo['operador_id'].nunique()}")
 
    return df_completo

 
if __name__ == "__main__":
 
    if not os.path.exists('dados'):
        os.makedirs('dados')
 
    df = gerar_dados_completos(num_empilhadeiras=300, num_dias=30)
    caminho_saida = 'dados/telemetria.parquet'
    df.to_parquet(caminho_saida, index=False)
    print(f"\n✓ Salvo em: {caminho_saida}")
 
    print("\n=== Primeiras linhas dos dados ===")   
    print(df.head())
    print("\n=== Info dos dados ===")
    print(df.info())
 
    print("\n=== Estatísticas ===")
    print(df.describe())
 
