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
        consumo_real = consumo_real * np.random.uniform(0.8, 1.2)
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





