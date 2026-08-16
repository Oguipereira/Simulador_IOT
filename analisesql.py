

import duckdb
import pandas as pd

print("=" * 80)
print("ANÁLISE SQL COM DUCKDB")
print("=" * 80)


print("\nCONECTANDO AO DUCKDB...")

# Cria conexão (em memória ou arquivo)
conn = duckdb.connect('dados/telemetria.duckdb')
print("Banco conectado")

# Carrega os dados do Parquet pra uma tabela SQL
print("   Loading dados/telemetria.parquet...")
conn.execute("""
    CREATE TABLE IF NOT EXISTS telemetria AS
    SELECT * FROM read_parquet('dados/telemetria.parquet')
""")
print("Tabela 'telemetria' criada")

# Valida a tabela
result = conn.execute("SELECT COUNT(*) as total FROM telemetria").fetchall()
total_linhas = result[0][0]
print(f"   ✓ Total de linhas: {total_linhas:,}")



print("\n" + "=" * 80)
print("QUERIES SQL - PERGUNTAS DE NEGÓCIO")
print("=" * 80)

# QUERY 1: Consumo total por empilhadeira
print("\nQUERY 1: Qual empilhadeira consumiu mais combustível?")
print("-" * 80)

query1 = """
SELECT
    empilhadeira_id,
    COUNT(*) as num_eventos,
    ROUND(AVG(velocidade_kmh), 2) as vel_media_kmh,
    ROUND(AVG(peso_carga_kg), 2) as carga_media_kg,
    ROUND(100 - MIN(combustivel_pct), 2) as consumo_total_pct,
    ROUND(AVG(temperatura_motor_celsius), 2) as temp_media_celsius
FROM telemetria
GROUP BY empilhadeira_id
ORDER BY consumo_total_pct DESC
LIMIT 10
"""

df1 = conn.execute(query1).df()
print(df1.to_string(index=False))

# QUERY 2: Eficiência por operador
print("\nQUERY 2: Qual operador é mais eficiente?")
print("-" * 80)

query2 = """
SELECT
    operador_id,
    COUNT(*) as num_eventos,
    ROUND(AVG(velocidade_kmh), 2) as vel_media_kmh,
    ROUND(AVG(peso_carga_kg), 2) as carga_media_kg,
    ROUND(100 - MIN(combustivel_pct), 2) as consumo_total_pct,
    ROUND(AVG(temperatura_motor_celsius), 2) as temp_media_celsius
FROM telemetria
GROUP BY operador_id
ORDER BY consumo_total_pct ASC
LIMIT 10
"""

df2 = conn.execute(query2).df()
print(df2.to_string(index=False))

# QUERY 3: Horas de operação por dia
print("\n4️⃣ QUERY 3: Quantas horas cada dia teve operação?")
print("-" * 80)

query3 = """
SELECT
    DATE(timestamp) as data,
    COUNT(DISTINCT empilhadeira_id) as num_empilhadeiras_ativas,
    COUNT(*) as total_eventos,
    ROUND(COUNT(*) / 1440.0, 2) as horas_operacao_media
FROM telemetria
GROUP BY DATE(timestamp)
ORDER BY data ASC
"""

df3 = conn.execute(query3).df()
print(df3.to_string(index=False))

# QUERY 4: Empilhadeiras com possível problema de manutenção
print("\nQUERY 4: Quais empilhadeiras têm sinais de manutenção necessária?")
print("-" * 80)

query4 = """
SELECT
    empilhadeira_id,
    ROUND(MAX(temperatura_motor_celsius), 2) as temp_maxima,
    COUNT(*) FILTER (WHERE temperatura_motor_celsius > 95) as eventos_quente,
    ROUND(100.0 * COUNT(*) FILTER (WHERE temperatura_motor_celsius > 95)
          / COUNT(*), 2) as pct_eventos_quente,
    ROUND(100 - MIN(combustivel_pct), 2) as consumo_total_pct
FROM telemetria
GROUP BY empilhadeira_id
HAVING COUNT(*) FILTER (WHERE temperatura_motor_celsius > 95) > 100
ORDER BY pct_eventos_quente DESC
LIMIT 10
"""

df4 = conn.execute(query4).df()
if len(df4) > 0:
    print(df4.to_string(index=False))
else:
    print("   Nenhuma empilhadeira com sinais críticos de manutenção")

# QUERY 5: Padrão de velocidade vs combustível
print("\nQUERY 5: Como velocidade impacta consumo?")
print("-" * 80)

query5 = """
SELECT
    CASE
        WHEN velocidade_kmh = 0 THEN 'Parado'
        WHEN velocidade_kmh BETWEEN 0.1 AND 5 THEN '0-5 km/h'
        WHEN velocidade_kmh BETWEEN 5.1 AND 10 THEN '5-10 km/h'
        WHEN velocidade_kmh BETWEEN 10.1 AND 15 THEN '10-15 km/h'
        ELSE '15+ km/h'
    END as faixa_velocidade,
    COUNT(*) as num_eventos,
    ROUND(AVG(combustivel_pct), 2) as combustivel_medio,
    ROUND(AVG(temperatura_motor_celsius), 2) as temp_media
FROM telemetria
GROUP BY faixa_velocidade
ORDER BY num_eventos DESC
"""

df5 = conn.execute(query5).df()
print(df5.to_string(index=False))

# QUERY 6: Eficiência de carga
print("\nQUERY 6: Como carga impacta consumo?")
print("-" * 80)

query6 = """
SELECT
    CASE
        WHEN peso_carga_kg < 500 THEN 'Leve (<500 kg)'
        WHEN peso_carga_kg BETWEEN 500 AND 1000 THEN 'Média (500-1000 kg)'
        WHEN peso_carga_kg BETWEEN 1000 AND 1500 THEN 'Pesada (1000-1500 kg)'
        WHEN peso_carga_kg BETWEEN 1500 AND 2000 THEN 'Muito Pesada (1500-2000 kg)'
        ELSE 'Máxima (2000+ kg)'
    END as faixa_carga,
    COUNT(*) as num_eventos,
    ROUND(AVG(combustivel_pct), 2) as combustivel_medio,
    ROUND(AVG(temperatura_motor_celsius), 2) as temp_media,
    ROUND(100 - MIN(combustivel_pct), 2) as consumo_medio_pct
FROM telemetria
GROUP BY faixa_carga
ORDER BY num_eventos DESC
"""

df6 = conn.execute(query6).df()
print(df6.to_string(index=False))

# QUERY 7: Top 10 operador-empilhadeira combinações mais eficientes
print("\nQUERY 7: Qual dupla (operador + empilhadeira) é mais eficiente?")
print("-" * 80)

query7 = """
SELECT
    operador_id,
    empilhadeira_id,
    COUNT(*) as num_eventos,
    ROUND(AVG(velocidade_kmh), 2) as vel_media,
    ROUND(AVG(peso_carga_kg), 2) as carga_media,
    ROUND(100 - MIN(combustivel_pct), 2) as consumo_pct
FROM telemetria
GROUP BY operador_id, empilhadeira_id
HAVING COUNT(*) > 500  -- Apenas duplas com experiência suficiente
ORDER BY consumo_pct ASC
LIMIT 10
"""

df7 = conn.execute(query7).df()
print(df7.to_string(index=False))

# QUERY 8: Análise de parada (tempo ocioso)
print("\nQUERY 8: Qual empilhadeira fica mais tempo parada?")
print("-" * 80)

query8 = """
SELECT
    empilhadeira_id,
    COUNT(*) as total_eventos,
    COUNT(*) FILTER (WHERE velocidade_kmh = 0) as eventos_parado,
    ROUND(100.0 * COUNT(*) FILTER (WHERE velocidade_kmh = 0) / COUNT(*), 2) as pct_parado,
    ROUND(AVG(CASE WHEN velocidade_kmh > 0 THEN velocidade_kmh END), 2) as vel_media_movimento
FROM telemetria
GROUP BY empilhadeira_id
ORDER BY pct_parado DESC
LIMIT 10
"""

df8 = conn.execute(query8).df()
print(df8.to_string(index=False))

# QUERY 9: Combustível crítico - quando o tanque fica muito vazio
print("\nQUERY 9: Quantas vezes tanque ficou crítico (<10%)?")
print("-" * 80)

query9 = """
SELECT
    empilhadeira_id,
    COUNT(*) FILTER (WHERE combustivel_pct < 10) as eventos_criticos,
    ROUND(MIN(combustivel_pct), 2) as combustivel_minimo,
    DATE(MAX(timestamp)) as ultima_ocorrencia
FROM telemetria
GROUP BY empilhadeira_id
HAVING COUNT(*) FILTER (WHERE combustivel_pct < 10) > 0
ORDER BY eventos_criticos DESC
LIMIT 10
"""

df9 = conn.execute(query9).df()
print(df9.to_string(index=False))

# QUERY 10: Resumo estatístico por hora do dia
print("\nQUERY 10: Padrão de operação por hora do dia")
print("-" * 80)

query10 = """
SELECT
    HOUR(timestamp) as hora,
    COUNT(*) as num_eventos,
    COUNT(DISTINCT empilhadeira_id) as num_empilhadeiras,
    ROUND(AVG(velocidade_kmh), 2) as vel_media,
    ROUND(AVG(combustivel_pct), 2) as combustivel_medio,
    ROUND(AVG(temperatura_motor_celsius), 2) as temp_media
FROM telemetria
GROUP BY HOUR(timestamp)
ORDER BY hora ASC
"""

df10 = conn.execute(query10).df()
print(df10.to_string(index=False))

# EXPORTAR RESULTADOS

print("\n" + "=" * 80)
print("EXPORTANDO RESULTADOS")
print("=" * 80)

# Salva cada resultado como CSV pra referência futura
resultados = {
    'consumo_empilhadeiras': df1,
    'eficiencia_operadores': df2,
    'operacao_por_dia': df3,
    'manutencao_necessaria': df4,
    'velocidade_impacto': df5,
    'carga_impacto': df6,
    'dupla_eficiente': df7,
    'tempo_parado': df8,
    'combustivel_critico': df9,
    'padrao_horario': df10
}

for nome, df in resultados.items():
    arquivo = f'resultados/{nome}.csv'
    df.to_csv(arquivo, index=False)
    print(f"   ✓ {arquivo}")

# INSIGHTS FINAIS

print("\n" + "=" * 80)
print("INSIGHTS DAS QUERIES SQL")
print("=" * 80)

kpi_consumo = conn.execute("""
    SELECT
        ROUND(100 - MIN(combustivel_pct), 2) as consumo_total,
        ROUND(AVG(combustivel_pct), 2) as combustivel_medio,
        ROUND(STDDEV(combustivel_pct), 2) as desvio_padrao
    FROM telemetria
""").fetchall()

kpi_temp = conn.execute("""
    SELECT
        ROUND(AVG(temperatura_motor_celsius), 2) as temp_media,
        ROUND(MAX(temperatura_motor_celsius), 2) as temp_maxima,
        COUNT(*) FILTER (WHERE temperatura_motor_celsius > 95) as eventos_alerta
    FROM telemetria
""").fetchall()

print(f"""
CONSUMO DE COMBUSTÍVEL:
   • Consumo total da frota: {kpi_consumo[0][0]:.2f}%
   • Nível médio ao longo do dia: {kpi_consumo[0][1]:.2f}%
   • Variação (desvio padrão): {kpi_consumo[0][2]:.2f}%

TEMPERATURA DO MOTOR:
   • Média: {kpi_temp[0][0]:.2f}°C
   • Máxima registrada: {kpi_temp[0][1]:.2f}°C
   • Alertas (T > 95°C): {kpi_temp[0][2]:,} eventos

OPORTUNIDADES:
   • Otimizar rotas das empilhadeiras menos eficientes
   • Treinar operadores com consumo elevado
   • Programar manutenção preventiva baseada em temperatura
   • Identificar padrões de ociosidade

DADOS PRONTOS
""")

# Fecha conexão
conn.close()
print("\nAnálise SQL concluída!")
print("Resultados salvos em: resultados/")
print("=" * 80)
