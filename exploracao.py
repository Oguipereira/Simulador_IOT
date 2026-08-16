"""
A ideia é ter mos a EXPLORAÇÃO DE DADOS (EDA - Exploratory Data Analysis)

  Você tem dados brutos e faz perguntas:
  - Quais colunas tenho?
  - Qual é o tamanho dos dados?
  - Qual é a distribuição?
  - Há outliers?
  - Qual é o padrão?

OBJETIVO: Entender os dados antes de treinar um modelo ML.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Deixa visual bonito
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)

print("\n" + "="*80)
print("EXPLORAÇÃO DE DADOS - STEP BY STEP")
print("="*80 + "\n")


print("CARREGANDO OS DADOS")
print("-" * 80)

df = pd.read_parquet('dados/telemetria.parquet')

print(f"Dados carregados!")
print(f"  Tamanho: {df.shape[0]:,} linhas")
print(f"  Colunas: {df.shape[1]}")
print(f"  Memória: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB\n")


print("O QUE TEM NOS DADOS?")
print("-" * 80)

print("\n Primeiras 5 linhas:")
print(df.head())

print("\n Nomes e tipos das colunas:")
print(df.dtypes)

print("\n Resumo das colunas:")
print(df.describe())



print("\n\n" + "="*80)
print("VALIDAÇÃO DOS DADOS")
print("="*80 + "\n")

print(" Valores faltando (NaN)?")
print(df.isnull().sum())

print("\n Valores únicos por coluna:")
for coluna in df.columns:
    unicos = df[coluna].nunique()
    print(f"  {coluna}: {unicos} valores únicos")


print("\n\n" + "="*80)
print("PERGUNTAS SIMPLES AOS DADOS")
print("="*80 + "\n")

num_empilhadeiras = df['empilhadeira_id'].nunique()
print(f"Quantas empilhadeiras? {num_empilhadeiras}")

num_operadores = df['operador_id'].nunique()
print(f" Quantos operadores? {num_operadores}")

data_inicio = df['timestamp'].min()
data_fim = df['timestamp'].max()
dias = (data_fim - data_inicio).days
print(f"Período: {data_inicio.date()} até {data_fim.date()} ({dias + 1} dias)")

vel_media = df['velocidade_kmh'].mean()
vel_max = df['velocidade_kmh'].max()
vel_min = df['velocidade_kmh'].min()
print(f" Velocidade: mín={vel_min:.1f}, média={vel_media:.1f}, máx={vel_max:.1f} km/h")

parado = (df['velocidade_kmh'] == 0).sum()
pct_parado = 100 * parado / len(df)
print(f"  Tempo parado: {parado:,} eventos ({pct_parado:.1f}%)")

comb_media = df['combustivel_pct'].mean()
comb_min = df['combustivel_pct'].min()
comb_max = df['combustivel_pct'].max()
print(f"  Combustível: mín={comb_min:.1f}%, média={comb_media:.1f}%, máx={comb_max:.1f}%")

carga_media = df['peso_carga_kg'].mean()
carga_min = df['peso_carga_kg'].min()
carga_max = df['peso_carga_kg'].max()
print(f" Carga: mín={carga_min:.0f}, média={carga_media:.0f}, máx={carga_max:.0f} kg")

temp_media = df['temperatura_motor_celsius'].mean()
temp_min = df['temperatura_motor_celsius'].min()
temp_max = df['temperatura_motor_celsius'].max()
print(f" Temperatura: mín={temp_min:.1f}°C, média={temp_media:.1f}°C, máx={temp_max:.1f}°C")

#  Visualização dos dados

print("\n\n" + "="*80)
print("VISUALIZANDO OS DADOS")
print("="*80 + "\n")

# Cria pasta de gráficos
import os
if not os.path.exists('graficos'):
    os.makedirs('graficos')

print("Gerando: Velocidade...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['velocidade_kmh'], bins=50, color='steelblue', edgecolor='black')
axes[0].axvline(vel_media, color='red', linestyle='--', linewidth=2, label=f'Média: {vel_media:.1f}')
axes[0].set_xlabel('Velocidade (km/h)')
axes[0].set_ylabel('Frequência')
axes[0].set_title('Distribuição de Velocidade')
axes[0].legend()

axes[1].boxplot(df['velocidade_kmh'])
axes[1].set_ylabel('Velocidade (km/h)')
axes[1].set_title('Box Plot de Velocidade')

plt.tight_layout()
plt.savefig('graficos/01_velocidade.png', dpi=100)
plt.close()
print("Salvo: graficos/01_velocidade.png")

print("Gerando: Combustível...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['combustivel_pct'], bins=50, color='green', edgecolor='black', alpha=0.7)
axes[0].axvline(comb_media, color='red', linestyle='--', linewidth=2, label=f'Média: {comb_media:.1f}%')
axes[0].set_xlabel('Combustível (%)')
axes[0].set_ylabel('Frequência')
axes[0].set_title('Distribuição de Combustível')
axes[0].legend()

# Correlação: velocidade vs combustível
axes[1].scatter(df['velocidade_kmh'], df['combustivel_pct'], alpha=0.05, s=1)
axes[1].set_xlabel('Velocidade (km/h)')
axes[1].set_ylabel('Combustível (%)')
axes[1].set_title('Combustível diminui conforme motor funciona')

plt.tight_layout()
plt.savefig('graficos/02_combustivel.png', dpi=100)
plt.close()
print("Salvo: graficos/02_combustivel.png")

#CARGA
print("Gerando: Carga...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['peso_carga_kg'], bins=50, color='orange', edgecolor='black', alpha=0.7)
axes[0].axvline(carga_media, color='red', linestyle='--', linewidth=2, label=f'Média: {carga_media:.0f}')
axes[0].set_xlabel('Carga (kg)')
axes[0].set_ylabel('Frequência')
axes[0].set_title('Distribuição de Carga Transportada')
axes[0].legend()

axes[1].scatter(df['peso_carga_kg'], df['combustivel_pct'], alpha=0.05, s=1)
axes[1].set_xlabel('Carga (kg)')
axes[1].set_ylabel('Combustível (%)')
axes[1].set_title('Carga pesada = mais consumo')

plt.tight_layout()
plt.savefig('graficos/03_carga.png', dpi=100)
plt.close()
print("Salvo: graficos/03_carga.png")

print("Gerando: Temperatura...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['temperatura_motor_celsius'], bins=50, color='red', edgecolor='black', alpha=0.7)
axes[0].axvline(temp_media, color='blue', linestyle='--', linewidth=2, label=f'Média: {temp_media:.1f}°C')
axes[0].axvline(95, color='orange', linestyle='--', linewidth=2, label='Alerta: 95°C')
axes[0].set_xlabel('Temperatura (°C)')
axes[0].set_ylabel('Frequência')
axes[0].set_title('Distribuição de Temperatura do Motor')
axes[0].legend()

# Correlação: velocidade vs temperatura
axes[1].scatter(df['velocidade_kmh'], df['temperatura_motor_celsius'], alpha=0.05, s=1)
axes[1].set_xlabel('Velocidade (km/h)')
axes[1].set_ylabel('Temperatura (°C)')
axes[1].set_title('Motor aquece quando empilhadeira se move')

plt.tight_layout()
plt.savefig('graficos/04_temperatura.png', dpi=100)
plt.close()
print("Salvo: graficos/04_temperatura.png")

# Descobrindo padrões de eficiência

print("\n\n" + "="*80)
print("="*80 + "\n")

# Empilhadeira mais/menos eficiente
consumo_por_emp = df.groupby('empilhadeira_id')['combustivel_pct'].min()
emp_mais_efic = consumo_por_emp.idxmax()  # Que tem mais combustível sobrando
emp_menos_efic = consumo_por_emp.idxmin()  # Que tem menos combustível sobrando

print(f"EFICIÊNCIA:")
print(f"   Empilhadeira MAIS eficiente: #{emp_mais_efic} (sobrou {consumo_por_emp[emp_mais_efic]:.1f}% combustível)")
print(f"   Empilhadeira MENOS eficiente: #{emp_menos_efic} (sobrou {consumo_por_emp[emp_menos_efic]:.1f}% combustível)")
print(f"   Diferença: {consumo_por_emp[emp_mais_efic] - consumo_por_emp[emp_menos_efic]:.1f}% (BIG!)")

# Operador mais/menos eficiente
consumo_por_op = df.groupby('operador_id')['combustivel_pct'].min()
op_mais_efic = consumo_por_op.idxmax()
op_menos_efic = consumo_por_op.idxmin()

print(f"\nOPERADORES:")
print(f"   Operador MAIS eficiente: #{op_mais_efic}")
print(f"   Operador MENOS eficiente: #{op_menos_efic}")

# Alertas de manutenção
temp_quente = (df['temperatura_motor_celsius'] > 95).sum()
comb_critico = (df['combustivel_pct'] < 5).sum()

print(f"\nALERTAS:")
print(f"   Motor muito quente (>95°C): {temp_quente:,} eventos")
print(f"   Combustível crítico (<5%): {comb_critico} eventos")


print("\n\n" + "="*80)
print("PRÓXIMOS PASSOS")
print("="*80 + "\n")


print("=" * 80)
print(f"Exploração concluída!")
print(f"Gráficos salvos em: graficos/")
print("=" * 80 + "\n")
