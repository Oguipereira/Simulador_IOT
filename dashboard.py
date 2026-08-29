
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


st.set_page_config(
    page_title="Telemetria Logística Industrial",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tema visual
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

#carregar os dados do modelo 

@st.cache_resource
def carregar_modelo():
    """Carrega modelo treinado (só uma vez)"""
    with open('modelos/xgboost_consumo.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def carregar_feature_names():
    """Carrega nomes das features (só uma vez)"""
    with open('modelos/feature_names.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def carregar_dados():
    """Carrega dados (só uma vez)"""
    return pd.read_parquet('dados/telemetria.parquet')

# Carrega tudo
modelo = carregar_modelo()
feature_names = carregar_feature_names()
df = carregar_dados()

# HEADER

st.title("Dashboard de Telemetria - Logística Industrial")
st.markdown("---")

st.write("""
Sistema inteligente de análise e previsão de consumo de combustível para frota de empilhadeiras.
Utilize este dashboard para visualizar dados históricos, analisar eficiência operacional e fazer previsões.
""")

# Sidebar

st.sidebar.title("Navegação")
pagina = st.sidebar.radio("Escolha uma página:", [
    "Dashboard",
    "Preditor de Consumo",
    "Análise de Dados",
    "Eficiência de Operadores",
    "Eficiência de Empilhadeiras"
])

# Pagina 1 _ Dash geral 

if pagina == "Dashboard":
    st.header("Dashboard")
    st.markdown("Visão geral da operação")

    # KPIs principais
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_emp = df['empilhadeira_id'].nunique()
        st.metric(
            label="Empilhadeiras",
            value=f"{total_emp}",
            delta="Ativas"
        )

    with col2:
        total_op = df['operador_id'].nunique()
        st.metric(
            label="Operadores",
            value=f"{total_op}",
            delta="Únicos"
        )

    with col3:
        consumo_medio = (100 - df['combustivel_pct'].mean())
        st.metric(
            label="Consumo Médio",
            value=f"{consumo_medio:.1f}%",
            delta="Por dia"
        )

    with col4:
        temp_media = df['temperatura_motor_celsius'].mean()
        st.metric(
            label="Temp. Média",
            value=f"{temp_media:.1f}°C",
            delta="Motores"
        )

    st.markdown("---")

    # Gráficos
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribuição de Velocidade")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df['velocidade_kmh'], bins=50, color='steelblue', edgecolor='black')
        ax.axvline(df['velocidade_kmh'].mean(), color='red', linestyle='--', label=f"Média: {df['velocidade_kmh'].mean():.1f}")
        ax.set_xlabel('Velocidade (km/h)')
        ax.set_ylabel('Frequência')
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Distribuição de Combustível")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df['combustivel_pct'], bins=50, color='green', edgecolor='black', alpha=0.7)
        ax.axvline(df['combustivel_pct'].mean(), color='red', linestyle='--', label=f"Média: {df['combustivel_pct'].mean():.1f}%")
        ax.set_xlabel('Combustível (%)')
        ax.set_ylabel('Frequência')
        ax.legend()
        st.pyplot(fig)
        plt.close()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Combustível vs Velocidade")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(df['velocidade_kmh'], df['combustivel_pct'], alpha=0.05, s=1)
        ax.set_xlabel('Velocidade (km/h)')
        ax.set_ylabel('Combustível (%)')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Carga vs Combustível")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(df['peso_carga_kg'], df['combustivel_pct'], alpha=0.05, s=1, color='orange')
        ax.set_xlabel('Carga (kg)')
        ax.set_ylabel('Combustível (%)')
        st.pyplot(fig)
        plt.close()

#Predição do consumo

elif pagina == "Preditor de Consumo":
    st.header("Preditor de Consumo - Previsões em Tempo Real")
    st.markdown("Insira os parâmetros da rota e obtenha previsão de consumo")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Parâmetros da Rota")

        velocidade = st.slider(
            "Velocidade (km/h)",
            min_value=0.0,
            max_value=20.0,
            value=10.0,
            step=0.5
        )

        peso_carga = st.slider(
            "Peso da Carga (kg)",
            min_value=0,
            max_value=3000,
            value=1000,
            step=100
        )

        temperatura = st.slider(
            "Temperatura do Motor (°C)",
            min_value=50.0,
            max_value=110.0,
            value=75.0,
            step=1.0
        )

        empilhadeira_id = st.number_input(
            "ID da Empilhadeira",
            min_value=1,
            max_value=300,
            value=1,
            step=1
        )

        operador_id = st.number_input(
            "ID do Operador",
            min_value=1000,
            max_value=9999,
            value=1500,
            step=1
        )

    with col2:
        st.subheader("Parâmetros Temporais")

        hora = st.slider(
            "Hora do Dia",
            min_value=8,
            max_value=17,
            value=10
        )

        dia_semana = st.selectbox(
            "Dia da Semana",
            options=[0, 1, 2, 3, 4],
            format_func=lambda x: ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta'][x]
        )

        semana_mes = st.slider(
            "Semana do Mês",
            min_value=1,
            max_value=5,
            value=2
        )

        # Calcula features derivadas
        razao_carga_vel = peso_carga / (velocidade + 1)
        em_movimento = 1 if velocidade > 0 else 0
        temp_normalizada = np.clip((temperatura - 60) / 40 * 100, 0, 100)
        combustivel_consumido_input = st.slider(
            "Combustível já Consumido (%)",
            min_value=0.0,
            max_value=100.0,
            value=30.0,
            step=5.0
        )

    # Fazer previsão
    if st.button("🔮 Fazer Previsão", use_container_width=True):
        # Preparar dados pra previsão
        dados_predicao = pd.DataFrame({
            'empilhadeira_id': [empilhadeira_id],
            'operador_id': [operador_id],
            'velocidade_kmh': [velocidade],
            'peso_carga_kg': [peso_carga],
            'temperatura_motor_celsius': [temperatura],
            'hora': [hora],
            'dia_semana': [dia_semana],
            'semana_mes': [semana_mes],
            'razao_carga_vel': [razao_carga_vel],
            'em_movimento': [em_movimento],
            'temp_normalizada': [temp_normalizada]
        })

        # Fazer previsão
        consumo_previsto = modelo.predict(dados_predicao)[0]

        st.markdown("---")
        st.subheader("Resultado da Previsão")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Consumo Previsto",
                value=f"{consumo_previsto:.2f}%",
                delta="Do tanque"
            )

        with col2:
            combustivel_restante = 100 - consumo_previsto
            st.metric(
                label="Combustível Restante",
                value=f"{combustivel_restante:.2f}%",
                delta="Aprox."
            )

        with col3:
            if consumo_previsto > 60:
                alerta = "Alto"
            elif consumo_previsto > 40:
                alerta = "Médio"
            else:
                alerta = "Baixo"
            st.metric(
                label="Nível de Risco",
                value=alerta
            )

        st.markdown("---")
        st.info(f"""
        **Interpretação:**
        • Consumo previsto: {consumo_previsto:.2f}%
        • Combustível restante: {combustivel_restante:.2f}%
        • Com esses parâmetros, o motor deve consumir aproximadamente {consumo_previsto:.2f}% do tanque
        """)

# ============================================================================
# PÁGINA 3: ANÁLISE DE DADOS
# ============================================================================

elif pagina == "Análise dos Dados":
    st.header("Análise Detalhada de Dados")

    # Filtros
    col1, col2 = st.columns(2)

    with col1:
        empilhadeiras_selecionadas = st.multiselect(
            "Selecione Empilhadeiras",
            options=sorted(df['empilhadeira_id'].unique()),
            default=[1, 2, 3],
            max_selections=10
        )

    with col2:
        operadores_selecionados = st.multiselect(
            "Selecione Operadores",
            options=sorted(df['operador_id'].unique()),
            default=list(sorted(df['operador_id'].unique()))[:5]
        )

    # Filtrar dados
    df_filtrado = df[
        (df['empilhadeira_id'].isin(empilhadeiras_selecionadas)) &
        (df['operador_id'].isin(operadores_selecionados))
    ]

    # Estatísticas
    st.subheader("Estatísticas dos Dados Filtrados")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total de Eventos", f"{len(df_filtrado):,}")

    with col2:
        st.metric("Velocidade Média", f"{df_filtrado['velocidade_kmh'].mean():.1f} km/h")

    with col3:
        st.metric("Consumo Médio", f"{(100 - df_filtrado['combustivel_pct'].mean()):.1f}%")

    with col4:
        st.metric("Temp. Máxima", f"{df_filtrado['temperatura_motor_celsius'].max():.1f}°C")

    # Gráficos dos dados filtrados
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribuição de Velocidade (Filtrada)")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df_filtrado['velocidade_kmh'], bins=50, color='steelblue', edgecolor='black')
        ax.set_xlabel('Velocidade (km/h)')
        ax.set_ylabel('Frequência')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Distribuição de Combustível (Filtrada)")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df_filtrado['combustivel_pct'], bins=50, color='green', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Combustível (%)')
        ax.set_ylabel('Frequência')
        st.pyplot(fig)
        plt.close()

#sinceramente nem sei se esse é realmente valido, mas vou deixar assim por enquanto

elif pagina == "Eficiência de Operadores":
    st.header("Análise de Eficiência por Operador")

    # Análise por operador
    eficiencia_op = df.groupby('operador_id').agg({
        'combustivel_pct': ['min', 'mean'],
        'velocidade_kmh': 'mean',
        'peso_carga_kg': 'mean',
        'timestamp': 'count'
    }).round(2)

    eficiencia_op.columns = ['comb_min', 'comb_media', 'vel_media', 'carga_media', 'num_eventos']
    eficiencia_op['consumo'] = (100 - eficiencia_op['comb_min']).round(2)
    eficiencia_op = eficiencia_op.sort_values('consumo')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Operadores MAIS Eficientes")
        top_eficientes = eficiencia_op.head(10)[['consumo', 'vel_media', 'num_eventos']]
        st.dataframe(top_eficientes, use_container_width=True)

    with col2:
        st.subheader("Operadores MENOS Eficientes")
        top_ineficientes = eficiencia_op.tail(10)[['consumo', 'vel_media', 'num_eventos']]
        st.dataframe(top_ineficientes, use_container_width=True)

    # Gráfico
    st.subheader("Distribuição de Consumo por Operador")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(eficiencia_op['consumo'], bins=30, color='steelblue', edgecolor='black')
    ax.set_xlabel('Consumo Total (%)')
    ax.set_ylabel('Número de Operadores')
    ax.axvline(eficiencia_op['consumo'].mean(), color='red', linestyle='--', label='Média')
    ax.legend()
    st.pyplot(fig)
    plt.close()

#Medindo a eficiencias das empilhadeira 
# esse é o mel 

elif pagina == "Eficiência de Empilhadeiras":
    st.header("Análise de Eficiência por Empilhadeira")

    # Análise por empilhadeira
    eficiencia_emp = df.groupby('empilhadeira_id').agg({
        'combustivel_pct': ['min', 'mean'],
        'velocidade_kmh': 'mean',
        'peso_carga_kg': 'mean',
        'temperatura_motor_celsius': 'mean',
        'timestamp': 'count'
    }).round(2)

    eficiencia_emp.columns = ['comb_min', 'comb_media', 'vel_media', 'carga_media', 'temp_media', 'num_eventos']
    eficiencia_emp['consumo'] = (100 - eficiencia_emp['comb_min']).round(2)
    eficiencia_emp = eficiencia_emp.sort_values('consumo')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Empilhadeiras MAIS Eficientes")
        top_eficientes = eficiencia_emp.head(10)[['consumo', 'vel_media', 'temp_media']]
        st.dataframe(top_eficientes, use_container_width=True)

    with col2:
        st.subheader("Empilhadeiras MENOS Eficientes (Manutenção?)")
        top_ineficientes = eficiencia_emp.tail(10)[['consumo', 'vel_media', 'temp_media']]
        st.dataframe(top_ineficientes, use_container_width=True)

    # Gráfico
    st.subheader("Distribuição de Consumo por Empilhadeira")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(eficiencia_emp['consumo'], bins=30, color='orange', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Consumo Total (%)')
    ax.set_ylabel('Número de Empilhadeiras')
    ax.axvline(eficiencia_emp['consumo'].mean(), color='red', linestyle='--', label='Média')
    ax.legend()
    st.pyplot(fig)
    plt.close()


st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    Dashboard de Telemetria v1.0 | Desenvolvido com Streamlit e XGBoost
</div>
""", unsafe_allow_html=True)