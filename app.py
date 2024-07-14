
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pygsheets
import pandas as pd
import warnings
from sklearn.neighbors import BallTree
import numpy as np
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
import time
from datetime import datetime

# Configurar autorefresh
st_autorefresh(interval=5 * 60 * 60 * 1000)  # Atualiza a cada 24 horas

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Medir o tempo de execução
start_time = time.time()

# Credencial json para leitura do Sheets
cred_path = r'C:\Users\alaaraujo\Downloads\cred.json'

# Autenticação do Google baseado na Credencial
gc = pygsheets.authorize(service_account_file=cred_path)

# Abrindo a Planilha pela URL do Sheets
spreadsheet = gc.open_by_url('https://docs.google.com/spreadsheets/d/1XjX6YR0Lz5Ie9DD2DYNUFmkFqHlgc1aQsib7gxkEADI/edit?gid=938609397')

# Carregando as duas Guias do Sheets
df_1 = spreadsheet.worksheet_by_title('GEO_P').get_as_df()
df_2 = spreadsheet.worksheet_by_title('GEO_TP').get_as_df()

# Garantir que as colunas PAIS estejam presentes em ambos os DataFrames
df_1.columns = df_1.columns.str.strip().str.upper()
df_2.columns = df_2.columns.str.strip().str.upper()

# Criar DataFrame de países encontrados
df_merged = pd.merge(df_1, df_2[['PAIS']].drop_duplicates(), on='PAIS', how='inner')
df_not_found = df_1[~df_1['PAIS'].isin(df_merged['PAIS'])]

# Aplicar valores padrão para países não encontrados
df_not_found['LAT2'] = np.nan
df_not_found['LONG2'] = np.nan
df_not_found['VOLUME_SUGERIDO'] = 0
df_not_found['RUN'] = np.nan
df_not_found['Distancia_km'] = 0
df_not_found['Raio'] = 'N'
df_not_found['Precisao'] = 0
df_not_found['COUNTRY_COMPARADO'] = np.nan  # Adicionando coluna de país comparado

# Filtrando e iterando por país no DataFrame mesclado
resultados = []

for pais in df_merged['PAIS'].unique():
    df_1_pais = df_merged[df_merged['PAIS'] == pais]
    df_2_pais = df_2[df_2['PAIS'] == pais]

    # Convertendo as coordenadas para radianos
    coords_1 = np.radians(df_1_pais[['LAT', 'LONG']].astype(float).values)
    coords_2 = np.radians(df_2_pais[['LAT', 'LONG']].astype(float).values)

    # Construção BallTree do Scikit-learn para identificação da coordenada
    tree = BallTree(coords_2, metric='haversine')

    # Encontrando a coordenada mais próxima em df_2 para cada ponto em df_1
    distances, indices = tree.query(coords_1, k=1)

    # Convertendo a distância de radianos para quilômetros
    distances_km = distances.flatten() * 6371  # Raio da Terra em quilômetros

    # Adicionar colunas ao DataFrame df_1
    df_1_pais['LAT2'] = df_2_pais.iloc[indices.flatten()]['LAT'].values
    df_1_pais['LONG2'] = df_2_pais.iloc[indices.flatten()]['LONG'].values
    df_1_pais['VOLUME_SUGERIDO'] = df_2_pais.iloc[indices.flatten()]['VOLUME_SUGERIDO'].values
    df_1_pais['RUN'] = df_2_pais.iloc[indices.flatten()]['RUN'].values
    df_1_pais['Distancia_km'] = distances_km
    df_1_pais['Raio'] = np.where(df_1_pais['Distancia_km'] <= 1.5, 'S', 'N')

    # Calcular a precisão como a inversa da distância normalizada
    max_distance = 6371  # Raio da Terra em quilômetros
    df_1_pais['Precisao'] = 1 - (df_1_pais['Distancia_km'] / max_distance)

    # Adicionar coluna com o país comparado
    df_1_pais['COUNTRY_COMPARADO'] = df_2_pais.iloc[indices.flatten()]['PAIS'].values

    resultados.append(df_1_pais)

# Concatenar todos os resultados
if resultados:
    df_resultado_s = pd.concat(resultados, ignore_index=True)
else:
    df_resultado_s = pd.DataFrame(columns=df_1.columns)

# Concatenar com DataFrame de países não encontrados
df_resultado_s = pd.concat([df_resultado_s, df_not_found], ignore_index=True)

# Remover linhas onde a coluna 'PLACES' (outra coluna chave) está vazia
df_resultado_s = df_resultado_s.dropna(subset=['PLACES'])

# Garantir que as colunas estejam no formato numérico
df_resultado_s['VOLUME'] = pd.to_numeric(df_resultado_s['VOLUME'], errors='coerce')
df_resultado_s['VOLUME_SUGERIDO'] = pd.to_numeric(df_resultado_s['VOLUME_SUGERIDO'], errors='coerce')
df_resultado_s['LAT'] = pd.to_numeric(df_resultado_s['LAT'], errors='coerce')
df_resultado_s['LONG'] = pd.to_numeric(df_resultado_s['LONG'], errors='coerce')
df_resultado_s['LAT2'] = pd.to_numeric(df_resultado_s['LAT2'], errors='coerce')
df_resultado_s['LONG2'] = pd.to_numeric(df_resultado_s['LONG2'], errors='coerce')

# Verificar se a coluna 'Distancia_km' é string, caso contrário, converter para string
if df_resultado_s['Distancia_km'].dtype != 'O':  # 'O' significa object, que é o tipo de string no pandas
    df_resultado_s['Distancia_km'] = df_resultado_s['Distancia_km'].astype(str)

df_resultado_s['Distancia_km'] = pd.to_numeric(df_resultado_s['Distancia_km'].str.replace(',', '.'), errors='coerce')

# Verificar se a coluna 'Precisao' é string, caso contrário, converter para string
if df_resultado_s['Precisao'].dtype != 'O':  # 'O' significa object, que é o tipo de string no pandas
    df_resultado_s['Precisao'] = df_resultado_s['Precisao'].astype(str)

df_resultado_s['Precisao'] = pd.to_numeric(df_resultado_s['Precisao'].str.replace(',', '.'), errors='coerce')

# Formatando colunas para string com separadores corretos
df_resultado_s['LONG2'] = df_resultado_s['LONG2'].apply(lambda x: f'{x:.6f}' if not pd.isna(x) else '')
#df_resultado_s['Distancia_km'] = df_resultado_s['Distancia_km'].apply(lambda x: f'{float(x):.6f}'.replace('.', ',') if not pd.isna(x) else '0,000000')
df_resultado_s['Precisao'] = df_resultado_s['Precisao'].apply(lambda x: f'{x:.6f}'.replace('.', ','))

# Selecionar as colunas na ordem desejada
df_resultado_s = df_resultado_s[['PLACES', 'FANTASIA', 'GOLIVE', 'LAT', 'LONG', 'VOLUME', 'VOLUME_SUGERIDO', 'LAT2', 'LONG2', 'Distancia_km', 'Raio', 'RUN', 'Precisao', 'PAIS', 'COUNTRY_COMPARADO']]

# Medir o tempo de carregamento dos dados
data_load_time = time.time() - start_time
# Medir o tempo total de execução
total_time = time.time() - start_time
end_datetime = datetime.now()
formatted_end_date = end_datetime.strftime('%d de %B de %Y, %H:%M:%S')

# Função para tradução dos textos
def translate(text, lang):
    translations = {
        "pt-br": {
            "Análise de Dados Geográficos": "Análise de Dados Geográficos",
            "Aderência": "Aderência",
            "Resumo da Análise Geográfica": "Resumo da Análise Geográfica",
            "Aderência de Novos Places Dentro do Raio <=1.5km": "Aderência de Novos Places Dentro do Raio <=1.5km",
            "Distribuição dos Places no Mapa": "Distribuição dos Places no Mapa",
            "Tabela de Resultados": "Tabela de Resultados",
            "Olá, Seja Bem Vindo! Estes dados foram atualizados em": "Olá, Seja Bem Vindo! Estes dados foram atualizados em",
            "Geos New Places x Geos Sugeridas The Place <=1.5km": "Geos Novos Places x Geos Sugeridas The Place <=1.5km",
            "Geo Sugeridas": "Geo Sugeridas",
            "Quantidade de novos Places no mês": "Quantidade de novos Places no mês",
            "Places <=1.5km": "Places <=1.5km",
            "% Oportunidade capturada do The Placer": "% Oportunidade capturada do The Placer",
            "% Ativação com aderência ao The Placer": "% Ativação com aderência ao The Placer",
            "Com Places Abertos em Raio 1.5km": "Com Places Abertos em Raio 1.5km",
            "Sem Places Abertos em Raio 1.5km": "Sem Places Abertos em Raio 1.5km",
            "Aderência:": "Aderência:",
            "Distribuição dos Places no Mapa": "Distribuição dos Places no Mapa",
            "PLACE NO RAIO DE 1.5km": "PLACE NO RAIO DE 1.5km",
            "VOLUME RECEBIDO": "VOLUME RECEBIDO",
            "VOLUME SUGERIDO": "VOLUME SUGERIDO",
            "SUGERIDO": "SUGERIDO",
            "COORDENADAS": "COORDENADAS",
            "DIST_CALC": "DIST_CALC",
            "PRECISAO": "PRECISAO",
            "GEO NOVOS PLACES X GEO SUGERIDA.": "GEO NOVOS PLACES X GEO SUGERIDA.",
            "Legenda": "Legenda",
            "GEO SUGERIDA": "GEO SUGERIDA",
            "PLACES NOVOS <=1.5km DA GEO COM PACOTES": "PLACES NOVOS <=1.5km DA GEO COM PACOTES",
            "PLACES NOVOS <=1.5km DA GEO S/PACOTES": "PLACES NOVOS <=1.5km DA GEO S/PACOTES",
            "PLACES NOVOS >1.5km DA GEO": "PLACES NOVOS >1.5km DA GEO",
            "Tabela de Resultados": "Tabela de Resultados",
            "Dispersão do volume médio diário realizado vs esperado": "Dispersão do volume médio diário realizado vs esperado",
            "Distância média do Place aberto vs Place sugerido do The Placer": "Distância média do Place aberto vs Place sugerido do The Placer"
        },
        "es": {
            "Análise de Dados Geográficos": "Análisis de Datos Geográficos",
            "Aderência": "Adherencia",
            "Resumo da Análise Geográfica": "Resumen del Análisis Geográfico",
            "Aderência de Novos Places Dentro do Raio <=1.5km": "Adherencia de Nuevos Places Dentro del Radio <=1.5km",
            "Distribuição dos Places no Mapa": "Distribución de los Places en el Mapa",
            "Tabela de Resultados": "Tabla de Resultados",
            "Olá, Seja Bem Vindo! Estes dados foram atualizados em": "¡Hola es bienvenido! Estos datos fueron actualizados en",
            "Geos New Places x Geos Sugeridas The Place <=1.5km": "Places Nuevos x Geos Sugeridos en The Place <=1.5km",
            "Geo Sugeridas": "Geos Sugeridas",
            "Quantidade de novos Places no mês": "Cantidad de nuevos Places en el mes",
            "Places <=1.5km": "Lugares <=1.5km",
            "% Oportunidade capturada do The Placer": "% Oportunidad capturada de The Placer",
            "% Ativação com aderência ao The Placer": "% Activación con adherencia a The Placer",
            "Com Places Abertos em Raio 1.5km": "Con Places Abiertos en Radio de 1.5km",
            "Sem Places Abertos em Raio 1.5km": "Sin Places Abiertos en Radio de 1.5km",
            "Aderência:": "Adherencia:",
            "Distribuição dos Places no Mapa": "Distribución de los Places en el Mapa",
            "PLACE NO RAIO DE 1.5km": "PLACE EN EL RADIO DE 1.5km",
            "VOLUME RECEBIDO": "VOLUMEN RECIBIDO",
            "VOLUME SUGERIDO": "VOLUMEN SUGERIDO",
            "SUGERIDO": "SUGERIDO",
            "COORDENADAS": "COORDENADAS",
            "DIST_CALC": "DIST_CALC",
            "PRECISAO": "PRECISION",
            "GEO NOVOS PLACES X GEO SUGERIDA.": "GEO NUEVOS PLACES X GEO SUGERIDA.",
            "Legenda": "Leyenda",
            "GEO SUGERIDA": "GEO SUGERIDA",
            "PLACES NOVOS <=1.5km DA GEO COM PACOTES": "PLACES NUEVOS <=1.5km DE LA GEO CON PAQUETES",
            "PLACES NOVOS <=1.5km DA GEO S/PACOTES": "PLACES NUEVOS <=1.5km DE LA GEO SIN PAQUETES",
            "PLACES NOVOS >1.5km DA GEO": "PLACES NUEVOS >1.5km DE LA GEO",
            "Tabela de Resultados": "Tabla de Resultados",
            "Dispersão do volume médio diário realizado vs esperado": "Dispersión del volumen medio diario realizado vs esperado",
            "Distância média do Place aberto vs Place sugerido do The Placer": "Distancia media del Place abierto vs Place sugerido de The Placer"
        }
    }
    return translations[lang].get(text, text)

# Função para alternar o idioma
def switch_language():
    if 'language' not in st.session_state:
        st.session_state.language = 'pt-br'
    return st.selectbox("Select Language", options=['pt-br', 'es'], index=0 if st.session_state.language == 'pt-br' else 1, key='language_selector')


# Escrever a data e hora de término em um arquivo de log
with open("last_run_log.txt", "w") as log_file:
    log_file.write(formatted_end_date)

# Ler a data e hora de execução do arquivo de log
with open("last_run_log.txt", "r") as log_file:
    last_run = log_file.read()

# Cabeçalho com seleção de idioma
st.markdown(f"""
    <style>
        .header {{
            background-color: #D1E8E2;
            padding: 10px;
            text-align: center;
            font-size: 30px;
            color: black;
        }}
        .footer {{
            background-color: #F1F1F1;
            padding: 10px;
            text-align: center;
            font-size: 12px;
            color: black;
            position: fixed;
            bottom: 0;
            width: 100%;
        }}
        .greeting {{
            text-align: right;
            font-size: 16px;
            color: black;
        }}
    </style>
    <div class="header">
        HUNTING REGIONAL teste
    </div>
""", unsafe_allow_html=True)

# Detectando a troca de idioma
current_language = switch_language()

# Mensagem de saudação e seleção de idioma
current_date = datetime.now().strftime('%d de %B de %Y')
greeting = translate('Olá, Seja Bem Vindo! Estes dados foram atualizados em', current_language)


st.markdown(f'<div class="greeting">{greeting} {last_run}</div>', unsafe_allow_html=True)

st.title(translate('Análise de Dados Geográficos', current_language))

# Filtro de país
paises = df_resultado_s['PAIS'].unique().tolist()
pais_selecionado = st.multiselect(translate('Selecione o(s) País(es)', current_language), paises, default=paises, key='pais_selecionado')

# Verificar se pelo menos um país foi selecionado
if not pais_selecionado:
    st.warning(translate('Por favor, selecione pelo menos um país.', current_language))
    st.stop()

# Filtrar dados com base nos países selecionados, se houver seleção
if pais_selecionado:
    df_filtrado = df_resultado_s[df_resultado_s['PAIS'].isin(pais_selecionado)]
else:
    df_filtrado = df_resultado_s.copy()

# Filtro de distância
distancia_opcoes = ['S', 'N']
distancia_selecionada = st.multiselect(translate('Selecionar Dentro do Raio', current_language), distancia_opcoes, default=distancia_opcoes, key='distancia_selecionada')

# Aplicar filtro de distância, se houver seleção
if distancia_selecionada:
    df_filtrado = df_filtrado[df_filtrado['Raio'].isin(distancia_selecionada)]

# Filtro de PLACE, obedecendo aos filtros de país e raio, se aplicável
places_disponiveis = df_filtrado['PLACES'].unique().tolist()
place_selecionado = st.selectbox(translate('Selecione o Place', current_language), options=[''] + places_disponiveis, index=0, key='place_selecionado')

# Aplicar filtro de PLACE, se houver seleção
if place_selecionado:
    df_filtrado = df_filtrado[df_filtrado['PLACES'] == place_selecionado]

# Exibir a aderência com gráfico de velocímetro
st.subheader(translate('Aderência', current_language))

# Dados Variáveis
geo_sugeridas_total = df_2[df_2['PAIS'].isin(pais_selecionado)].shape[0]
Places_Novos_1km_com_pacotes = df_filtrado.query("Raio == 'S' and VOLUME > 0").shape[0]
Places_Novos_1km_sem_pacotes = df_filtrado.query("Raio == 'S' and VOLUME == 0").shape[0]
Places_Novos_mais_1km = df_filtrado.query("Raio == 'N'").shape[0]
Places_Novos_total = df_filtrado.shape[0]
if geo_sugeridas_total > 0:
    Places_Novos_1km_x_the_place = (Places_Novos_1km_com_pacotes + Places_Novos_1km_sem_pacotes) / geo_sugeridas_total * 100
else:
    Places_Novos_1km_x_the_place = 0

if (Places_Novos_1km_com_pacotes + Places_Novos_1km_sem_pacotes + Places_Novos_mais_1km) > 0:
    Places_Novos_1km_x_novos_geral = (Places_Novos_1km_com_pacotes + Places_Novos_1km_sem_pacotes) / (Places_Novos_1km_com_pacotes + Places_Novos_1km_sem_pacotes + Places_Novos_mais_1km) * 100
else:
    Places_Novos_1km_x_novos_geral = 0



# Criar gráfico de velocímetro
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=Places_Novos_1km_x_the_place,
    title={'text': translate("Geos New Places x Geos Sugeridas The Place <=1.5km", current_language)},
    number={'suffix': "%", 'valueformat': '.2f'},
    gauge={
        'axis': {'range': [None, 100]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [0, 50], 'color': 'lightgray'},
            {'range': [50, 80], 'color': 'gray'},
            {'range': [80, 100], 'color': 'green'}
        ],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 80
        }
    }
))

fig_gauge.update_layout(height=400)
st.plotly_chart(fig_gauge)

# Resumo da Análise Geográfica
st.subheader(translate('Resumo da Análise Geográfica', current_language))

# Dados para a tabela
novos_places_no_periodo = Places_Novos_total
places_1km = df_filtrado.query("Raio == 'S'").shape[0]
if novos_places_no_periodo > 0:
    aderencia_places_no_raio = places_1km / novos_places_no_periodo * 100
else:
    aderencia_places_no_raio = 0

aderencia_geos_the_place = Places_Novos_1km_x_the_place

# Calcular a dispersão do volume médio diário realizado vs esperado
dias_uteis = 22
volume_diario_realizado = df_filtrado['VOLUME'].sum() / dias_uteis
volume_diario_sugerido = df_filtrado['VOLUME_SUGERIDO'].sum() / dias_uteis
dispersao_volume = ((volume_diario_realizado - volume_diario_sugerido) / volume_diario_sugerido) * 100

# Calcular a distância média do Place aberto vs Place sugerido do The Placer
distancia_media = df_filtrado['Distancia_km'].mean()

table_data = [
    [translate("Geo Sugeridas", current_language), geo_sugeridas_total],
    [translate("Quantidade de novos Places no mês", current_language), novos_places_no_periodo],
    [translate("Places <=1.5km", current_language), places_1km],
    [translate("% Oportunidade capturada do The Placer", current_language), f"{aderencia_places_no_raio:.2f}%"],
    [translate("% Ativação com aderência ao The Placer", current_language), f"{aderencia_geos_the_place:.2f}%"],
    [translate("Dispersão do volume médio diário realizado vs esperado", current_language), f"{dispersao_volume:.2f}%"],
    [translate("Distância média do Place aberto vs Place sugerido do The Placer", current_language), f"{distancia_media:.2f} km"]
]

# Exibir a tabela
st.table(pd.DataFrame(table_data, columns=[translate("Métrica", current_language), translate("Valor", current_language)]))

# Gráfico de Pizza
st.subheader(translate('Aderência de Novos Places Dentro do Raio <=1.5km', current_language))

fig = go.Figure(data=[
    go.Pie(labels=[translate('Com Places Abertos em Raio 1.5km', current_language), translate('Sem Places Abertos em Raio 1.5km', current_language)],
           values=[Places_Novos_1km_com_pacotes + Places_Novos_1km_sem_pacotes, geo_sugeridas_total - (Places_Novos_1km_com_pacotes + Places_Novos_1km_sem_pacotes)],
           hole=0.3)
])

fig.update_layout(title_text=translate(f'Aderência: {Places_Novos_1km_x_the_place:.2f}% Places Novos dentro de 1.5 km da Sugestão do The Place', current_language), height=600)
st.plotly_chart(fig)

# Mapa Interativo
st.subheader(translate('Distribuição dos Places no Mapa', current_language))

# Adicionar legenda abaixo do mapa
legend_html = f"""
<div style='padding: 10px; background-color:white; width: 100%;'>
<b>{translate("Legenda", current_language)}</b> <br>
<i class="fa fa-map-marker fa-2x" style="color:blue; font-size:20px;">&#9679;</i>&nbsp; <span style="font-size: 8px;">{translate("GEO SUGERIDA", current_language)}</span><br>
<i class="fa fa-map-marker fa-2x"style="color:green; font-size:20px;">&#9679;</i>&nbsp; <span style="font-size: 8px;">{translate("PLACES NOVOS <=1.5km DA GEO COM PACOTES", current_language)}</span><br>
<i class="fa fa-map-marker fa-2x" style="color:red; font-size:20px;">&#9679;</i>&nbsp; <span style="font-size: 8px;">{translate("PLACES NOVOS <=1.5km DA GEO S/PACOTES", current_language)}</span><br>
<i class="fa fa-map-marker fa-2x" style="color:purple; font-size:20px;">&#9679;</i>&nbsp; <span style="font-size: 8px;">{translate("PLACES NOVOS >1.5km DA GEO", current_language)}</span><br>
</div>
"""

st.markdown(legend_html, unsafe_allow_html=True)

# Garantir que as colunas VOLUME, VOLUME_SUGERIDO e Precisao estejam no formato numérico
df_filtrado['VOLUME'] = pd.to_numeric(df_filtrado['VOLUME'], errors='coerce')
df_filtrado['VOLUME_SUGERIDO'] = pd.to_numeric(df_filtrado['VOLUME_SUGERIDO'], errors='coerce')
df_filtrado['Precisao'] = pd.to_numeric(df_filtrado['Precisao'].str.replace(',', '.'), errors='coerce')

# Inicializar o mapa centrado na média das coordenadas do DataFrame com largura e altura específicas
mapa = folium.Map(
    location=[df_filtrado['LAT'].mean(), df_filtrado['LONG'].mean()],
    zoom_start=2,
    width='100%',  # Ajuste a largura do mapa
    height='100%'  # Ajuste a altura do mapa
)

# Adicionar pontos com cores baseadas no valor de VOLUME e Raio
for _, row in df_filtrado.iterrows():
    if row['Raio'] == 'S':
        if pd.isna(row['VOLUME']) or row['VOLUME'] == 0:
            color = 'red'
        elif row['VOLUME'] > 0:
            color = 'green'
        else:
            color = 'red'  # Caso tenha outros valores, define a cor como vermelha
    else:
        color = 'purple'  # Defina a cor como roxa para Raio == 'N'

    popup_text = f"""
    <b><span style='color:{color};'>{translate("PLACE NO RAIO DE 1.5km", current_language)}</span></b> = {row['PLACES']} - {row['FANTASIA']} <br>
    <b><span style='color:{color};'>GOLIVE</span></b> = {row['GOLIVE']}<br>
    <b><span style='color:{color};'>{translate("VOLUME RECEBIDO", current_language)}</span></b> = {row['VOLUME']}<br>
    <b><span style='color:{color};'>{translate("VOLUME SUGERIDO", current_language)}</span></b> = {row['VOLUME_SUGERIDO']}<br>
    <b><span style='color:{color};'>%{translate("SUGERIDO", current_language)}</span></b> = {round(((row['VOLUME']/row['VOLUME_SUGERIDO'])*100),2) if row['VOLUME_SUGERIDO'] != 0 else 'N/A'}%<br>
    <b><span style='color:{color};'>{translate("COORDENADAS", current_language)}</span></b> = {row['LAT']}, {row['LONG']}<br>
    <b><span style='color:{color};'>{translate("DIST_CALC", current_language)}</span></b> = {row['Distancia_km']}<br>
    <b><span style='color:{color};'>{translate("PRECISAO", current_language)}</span></b> = {round(float(row['Precisao'])*100, 4)}%
    """
    folium.Marker(
        location=[row['LAT'], row['LONG']],
        popup=folium.Popup(popup_text, max_width=300),
        icon=folium.Icon(color=color, icon='info-sign')
    ).add_to(mapa)

    # Adicionar linha entre a coordenada original e sugerida
    folium.PolyLine(
        locations=[(row['LAT'], row['LONG']), (row['LAT2'], row['LONG2'])],
        color='blue',
        weight=2,
        opacity=0.7
    ).add_to(mapa)

# Adicionar pontos azuis para LAT2 e LONG2 (geo_sugerida)
for _, row in df_filtrado.iterrows():
    popup_text = f"""
    <b><span style='color:blue;'>{translate("PLACE_SUGERIDO", current_language)}</span></b> = x<br>
    <b><span style='color:blue;'>{translate("VOLUME SUGERIDO", current_language)}</span></b> = {row['VOLUME_SUGERIDO']}<br>
    <b><span style='color:blue;'>{translate("COORDENADAS", current_language)}</span></b> = {row['LAT2']}, {row['LONG2']}<br>
    """
    folium.Marker(
        location=[row['LAT2'], row['LONG2']],
        popup=folium.Popup(popup_text, max_width=300),
        icon=folium.Icon(color='blue', icon='info-sign', icon_size=(15, 15))
    ).add_to(mapa)

titulo_html = f"""
<h4 style="position: fixed;
            top: 0px; left: 12.0%; transform: translateX(-10%);
            background-color: white; z-index:9999;
            padding: 3px; border: 2px solid black;">
    {translate("GEO NOVOS PLACES X GEO SUGERIDA.", current_language)}
</h4>
"""
mapa.get_root().html.add_child(folium.Element(titulo_html))

folium_static(mapa)

# Mostrar tabela filtrada
st.subheader(translate('Tabela de Resultados', current_language))

# Remover separadores de milhar das colunas numéricas
df_filtrado['VOLUME'] = df_filtrado['VOLUME'].apply(lambda x: str(x).replace(',', ''))
df_filtrado['VOLUME_SUGERIDO'] = df_filtrado['VOLUME_SUGERIDO'].apply(lambda x: str(x).replace(',', ''))

# Renomear colunas para exibição
df_filtrado['PLACES'] = df_filtrado['PLACES'].astype(str).str.replace(',', '')
df_filtrado = df_filtrado.drop(columns=['LAT', 'LONG', 'LAT2', 'LONG2', 'COUNTRY_COMPARADO'])
df_filtrado.columns = ['PLACES', 'FANTASIA', 'GOLIVE', 'VOL REL', 'VOL SUG', 'DIST_KM', '<=1.5KM', 'RUN', 'PRECISAO', 'PAIS']
st.dataframe(df_filtrado)

# Rodapé
st.markdown('<div class="footer">Power By Hunting Regional © 2024</div>', unsafe_allow_html=True)

# Medir o tempo total de execução
total_time = time.time() - start_time
st.write(f"Tempo total de execução: {total_time:.2f} segundos")
