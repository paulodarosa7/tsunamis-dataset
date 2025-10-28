import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import folium
from folium.plugins import HeatMap
import folium
from streamlit.components.v1 import html
from more.sidebar import download_dataframe


st.title("🌊 Earthquake & Tsunami Risk - Global Dataset")

df = pd.read_csv('/home/operador/Documentos/analise de Dados/tsunamis-dataset/dataset/tsunamis_database.csv')

st.title("Pagina 3 - Mapas")

st.subheader("Mapa mundi de tsunamis e maremotos")

# Remove valores ausentes em latitude e longitude
df = df.dropna(subset=['latitude', 'longitude', 'magnitude'])

# Filtro opcional para magnitude mínima
magnitude_min = st.slider("Magnitude mínima", df['magnitude'].min(), df['magnitude'].max(), df['magnitude'].min())
filtro = df[df['magnitude'] >= magnitude_min]


# Cria mapa
mapa = folium.Map(location=[0, 0], zoom_start=2)

# Cria lista com localização e peso
heat_data = [[row['latitude'], row['longitude'], row['magnitude']] 
            for _, row in filtro.iterrows()]

# Adiciona o mapa de calor
HeatMap(heat_data, radius=10, blur=15, max_zoom=6).add_to(mapa)

# Exibe no Streamlit
html(mapa._repr_html_(), height=1000)

# utilização streamlit globe
# https://github.com/padmalcom/streamlit_globe
from streamlit_globe import streamlit_globe

st.subheader("Globe")
pointsData=[{'lat': row['latitude'], 'lng': row['longitude'], 'size': 0.3, 'color': 'red', 'magnitude': row['magnitude']}
            for _, row in filtro.iterrows()]
labelsData=[{'lat': row['latitude'], 'lng': row['longitude'], 'size': 0.3, 'color': 'red', 'text': f'Magnitude: {row["magnitude"]}'}
            for _, row in filtro.iterrows()]
streamlit_globe(pointsData=pointsData, labelsData=labelsData, daytime='day', width=800, height=600)

from streamlit_apexjs import st_apexcharts

quantidade_tsunami = filtro['tsunami'].value_counts()

options = {
    'chart': {
        'toolbar': {
            'show': True,
            'texto-color': 'k'
        }
    },
    'labels': [f'Tsunami: {quantidade_tsunami.get(1, 0)}', f'Sem Tsunami: {quantidade_tsunami.get(0, 0)}'],  # Ajustando os rótulos com base nos valores
    'legend': {
        'show': True,
        'position': 'top',
    }
}

series = [int(quantidade_tsunami.get(1, 0)), int(quantidade_tsunami.get(0, 0))]  # Conta o número de tsunamis (1) e não-tsunamis (0)

# Cria o gráfico donut com os dados corrigidos
st.subheader("Distribuição de Tsunamis")
st_apexcharts(options, series, 'donut','600', "Proporção Tsunami e Não Tsunami")

#sidebar download da base
download_dataframe(df)
