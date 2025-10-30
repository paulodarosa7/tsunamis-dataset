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



df = pd.read_csv('/home/aluno/tsunamis-dataset/tsunamis-dataset/dataset/tsunamis_database.csv')

st.title("Pagina 3 - Mapas")

st.subheader("Mapa mundi de tsunamis e maremotos")


df = df.dropna(subset=['latitude', 'longitude', 'magnitude'])


magnitude_min = st.slider("Magnitude mínima", df['magnitude'].min(), df['magnitude'].max(), df['magnitude'].min())
filtro = df[df['magnitude'] >= magnitude_min]


mapa = folium.Map(location=[0, 0], zoom_start=2)

heat_data = [[row['latitude'], row['longitude'], row['magnitude']] 
            for _, row in filtro.iterrows()]

HeatMap(heat_data, radius=10, blur=15, max_zoom=6).add_to(mapa)

html(mapa._repr_html_(), height=1000)

# https://github.com/padmalcom/streamlit_globe
from streamlit_globe import streamlit_globe

st.subheader("Globe")
pointsData=[{'lat': row['latitude'], 'lng': row['longitude'], 'size': 0.3, 'color': 'red', 'magnitude': row['magnitude']}
            for _, row in filtro.iterrows()]
labelsData=[{'lat': row['latitude'], 'lng': row['longitude'], 'size': 0.3, 'color': 'red', 'text': f'Magnitude: {row["magnitude"]}'}
            for _, row in filtro.iterrows()]
streamlit_globe(pointsData=pointsData, labelsData=labelsData, daytime='day', width=800, height=600)



#sidebar download da base
download_dataframe(df)
st.link_button("Link Dataset - Kaggle", "https://www.kaggle.com/datasets/ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset")
