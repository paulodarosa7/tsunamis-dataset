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



st.title("Pagina 1 - Entendendo a Base de Dados")

st.subheader("Primeiras linhas do dataset")
st.dataframe(df.head())

st.write("Dimensões da base:", df.shape)

st.subheader("Informações gerais")
col = st.selectbox("escolha uma coluna para ver seus dados: ", df.columns)
st.write(df[col].head())

st.write("Grafico de pizza por incidentes por ano")

percent_anual = df.groupby('Year')['tsunami'].mean() * 100

fig, ax = plt.subplots(figsize=(10, 10))
ax.pie(percent_anual, labels=percent_anual.index, autopct='%1.1f%%')
ax.set_title("Percentual de tsunamis por ano")

st.pyplot(fig)

st.write("Relação por mês do ano")
import matplotlib.pyplot as plt

# Filtrar apenas eventos com tsunami confirmado
df_tsunamis_confirmados = df[df["tsunami"] == 1]

tsunamis_por_mes = df_tsunamis_confirmados["Month"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(tsunamis_por_mes.index, tsunamis_por_mes.values)

ax.set_title("Quantidade de tsunamis por mês")
ax.set_xlabel("Mês")
ax.set_ylabel("Número de Tsunamis")
ax.set_xticks(range(1, 13))

st.pyplot(fig)

st.write("Caso queira, pode escolher um ano e um mês para ver os valores individuais.")
st.subheader("Filtro por Ano e Mês")

anos_disponiveis = sorted(df["Year"].unique())
meses_disponiveis = sorted(df["Month"].unique())

ano_selecionado = st.selectbox("Selecione o ano", anos_disponiveis)
mes_selecionado = st.selectbox("Selecione o mês", meses_disponiveis)

filtro = df[(df["Year"] == ano_selecionado) & (df["Month"] == mes_selecionado)]

total_eventos = len(filtro)
total_tsunamis = filtro["tsunami"].sum()
magnitude_media = filtro["magnitude"].mean()

st.write(f"Total de eventos: **{total_eventos}**")
st.write(f"Tsunamis confirmados: **{total_tsunamis}**")

if total_tsunamis > 0:
    st.error("Tsunamis registrados neste período")
else:
    st.success("Nenhum tsunami registrado neste período")



#sidebar download da base
download_dataframe(df)