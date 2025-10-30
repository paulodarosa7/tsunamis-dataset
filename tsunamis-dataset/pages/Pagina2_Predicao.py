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


st.title("Pagina 2 - Prevendo possivel tsunami a partir de dados escolhidos pelo o usuário")

features = ["magnitude", "depth", "latitude", "longitude", "nst", "dmin"]
X = df[features]
y = df["tsunami"]

# Separar em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# modelo random forest
# Treinar o modelo
model = RandomForestClassifier(random_state=42,max_depth=12, min_samples_leaf=9)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# avalia o modelo
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

st.write("### Avaliação do Modelo")
st.write(f"Acurácia: {accuracy:.2f}")
st.write(f"Precisão: {precision:.2f}")
st.write(f"Recall: {recall:.2f}")

st.subheader("Chances de tsunami, apartir de dados do usuario")

magn = st.slider("Magnitude", df["magnitude"].min(), df["magnitude"].max())
prof = st.slider("Profundidade (km)", df["depth"].min(), df["depth"].max())

lat = st.slider("Latitude", float(df["latitude"].min()), float(df["latitude"].max()))
lon = st.slider("Longitude", float(df["longitude"].min()), float(df["longitude"].max()))

nst = st.slider("Número de Estações: ", df["nst"].min(), df["nst"].max())
dmin = st.slider("Distância à Estação Sísmica Mais Próxima", float(df["dmin"].min()), float(df["dmin"].max()))


# Criar dataframe para prever
entrada = pd.DataFrame([[magn, prof, lat, lon, nst, dmin]], columns=features)

resultado = model.predict(entrada)[0]
percent = model.predict_proba(entrada)[0][1]

if resultado == 1:
    st.error(f"chance alta de tsunami: {percent:.2f}%")
else:
    st.success(f"Chance baixa de tsunami: {percent:.2f}%")


#sidebar download da base
download_dataframe(df)

st.link_button("Link Dataset - Kaggle", "https://www.kaggle.com/datasets/ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset")
