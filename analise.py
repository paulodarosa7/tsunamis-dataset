import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.title("🌊 Earthquake & Tsunami Risk - Global Dataset")

df = pd.read_csv('/mnt/c/Users/da rosa/Documents/ciencia de dados/dataset/tsunamis_database.csv')

st.subheader("Primeiras linhas do dataset")
st.dataframe(df.head())

st.write("Dimensões da base:", df.shape)

st.subheader("Informações gerais")
col = st.selectbox("escolha uma coluna para ver seus dados: ", df.columns)
st.write(df[col].head())


features = ["magnitude", "depth", "latitude", "longitude"]
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
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.write(f"MAE: {mae:.2f}")
st.write(f"RMSE: {rmse:.2f}")
st.write(f"R²: {r2:.2f}")

st.subheader("Chances de tsunami")

magn = st.slider("Magnitude", df["magnitude"].min(), df["magnitude"].max())
prof = st.slider("Profundidade (km)", df["depth"].min(), df["depth"].max())

lat = st.slider("Latitude", float(df["latitude"].min()), float(df["latitude"].max()))
lon = st.slider("Longitude", float(df["longitude"].min()), float(df["longitude"].max()))

# Criar dataframe para prever
entrada = pd.DataFrame([[magn, prof, lat, lon]], columns=features)

resultado = model.predict(entrada)[0]
percent = model.predict_proba(entrada)[0][1]

if resultado == 1:
    st.error(f"chance alta de tsunami: {percent:.2f}%")
else:
    st.success(f"Chance baixa de tsunami: {percent:.2f}%")
    
    
import pydeck as pdk

st.subheader("🗺 Mapa global dos terremotos")

# Criar base pro mapa
df_map = df.copy()
df_map["tsunami_color"] = df_map["tsunami"].apply(lambda x: [255, 0, 0] if x == 1 else [0, 150, 255])

# Configurar view inicial
view_state = pdk.ViewState(
    latitude=df_map["latitude"].mean(),
    longitude=df_map["longitude"].mean(),
    zoom=1,
    pitch=0
)

# Configurar camadas
layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_map,
    get_position=["longitude", "latitude"],
    get_color="tsunami_color",
    get_radius=30000,  # quanto maior, mais visível
    pickable=True
)

# Montar o mapa
r = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"text": "Magnitude: {magnitude}\nProfundidade: {depth} km\nTsunami: {tsunami}"}
)

st.pydeck_chart(r)
