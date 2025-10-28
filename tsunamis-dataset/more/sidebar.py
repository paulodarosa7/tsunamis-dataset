import streamlit as st
import pandas as pd

df = pd.read_csv('/home/operador/Documentos/analise de Dados/tsunamis-dataset/dataset/tsunamis_database.csv')

@st.cache_data
def convert_for_download(df):
    return df.to_csv(index=False).encode("utf-8")

# Sidebar de download
def download_dataframe(df):
    st.sidebar.write("Download Dataset")
    csv = convert_for_download(df)
    st.sidebar.download_button(
        label="CSV",
        data=csv,
        file_name="tsunamis_database.csv",
        mime="text/csv",
    )