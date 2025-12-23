import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from pytorch_tabnet.tab_model import TabNetClassifier

st.set_page_config(page_title="Forecasting", page_icon="🦠", layout="wide")
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;900&family=Roboto+Condensed:wght@700&display=swap');

    [data-testid="stSidebar"] { background-color: #050505; border-right: 1px solid #222; }
    
    [data-testid="stSidebarNav"] a {
        font-family: 'Montserrat', sans-serif !important; font-weight: 600 !important; font-size: 13px !important;
        text-transform: uppercase !important; color: #888888 !important; margin-bottom: 4px !important;
        padding: 8px 12px !important; border-radius: 6px !important; transition: all 0.3s ease !important;
    }
    [data-testid="stSidebarNav"] a:hover { color: #00E5FF !important; background-color: #111111 !important; }
    [data-testid="stSidebarNav"] a[aria-current="page"] {
        color: #000000 !important; background-color: #00E5FF !important; font-weight: 800 !important;
        box-shadow: 0 0 12px rgba(0, 229, 255, 0.6) !important;
    }
    
    .sidebar-brand-text { font-family: 'Montserrat', sans-serif; font-weight: 900; font-size: 20px; color: #00E5FF; margin-top: 12px; letter-spacing: 1px; }
    .section-title {
        font-family: 'Roboto Condensed', sans-serif; font-size: 36px; font-weight: 700; color: #fff;
        margin-bottom: 5px; text-transform: uppercase; border-left: 5px solid #00E5FF; padding-left: 15px;
    }
            
    .stApp {
        background-color: #050505 !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #050505 !important;
        border-right: 1px solid #1A1A1A; 
    }

    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif;
        color: #E0E0E0; 
    }

    .section-title {
        font-family: 'Roboto Condensed', sans-serif;
        font-size: 34px;
        font-weight: 800;
        color: #fff;
        text-transform: uppercase;
        margin-bottom: 20px;
        text-shadow: 0 0 15px rgba(0, 229, 255, 0.4);
        border-left: 6px solid #00E5FF;
        padding-left: 20px;
        background: linear-gradient(90deg, rgba(0,229,255,0.1), transparent); 
    }

    .stNumberInput input {
        background-color: #0F0F0F !important; 
        color: #00E5FF !important; 
        border: 1px solid #333 !important;
        border-radius: 6px;
        font-weight: bold;
    }
    
    .stNumberInput input:focus {
        border-color: #00E5FF !important;
        box-shadow: 0 0 10px rgba(0, 229, 255, 0.3) !important;
        background-color: #151515 !important;
    }

    div[data-testid="stNumberInput"] label {
        color: #AAAAAA !important;
        font-size: 13px !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    button[kind="secondary"] {
        background-color: #1A1A1A !important;
        color: #fff !important;
        border: 1px solid #333 !important;
    }

    [data-testid="stForm"] {
        background-color: #0A0A0A;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #222;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(45deg, #000, #111);
        color: #00E5FF;
        border: 1px solid #00E5FF;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 2px;
        padding: 15px 0;
        border-radius: 8px;
        transition: all 0.3s ease;
        margin-top: 10px;
    }
    
    .stButton > button:hover {
        background: #00E5FF;
        color: #000;
        box-shadow: 0 0 25px rgba(0, 229, 255, 0.6);
        border-color: #00E5FF;
    }

    div[data-testid="metric-container"] {
        background-color: #0F0F0F;
        border: 1px solid #222;
        padding: 15px;
        border-radius: 10px;
        transition: 0.3s;
    }
    div[data-testid="metric-container"]:hover {
        border-color: #00E5FF;
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.1);
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #00E5FF !important;
        text-shadow: 0 0 10px rgba(0, 229, 255, 0.3);
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    col_kiri, col_tengah, col_kanan = st.columns([1, 2, 1])
    with col_tengah:
        st.image("Assets/LOGO.jpeg", width=80) 
    
    st.markdown('<p class="sidebar-brand-text" style="text-align: center;">COVID-PREDICTION</p>', unsafe_allow_html=True)
    st.info("Ujian Akhir Praktikum\nMachine Learning")

@st.cache_resource
def load_scaler():
    return joblib.load('Models/scaler_covid.pkl')

scaler = load_scaler()

st.markdown('<div class="section-title">FORECASTING (BATCH)</div>', unsafe_allow_html=True)
st.markdown("Unggah data CSV untuk memprediksi risiko wilayah dalam jumlah besar sekaligus.")

c1, c2 = st.columns([2, 1])

with c2:
    st.markdown("### KONFIGURASI")
    model_option = st.selectbox(
        "Pilih Model Eksekutor:",
        ("MLP (Recommended - 82%)", "AutoEncoder (76%)", "TabNet (70%)")
    )
    st.info("Pilih model yang akan digunakan untuk memproses seluruh data dalam file CSV.")

with c1:
    st.markdown("### UPLOAD DATA")
    uploaded_file = st.file_uploader("Upload File CSV (Format Data Test)", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"File berhasil dimuat! Total Data: {len(df)} baris.")
    
    with st.expander("Lihat Preview Data Asli"):
        st.dataframe(df.head())

    st.markdown("---")
    if st.button(f"JALANKAN PREDIKSI"):
        
        X_batch = df.drop(columns=['Actual_Risk', 'Target_Risk', 'Risk_Level'], errors='ignore')
        
        try:
            X_batch_scaled = scaler.transform(X_batch)
            preds = []
            
            if "MLP" in model_option:
                model = tf.keras.models.load_model('Models/Model_1.h5')
                raw_preds = model.predict(X_batch_scaled)
                preds = np.argmax(raw_preds, axis=1)
                
            elif "AutoEncoder" in model_option:
                model = tf.keras.models.load_model('Models/Model_2.h5')
                raw_preds = model.predict(X_batch_scaled)
                preds = np.argmax(raw_preds, axis=1)
                
            elif "TabNet" in model_option:
                model = TabNetClassifier()
                model.load_model('Models/Model_3.zip')
                preds = model.predict(X_batch_scaled)
            
            label_map = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
            df['Prediction'] = [label_map[p] for p in preds]
        
            st.success("Prediksi Selesai!")
            st.dataframe(df)
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.markdown("#### Ringkasan Statistik")
                st.write(df['Prediction'].value_counts())
            with col_res2:
                st.markdown("#### Visualisasi Sebaran")
                st.bar_chart(df['Prediction'].value_counts(), color="#00E5FF")

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="DOWNLOAD HASIL (CSV)",
                data=csv,
                file_name=f'Result_{model_option.split(" ")[0]}.csv',
                mime='text/csv',
            )

        except Exception as e:
            st.error(f"Error Format Data: {e}")