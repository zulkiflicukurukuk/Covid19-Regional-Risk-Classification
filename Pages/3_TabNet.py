import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pytorch_tabnet.tab_model import TabNetClassifier

st.set_page_config(page_title="TabNet Simulation", page_icon="🦠", layout="wide")

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
def load_assets():
    scaler = joblib.load('Models/scaler_covid.pkl')
    model = TabNetClassifier()
    model.load_model('Models/Model_3.zip')
    return scaler, model

try:
    scaler, model = load_assets()
except:
    st.error("Gagal memuat TabNet. Pastikan Model_3.zip ada.")
    st.stop()

st.markdown('<div class="section-title">TABNET SOTA MODEL</div>', unsafe_allow_html=True)

c_img1, c_img2, c_img3 = st.columns([1, 2, 1]) 
with c_img2:
    st.image("Assets/TabNet.jpeg", 
             caption="Arsitektur TabNet (Sequential Attention)", use_container_width=True)

st.markdown("<h3 style='color: #00E5FF; margin-bottom: 10px;'>INPUT PARAMETER</h3>", unsafe_allow_html=True)

with st.form("tabnet_form"):
    c1, c2, c3 = st.columns(3)
    with c1: active = st.number_input("Total Active Cases", value=500)
    with c2: cases_pm = st.number_input("Cases per Million", value=15000.0)
    with c3: deaths_pm = st.number_input("Deaths per Million", value=300.0)
    
    c4, c5, c6 = st.columns(3)
    with c4: density = st.number_input("Population Density", value=150.0)
    with c5: growth = st.number_input("Growth Factor", value=1.0)
    with c6: fatality = st.number_input("Fatality Rate (%)", value=2.5)

    c7, c8, c9 = st.columns(3)
    with c7: recovered = st.number_input("Recovered Rate (%)", value=95.0)
    with c8: cities = st.number_input("Total Cities", value=5)
    with c9: regencies = st.number_input("Total Regencies", value=10)

    c10, c11 = st.columns(2)
    with c10: long = st.number_input("Longitude", value=106.8)
    with c11: lat = st.number_input("Latitude", value=-6.2)
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict = st.form_submit_button("JALANKAN PREDIKSI")

if predict:
    input_data = np.array([[active, cases_pm, deaths_pm, density, growth, fatality, recovered, cities, regencies, long, lat]])
    scaled_data = scaler.transform(input_data)
    
    prediction_proba = model.predict_proba(scaled_data)
    probs = prediction_proba[0]
    class_idx = np.argmax(probs)
    confidence = np.max(probs) * 100
    
    labels = ["Low Risk", "Medium Risk", "High Risk"]
    result_text = labels[class_idx]
    
    st.markdown("---")
    st.markdown(f"<h2 style='text-align: center; color: #fff;'>HASIL ANALISIS</h2>", unsafe_allow_html=True)
    
    r1, r2, r3 = st.columns(3)
    with r1: st.metric("Prediksi Risiko", result_text)
    with r2: st.metric("Confidence Score", f"{confidence:.1f}%")
    with r3: st.metric("Status Wilayah", "AMAN" if class_idx==0 else "BAHAYA")

    g1, g2 = st.columns([1.5, 1])
    with g1:
        st.markdown("#### Probabilitas Model")
        prob_df = pd.DataFrame(probs, index=labels, columns=["Probability"])
        st.bar_chart(prob_df, color="#00E5FF")
    with g2:
        st.markdown("#### Analisis Logis")
        reasons = []
        if active > 1000: reasons.append(f"**Kasus Aktif Tinggi ({active})**")
        if density > 1000: reasons.append(f"**Wilayah Padat ({density}/km²)**")
        if growth > 1.0: reasons.append(f"**Tren Naik (Growth {growth})**")
        if reasons:
            for r in reasons: st.warning(r)
        else:
            st.success("Parameter wilayah stabil.")
        st.info(f"TabNet mendeteksi pola **{result_text}** via Attention Mechanism.")