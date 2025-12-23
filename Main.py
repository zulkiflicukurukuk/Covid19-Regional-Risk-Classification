import streamlit as st
import base64
import os

st.set_page_config(
    page_title="COVID19-REGIONAL-RISK-CLASSIFICATION",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_img_as_base64(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;900&family=Roboto+Condensed:wght@700&display=swap');
    
    .stApp { background-color: #050505 !important; }
    [data-testid="stSidebar"] { background-color: #050505 !important; border-right: 1px solid #1A1A1A; }
    html, body, [class*="css"] { font-family: 'Montserrat', sans-serif; color: #E0E0E0; }

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
        font-family: 'Roboto Condensed', sans-serif; font-size: 36px; font-weight: 800; color: #fff;
        margin-bottom: 20px; text-transform: uppercase; border-left: 6px solid #00E5FF; padding-left: 20px;
        background: linear-gradient(90deg, rgba(0,229,255,0.1), transparent);
        text-shadow: 0 0 15px rgba(0, 229, 255, 0.4);
    }

    [data-testid="column"] {
        background-color: #0A0A0A; 
        border-radius: 12px 12px 0 0; 
        border: 1px solid #222;
        border-bottom: none; 
        height: 100% !important;
        display: flex;
        flex-direction: column;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    [data-testid="column"]:hover {
        transform: translateY(-5px);
        border-color: #00E5FF;
        box-shadow: 0 10px 30px rgba(0, 229, 255, 0.15);
    }

    .card-img {
        width: 100%; height: 200px !important; object-fit: cover; 
        border-bottom: 1px solid #222; opacity: 0.9;
    }

    .card-content {
        padding: 20px;
        flex-grow: 1;
        display: flex;
        flex-direction: column;
        min-height: 280px; 
        justify-content: flex-start;
    }

    .model-tag {
        font-size: 11px; font-weight: 800; color: #000; background-color: #00E5FF;
        padding: 3px 8px; border-radius: 4px; display: inline-block; margin-bottom: 10px; width: fit-content;
    }

    .dataset-info {
        font-size: 12px; color: #888; margin-top: auto; 
        line-height: 1.6; border-top: 1px solid #333; padding-top: 15px;
    }

    .stButton {
        margin-top: -15px !important; 
    }
    
    .stButton > button {
        width: 100%;
        background: linear-gradient(45deg, #000, #111);
        color: #00E5FF;
        border: 1px solid #00E5FF;
        border-top: 1px solid #333; 
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 2px;
        padding: 15px 0;
        border-radius: 0 0 12px 12px; 
        height: 55px !important;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: #00E5FF;
        color: #000;
        box-shadow: 0 0 25px rgba(0, 229, 255, 0.6);
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    col_kiri, col_tengah, col_kanan = st.columns([1, 2, 1])
    with col_tengah:
        st.image("Assets/LOGO.jpeg", width=80) 
    
    st.markdown('<p class="sidebar-brand-text" style="text-align: center;">COVID-PREDICTION</p>', unsafe_allow_html=True)
    st.info("Ujian Akhir Praktikum\nMachine Learning")

st.markdown('<div class="section-title">Select Model Prediction</div>', unsafe_allow_html=True)
st.markdown('<p style="color: #888; margin-bottom: 30px; margin-left: 20px;">Pilih model untuk melakukan prediksi risiko wilayah.</p>', unsafe_allow_html=True)

img_mlp = get_img_as_base64("Assets/MLP.jpeg")
img_ae = get_img_as_base64("Assets/AutoEncoder.jpeg")
img_tab = get_img_as_base64("Assets/TabNet.jpeg")

holder = "https://via.placeholder.com/800x400/111/00E5FF?text=No+Image"
src_mlp = f"data:image/jpeg;base64,{img_mlp}" if img_mlp else holder
src_ae = f"data:image/jpeg;base64,{img_ae}" if img_ae else holder
src_tab = f"data:image/jpeg;base64,{img_tab}" if img_tab else holder

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <img src="{src_mlp}" class="card-img">
    <div class="card-content">
        <div>
            <span class="model-tag">ACCURACY: 82%</span>
            <h3 style="margin: 5px 0 10px 0; font-family:'Roboto Condensed'; color: #fff; font-size: 24px;">MLP BASELINE</h3>
            <p style="font-size: 13px; color: #ccc; margin-bottom: 15px; font-weight: 600;">
                Feedforward Neural Network
            </p>
        </div>
        <div class="dataset-info">
            Model Non-Pretrained sederhana namun efektif. Terbukti memiliki akurasi tertinggi untuk data tabular ini.
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("LAUNCH MLP", key="btn_mlp"):
        st.switch_page("pages/1_MLP.py")

with col2:
    st.markdown(f"""
    <img src="{src_ae}" class="card-img">
    <div class="card-content">
        <div>
            <span class="model-tag">ACCURACY: 76%</span>
            <h3 style="margin: 5px 0 10px 0; font-family:'Roboto Condensed'; color: #fff; font-size: 24px;">AUTOENCODER</h3>
            <p style="font-size: 13px; color: #ccc; margin-bottom: 15px; font-weight: 600;">
                Transfer Learning (Embedding)
            </p>
        </div>
        <div class="dataset-info">
            Menggunakan teknik Unsupervised Pretraining untuk kompresi fitur, kemudian di-finetune untuk klasifikasi.
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("LAUNCH AUTOENCODER", key="btn_ae"):
        st.switch_page("pages/2_AutoEncoder.py")

with col3:
    st.markdown(f"""
    <img src="{src_tab}" class="card-img">
    <div class="card-content">
        <div>
            <span class="model-tag">ACCURACY: 70%</span>
            <h3 style="margin: 5px 0 10px 0; font-family:'Roboto Condensed'; color: #fff; font-size: 24px;">TABNET</h3>
            <p style="font-size: 13px; color: #ccc; margin-bottom: 15px; font-weight: 600;">
                SOTA Attention Architecture
            </p>
        </div>
        <div class="dataset-info">
            Model canggih dari Google Cloud AI. Menggunakan mekanisme Attention untuk memproses data tabular.
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("LAUNCH TABNET", key="btn_tabnet"):
        st.switch_page("pages/3_TabNet.py")