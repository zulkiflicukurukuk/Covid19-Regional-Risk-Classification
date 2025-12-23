import streamlit as st
import base64
import os

st.set_page_config(page_title="Tools & Evaluation", page_icon="🦠", layout="wide")

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
    .stApp { background-color: #050505 !important; }
    [data-testid="stSidebar"] { background-color: #050505 !important; border-right: 1px solid #1A1A1A; }
    html, body, [class*="css"] { font-family: 'Montserrat', sans-serif; color: #E0E0E0; }
    
    .section-title {
        font-family: 'Roboto Condensed', sans-serif; font-size: 34px; font-weight: 800; color: #fff;
        text-transform: uppercase; margin-bottom: 20px; text-shadow: 0 0 15px rgba(0, 229, 255, 0.4);
        border-left: 6px solid #00E5FF; padding-left: 20px;
        background: linear-gradient(90deg, rgba(0,229,255,0.1), transparent);
    }

    .card-tool {
        background-color: #0A0A0A;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #222;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        transition: 0.3s;
        text-align: left; 
    }
    .card-tool:hover {
        border-color: #00E5FF;
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.15);
    }
    
    .card-tool h3 {
        color: #00E5FF !important;
        font-family: 'Roboto Condensed' !important;
        margin-top: 0;
        border-bottom: 1px solid #333;
        padding-bottom: 15px;
        margin-bottom: 15px;
        font-size: 24px;
    }

    .card-tool p {
        line-height: 1.8; 
        color: #CCCCCC;
        margin-bottom: 15px;
        font-size: 14px;
        text-align: justify; 
    }

    .card-tool img {
        border-radius: 8px; border: 1px solid #333; max-width: 100%; height: auto; display: block; margin: 0 auto;
    }
    
    .highlight {
        color: #fff; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    col_kiri, col_tengah, col_kanan = st.columns([1, 2, 1])
    with col_tengah:
        st.image("Assets/LOGO.jpeg", width=80) 
    
    st.markdown('<p class="sidebar-brand-text" style="text-align: center;">COVID-PREDICTION</p>', unsafe_allow_html=True)
    st.info("Ujian Akhir Praktikum\nMachine Learning")

def get_img_as_base64(file_path):
    if not os.path.exists(file_path): return None
    with open(file_path, "rb") as f: data = f.read()
    return base64.b64encode(data).decode()

st.markdown('<div class="section-title">TOOLS & EVALUATION</div>', unsafe_allow_html=True)

tab_eval, tab_info = st.tabs(["EVALUASI PERFORMA", "INFORMASI MODEL"])

with tab_eval:
    st.markdown("Visualisasi performa ketiga model berdasarkan Data Test (Unseen Data).")
    col1, col2 = st.columns(2)
    img_cm = get_img_as_base64("Assets/confusion_matrix.png")
    img_hist = get_img_as_base64("Assets/grafik_loss_accuracy.png")

    with col1:
        if img_cm:
            st.markdown(f"""<div class="card-tool" style="text-align:center;"><h3>1. CONFUSION MATRIX</h3><img src="data:image/png;base64,{img_cm}"><p style="text-align:center; font-style:italic; margin-top:10px;">Detail kesalahan prediksi tiap kelas.</p></div>""", unsafe_allow_html=True)
    with col2:
        if img_hist:
            st.markdown(f"""<div class="card-tool" style="text-align:center;"><h3>2. LEARNING HISTORY</h3><img src="data:image/png;base64,{img_hist}"><p style="text-align:center; font-style:italic; margin-top:10px;">Grafik Loss & Accuracy selama training.</p></div>""", unsafe_allow_html=True)

    st.markdown("### RINGKASAN AKURASI")
    st.dataframe([
        {"Model": "MLP (Baseline)", "Akurasi": "82%", "Status": "Best Model 👑", "Karakter": "Efisien & Stabil"},
        {"Model": "AutoEncoder", "Akurasi": "76%", "Status": "Good", "Karakter": "Robust terhadap Noise"},
        {"Model": "TabNet", "Akurasi": "70%", "Status": "Underfitting", "Karakter": "Butuh Data Masif"},
    ], use_container_width=True)

with tab_info:
    st.markdown("""
    <div class="card-tool">
        <h3>1. MLP (Multi-Layer Perceptron)</h3>
        <p>
            MLP adalah arsitektur Deep Learning paling fundamental yang terdiri dari sekumpulan lapisan neuron yang saling terhubung penuh (Fully Connected Layers). Dalam proyek ini, MLP bertindak sebagai <span class="highlight">Baseline Model</span>. Model ini bekerja dengan cara memetakan input fitur (seperti total kasus, kepadatan penduduk) melalui lapisan tersembunyi (hidden layers) yang menggunakan fungsi aktivasi non-linear (seperti ReLU) untuk menangkap pola kompleks, sebelum akhirnya dikeluarkan melalui lapisan output Softmax untuk klasifikasi risiko.
        </p>
        <p>
            Keunggulan utama MLP pada dataset ini adalah kesederhanaannya. Karena data COVID-19 yang digunakan berbentuk tabular dengan jumlah fitur yang terbatas (11 fitur), MLP mampu mempelajari hubungan antar variabel secara efisien tanpa mengalami <i>overfitting</i> yang berlebihan. 
        </p>
        <p>
            Hasil eksperimen menunjukkan bahwa MLP mencapai akurasi tertinggi <b>(82%)</b>. Ini membuktikan hipotesis bahwa untuk dataset berukuran menengah dengan struktur fitur yang jelas, model yang lebih sederhana seringkali mengungguli model yang terlalu kompleks (SOTA) karena memiliki generalisasi yang lebih baik.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card-tool">
        <h3>2. AutoEncoder (Transfer Learning)</h3>
        <p>
            AutoEncoder adalah jenis jaringan saraf yang awalnya digunakan untuk pembelajaran tanpa pengawasan (Unsupervised Learning). Model ini dilatih dalam dua tahap. Tahap pertama adalah <span class="highlight">Pretraining</span>, di mana model belajar untuk mengompresi data input menjadi representasi laten (bottleneck) dan kemudian merekonstruksinya kembali. Proses ini memaksa model untuk membuang "noise" dan hanya menyimpan fitur-fitur esensial dari data penyebaran COVID-19.
        </p>
        <p>
            Tahap kedua adalah <span class="highlight">Fine-Tuning</span>. Bagian Decoder dibuang, dan bagian Encoder (yang sudah pintar mengekstrak fitur) disambungkan dengan lapisan klasifikasi baru (Classifier Head). Model kemudian dilatih ulang menggunakan label risiko (Low/Medium/High).
        </p>
        <p>
            Dengan akurasi <b>76%</b>, AutoEncoder terbukti sangat efektif dalam menangani data yang memiliki variasi tinggi. Teknik ini sangat berguna ketika data label terbatas, karena model sudah "belajar" struktur data terlebih dahulu sebelum diminta untuk melakukan klasifikasi.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card-tool">
        <h3>3. TabNet (Attentive Interpretable Tabular Learning)</h3>
        <p>
            TabNet adalah arsitektur mutakhir (State-of-the-Art) yang dikembangkan oleh Google Cloud AI. Model ini mencoba menggabungkan keunggulan kemampuan interpretasi dari <i>Decision Trees</i> dengan kemampuan representasi dari <i>Deep Neural Networks</i>. Fitur utamanya adalah mekanisme <span class="highlight">Sequential Attention</span>, yang memungkinkan model untuk memilih fitur mana yang paling penting untuk diproses pada setiap langkah keputusan (Decision Step).
        </p>
        <p>
            Secara teori, TabNet mampu memberikan performa tinggi sekaligus interpretabilitas (menjelaskan alasan di balik prediksi). Model ini menggunakan <i>Soft Feature Selection</i> untuk mengabaikan kolom data yang tidak relevan secara dinamis, mirip dengan cara manusia memfilter informasi.
        </p>
        <p>
            Namun, pada eksperimen ini, TabNet memperoleh akurasi terendah <b>(70%)</b>. Hal ini kemungkinan besar disebabkan oleh ukuran dataset yang belum cukup besar untuk melatih parameter TabNet yang sangat kompleks. TabNet biasanya membutuhkan jutaan baris data untuk mencapai performa optimalnya (*data-hungry*), sehingga pada dataset berskala 30.000 baris, model cenderung mengalami kesulitan konvergensi dibandingkan MLP yang lebih ringan.
        </p>
    </div>
    """, unsafe_allow_html=True)