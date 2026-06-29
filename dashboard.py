import streamlit as st
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(
    page_title="Heart Disease Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ATTRIBUTES = pd.DataFrame({
    "Atribut": ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope", "HeartDisease"],
    "Tipe": ["Numerik", "Kategorik", "Kategorik", "Numerik", "Numerik", "Biner", "Kategorik", "Numerik", "Biner", "Numerik", "Kategorik", "Target"],
    "Keterangan": [
        "Usia pasien dalam tahun, rentang 28 sampai 77 tahun.",
        "Jenis kelamin pasien: M untuk laki-laki dan F untuk perempuan.",
        "Tipe nyeri dada: TA, ATA, NAP, dan ASY/asymptomatic.",
        "Tekanan darah istirahat dalam mmHg.",
        "Kadar kolesterol serum dalam mg/dL.",
        "Gula darah puasa: 1 jika > 120 mg/dL, 0 jika tidak.",
        "Hasil ECG istirahat: Normal, ST, atau LVH.",
        "Detak jantung maksimum yang dicapai pasien.",
        "Angina akibat olahraga: Y atau N.",
        "Depresi segmen ST dari hasil exercise test.",
        "Kemiringan segmen ST: Up, Flat, atau Down.",
        "Status penyakit jantung: 1 sakit jantung, 0 normal/sehat.",
    ],
})

PREPROCESSING = pd.DataFrame({
    "Tahap": ["Data checking", "Anomali Cholesterol", "Anomali RestingBP", "Encoding", "Scaling", "Train-test split"],
    "Penjelasan": [
        "Tidak ditemukan missing value eksplisit dan tidak ditemukan baris duplikat.",
        "172 baris bernilai 0 dianggap missing value tersamar dan diganti median valid 237 mg/dL.",
        "1 baris bernilai 0 dianggap tidak valid karena tekanan darah pasien hidup tidak mungkin nol.",
        "Atribut kategorikal diubah menjadi angka memakai Label Encoding.",
        "Fitur numerik distandarisasi memakai StandardScaler agar skala fitur seimbang.",
        "Data dibagi 80:20 menjadi 734 data latih dan 184 data uji dengan stratify target.",
    ],
})

MODEL_METRICS = pd.DataFrame({
    "Metrik": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC AUC"],
    "Logistic Regression": ["84,78%", "84,91%", "88,24%", "86,54%", "0,8989"],
    "Gaussian Naive Bayes": ["87,50%", "88,35%", "89,22%", "88,78%", "0,9225"],
    "Model Terbaik": ["Naive Bayes", "Naive Bayes", "Naive Bayes", "Naive Bayes", "Naive Bayes"],
})

SAMPLE_PATIENTS = {
    "Contoh Risiko Rendah": {"Age": 42, "Sex": "F", "ChestPainType": "ATA", "RestingBP": 120, "Cholesterol": 210, "FastingBS": 0, "RestingECG": "Normal", "MaxHR": 172, "ExerciseAngina": "N", "Oldpeak": 0.2, "ST_Slope": "Up"},
    "Contoh Risiko Sedang": {"Age": 54, "Sex": "M", "ChestPainType": "NAP", "RestingBP": 135, "Cholesterol": 245, "FastingBS": 0, "RestingECG": "ST", "MaxHR": 142, "ExerciseAngina": "N", "Oldpeak": 1.1, "ST_Slope": "Flat"},
    "Contoh Risiko Tinggi": {"Age": 61, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 150, "Cholesterol": 280, "FastingBS": 1, "RestingECG": "LVH", "MaxHR": 115, "ExerciseAngina": "Y", "Oldpeak": 2.6, "ST_Slope": "Flat"},
    "Contoh Silent Risk ASY": {"Age": 57, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 140, "Cholesterol": 237, "FastingBS": 0, "RestingECG": "Normal", "MaxHR": 128, "ExerciseAngina": "Y", "Oldpeak": 1.8, "ST_Slope": "Down"},
}


@st.cache_resource
def train_prediction_model():
    df = pd.read_csv("heart.csv")
    median_cholesterol = df.loc[df["Cholesterol"] != 0, "Cholesterol"].median()
    median_restingbp = df.loc[df["RestingBP"] != 0, "RestingBP"].median()
    df["Cholesterol"] = df["Cholesterol"].replace(0, median_cholesterol)
    df["RestingBP"] = df["RestingBP"].replace(0, median_restingbp)

    categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
    encoders = {}
    for col in categorical_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoders[col] = encoder

    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = GaussianNB()
    model.fit(X_scaled, y)
    return model, scaler, encoders, list(X.columns), median_cholesterol, median_restingbp


def predict_patient(patient):
    model, scaler, encoders, columns, median_cholesterol, median_restingbp = train_prediction_model()
    patient = patient.copy()
    if patient["Cholesterol"] == 0:
        patient["Cholesterol"] = median_cholesterol
    if patient["RestingBP"] == 0:
        patient["RestingBP"] = median_restingbp
    for col, encoder in encoders.items():
        patient[col] = encoder.transform([patient[col]])[0]
    input_df = pd.DataFrame([patient], columns=columns)
    input_scaled = scaler.transform(input_df)
    prediction = int(model.predict(input_scaled)[0])
    probability = float(model.predict_proba(input_scaled)[0][1])
    return prediction, probability


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@700;800&family=Manrope:wght@400;600;700;800&display=swap');
    :root{--cream:#FFF8EA;--paper:#FFFDF7;--ink:#17362F;--leaf:#315F38;--mint:#EAF4E7;--gold:#F2C14E;--coral:#E86F51;--line:rgba(49,95,56,.16)}
    html,body,[class*="css"]{font-family:'Manrope',sans-serif}.stApp{color:var(--ink);background:radial-gradient(circle at 8% 12%,rgba(242,193,78,.24),transparent 28%),radial-gradient(circle at 92% 18%,rgba(157,192,139,.38),transparent 30%),linear-gradient(135deg,#FFF8EA 0%,#F6F1DE 48%,#E9F3E2 100%)}
    .block-container{max-width:1220px;padding-top:2.2rem;padding-bottom:3rem}[data-testid="stSidebar"]{background:linear-gradient(180deg,#EAF4E7 0%,#F9F1D7 100%);border-right:1px solid var(--line)}[data-testid="stSidebar"] *{color:var(--leaf)}
    .hero{position:relative;overflow:hidden;padding:44px 48px;border-radius:34px;background:linear-gradient(115deg,rgba(255,253,247,.96),rgba(234,244,231,.88)),repeating-linear-gradient(45deg,rgba(49,95,56,.06) 0 1px,transparent 1px 14px);border:1px solid var(--line);box-shadow:0 24px 80px rgba(23,54,47,.12)}.hero:after{content:"";position:absolute;width:280px;height:280px;right:-80px;top:-90px;border-radius:999px;background:linear-gradient(135deg,rgba(242,193,78,.65),rgba(232,111,81,.25))}.kicker{display:inline-flex;padding:8px 14px;border-radius:999px;background:rgba(49,95,56,.10);color:var(--leaf);font-weight:800;letter-spacing:.08em;text-transform:uppercase;font-size:12px}.hero h1{position:relative;z-index:1;margin:18px 0 12px;max-width:920px;color:var(--leaf);font-family:'Fraunces',serif;font-size:clamp(42px,6vw,76px);line-height:.96;letter-spacing:-.04em}.hero p{position:relative;z-index:1;max-width:900px;color:#304F3F;font-size:17px;line-height:1.8}
    .badge-row{display:flex;flex-wrap:wrap;gap:10px;margin-top:18px}.badge{padding:9px 13px;border-radius:999px;background:rgba(242,193,78,.24);color:var(--leaf);font-weight:800;font-size:13px}.metric-card,.glass-panel{background:rgba(255,253,247,.86);border:1px solid var(--line);box-shadow:0 18px 45px rgba(23,54,47,.09)}.metric-card{min-height:155px;padding:24px;border-radius:26px;transition:.18s}.metric-card:hover{transform:translateY(-4px);box-shadow:0 24px 55px rgba(23,54,47,.14)}.metric-icon{width:44px;height:44px;display:grid;place-items:center;border-radius:16px;background:var(--mint);font-size:23px;margin-bottom:16px}.metric-title{color:#5D715F;font-size:14px;font-weight:800;margin-bottom:6px}.big-number{color:var(--leaf);font-family:'Fraunces',serif;font-size:42px;line-height:1;font-weight:800;margin-bottom:8px}.metric-desc{color:#55705C;font-size:14px;font-weight:600}.section-title{margin:14px 0 10px;color:var(--leaf);font-family:'Fraunces',serif;font-size:34px;letter-spacing:-.03em}.glass-panel{padding:24px;border-radius:28px}.text-card{padding:24px;border-radius:24px;background:rgba(255,253,247,.82);border:1px solid var(--line);line-height:1.75}.insight-box{background:linear-gradient(135deg,#315F38,#17362F);color:#FFF8EA;padding:24px;border-radius:24px;line-height:1.75;box-shadow:0 22px 48px rgba(23,54,47,.20)}.matrix-box{background:#FFFDF7;color:var(--ink);padding:22px;border-radius:22px;border:1px solid var(--line)}
    [data-testid="stMetric"]{background:rgba(255,253,247,.82);border:1px solid var(--line);border-radius:20px;padding:16px 18px;box-shadow:0 12px 30px rgba(23,54,47,.07)}[data-testid="stMetricValue"]{color:var(--leaf);font-family:'Fraunces',serif}[data-testid="stDataFrame"]{border-radius:22px;overflow:hidden;border:1px solid var(--line)}hr{border-color:rgba(49,95,56,.14)!important;margin:2rem 0!important}@media(max-width:768px){.hero{padding:30px 24px;border-radius:24px}.hero h1{font-size:40px}.metric-card{min-height:auto}}
    </style>
    """, unsafe_allow_html=True)


def hero(title, subtitle, badges):
    badge_html = "".join(f"<span class='badge'>{badge}</span>" for badge in badges)
    st.markdown(f"""
    <div class="hero">
      <div class="kicker">Laporan Data Mining 2026</div>
      <h1>{title}</h1>
      <p>{subtitle}</p>
      <div class="badge-row">{badge_html}</div>
    </div>
    """, unsafe_allow_html=True)


def page_overview():
    hero(
        "Heart Disease Prediction Dashboard",
        "<b>Sistem visualisasi hasil penelitian penyakit jantung.</b> Dashboard ini dibuat terpisah per halaman supaya rapi: ringkasan project, data understanding, evaluasi model, dan kalkulator prediksi pasien baru.",
        ["918 Data Pasien", "11 Fitur + 1 Target", "Balanced Dataset", "ROC AUC NB 0.9225"],
    )
    st.write("")
    cols = st.columns(3)
    cards = [("👥", "Total Data", "918", "Jumlah seluruh data pasien"), ("🫀", "Sakit Jantung", "508", "55,3% dari total data"), ("✅", "Tidak Sakit", "410", "44,7% dari total data")]
    for col, (icon, title, value, desc) in zip(cols, cards):
        with col:
            st.markdown(f"<div class='metric-card'><div class='metric-icon'>{icon}</div><div class='metric-title'>{title}</div><div class='big-number'>{value}</div><div class='metric-desc'>{desc}</div></div>", unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-title">Pendahuluan Penelitian</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="text-card">
    Penyakit jantung atau cardiovascular disease menjadi salah satu penyebab kematian tertinggi di dunia. WHO mencatat penyakit kardiovaskular menyebabkan sekitar 17,9 juta kematian setiap tahun. Di Indonesia, prevalensi penyakit jantung juga meningkat dan menjadi salah satu penyakit dengan biaya pelayanan kesehatan terbesar.<br><br>
    Karena itu, penelitian ini memanfaatkan data mining dan machine learning untuk membantu proses deteksi dini. Atribut klinis seperti usia, jenis kelamin, tekanan darah, kolesterol, gula darah, ECG, detak jantung maksimum, dan hasil exercise test digunakan untuk membaca pola risiko penyakit jantung.
    </div>
    """, unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Rumusan Masalah</div>', unsafe_allow_html=True)
        st.markdown("""<div class="text-card">1. Bagaimana karakteristik data klinis pasien?<br>2. Atribut apa yang berhubungan kuat dengan penyakit jantung?<br>3. Bagaimana membangun model Logistic Regression?<br>4. Bagaimana performa Naive Bayes dibanding model lain?</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-title">Tujuan</div>', unsafe_allow_html=True)
        st.markdown("""<div class="text-card">Mengidentifikasi pola data, mengelompokkan risiko pasien, membangun model klasifikasi Logistic Regression dan Gaussian Naive Bayes, lalu membandingkan performa keduanya berdasarkan metrik evaluasi.</div>""", unsafe_allow_html=True)


def page_data():
    hero("Data Understanding", "<b>Kenali dataset sebelum modeling.</b> Halaman ini menjelaskan sumber data, atribut, kualitas data, dan insight EDA dari laporan.", ["Kaggle fedesoriano", "918 Baris", "12 Kolom", "No Duplicate"])
    st.markdown('<div class="section-title">Deskripsi Dataset</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="text-card">
    Dataset yang digunakan adalah <b>Heart Failure Prediction Dataset</b> dari Kaggle oleh fedesoriano. Dataset ini merupakan gabungan lima dataset penyakit jantung populer: Cleveland, Hungarian, Switzerland, Long Beach VA, dan Stalog Heart. Data terdiri dari 918 baris, 12 kolom, 11 atribut fitur, dan 1 atribut target HeartDisease.
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(ATTRIBUTES, use_container_width=True, hide_index=True)
    st.markdown('<div class="section-title">Insight Awal Data</div>', unsafe_allow_html=True)
    i1, i2, i3 = st.columns(3)
    i1.metric("Rata-rata Usia", "53.5 tahun", "28-77 tahun")
    i2.metric("Cholesterol 0", "172 baris", "anomali tersamar")
    i3.metric("Laki-laki", "79.0%", "dominan di dataset")
    st.markdown("""
    <div class="text-card">
    Hasil EDA menunjukkan pasien berusia di atas 50 tahun cenderung memiliki risiko lebih tinggi. Tipe nyeri dada ASY atau asymptomatic memiliki proporsi sakit jantung tertinggi, yaitu 79,0%. Atribut Oldpeak, MaxHR, ST_Slope, ChestPainType, dan ExerciseAngina menjadi fitur yang paling informatif terhadap status penyakit jantung.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="section-title">Preprocessing Data</div>', unsafe_allow_html=True)
    st.dataframe(PREPROCESSING, use_container_width=True, hide_index=True)


def page_model():
    hero("Model Evaluation", "<b>Perbandingan model klasifikasi.</b> Halaman ini menampilkan evaluasi Logistic Regression dan Gaussian Naive Bayes berdasarkan metrik utama.", ["Accuracy", "Precision", "Recall", "F1-Score", "ROC AUC"])
    st.markdown('<div class="section-title">Perbandingan Performa Model</div>', unsafe_allow_html=True)
    st.dataframe(MODEL_METRICS, use_container_width=True, hide_index=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy NB", "87.50%", "+2.72%")
    c2.metric("Precision NB", "88.35%", "+3.44%")
    c3.metric("Recall NB", "89.22%", "+0.98%")
    c4.metric("F1-Score NB", "88.78%", "+2.24%")
    c5.metric("ROC AUC NB", "0.9225", "+0.0236")
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown('<div class="section-title">Grafik Perbandingan</div>', unsafe_allow_html=True)
        st.image("output/model_comparison.png", use_container_width=True)
    with right:
        st.markdown('<div class="section-title">Insight Model</div>', unsafe_allow_html=True)
        st.markdown("""<div class="insight-box"><b>Gaussian Naive Bayes menjadi model terbaik.</b><br><br>Model ini unggul pada Accuracy, Precision, Recall, F1-Score, dan ROC AUC. Recall tinggi penting karena model lebih mampu mendeteksi pasien yang benar-benar sakit.</div>""", unsafe_allow_html=True)
        st.write("")
        st.markdown("""<div class="matrix-box"><b>Confusion Matrix Naive Bayes</b><br><br>True Negative: 70<br>False Positive: 12<br>False Negative: 11<br>True Positive: 91</div>""", unsafe_allow_html=True)
    col_cm, col_roc = st.columns(2)
    with col_cm:
        st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
        st.image("output/confusion_matrix_nb.png", use_container_width=True)
    with col_roc:
        st.markdown('<div class="section-title">ROC Curve</div>', unsafe_allow_html=True)
        st.image("output/roc_curve_nb.png", use_container_width=True)


def page_calculator():
    hero("Kalkulator Risiko Pasien", "<b>Fitur interaktif dashboard.</b> Pilih contoh pasien atau input manual data klinis baru, lalu sistem menghitung probabilitas penyakit jantung dengan Gaussian Naive Bayes.", ["Input Manual", "Contoh Data", "Naive Bayes", "Real-time Prediction"])
    selected_sample = st.selectbox("Pilih contoh data cepat", list(SAMPLE_PATIENTS.keys()))
    sample = SAMPLE_PATIENTS[selected_sample]
    with st.form("prediction_form"):
        left_input, right_input = st.columns(2)
        with left_input:
            st.markdown("### Profil Klinis")
            age = st.number_input("Usia (tahun)", 20, 90, int(sample["Age"]))
            sex = st.selectbox("Jenis Kelamin", ["M", "F"], index=["M", "F"].index(sample["Sex"]), format_func=lambda x: "Laki-laki" if x == "M" else "Perempuan")
            chest_pain = st.selectbox("Tipe Nyeri Dada", ["TA", "ATA", "NAP", "ASY"], index=["TA", "ATA", "NAP", "ASY"].index(sample["ChestPainType"]))
            resting_bp = st.number_input("Resting Blood Pressure (mmHg)", 0, 220, int(sample["RestingBP"]))
            cholesterol = st.number_input("Cholesterol (mg/dL)", 0, 650, int(sample["Cholesterol"]))
            fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1], index=[0, 1].index(sample["FastingBS"]), format_func=lambda x: "Ya" if x == 1 else "Tidak")
        with right_input:
            st.markdown("### Hasil Pemeriksaan")
            resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"], index=["Normal", "ST", "LVH"].index(sample["RestingECG"]))
            max_hr = st.number_input("Max Heart Rate", 60, 220, int(sample["MaxHR"]))
            exercise_angina = st.selectbox("Exercise Angina", ["N", "Y"], index=["N", "Y"].index(sample["ExerciseAngina"]), format_func=lambda x: "Ya" if x == "Y" else "Tidak")
            oldpeak = st.number_input("Oldpeak / ST Depression", -3.0, 7.0, float(sample["Oldpeak"]), step=0.1)
            st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"], index=["Up", "Flat", "Down"].index(sample["ST_Slope"]))
        submitted = st.form_submit_button("Hitung Prediksi Risiko", use_container_width=True)
    if submitted:
        patient = {"Age": age, "Sex": sex, "ChestPainType": chest_pain, "RestingBP": resting_bp, "Cholesterol": cholesterol, "FastingBS": fasting_bs, "RestingECG": resting_ecg, "MaxHR": max_hr, "ExerciseAngina": exercise_angina, "Oldpeak": oldpeak, "ST_Slope": st_slope}
        prediction, probability = predict_patient(patient)
        risk_percent = probability * 100
        status = "Berisiko Sakit Jantung" if prediction == 1 else "Tidak Terindikasi Sakit Jantung"
        color = "#E86F51" if prediction == 1 else "#315F38"
        st.markdown(f"""
        <div class="glass-panel" style="border-left:8px solid {color};">
            <div style="font-size:18px;font-weight:800;color:{color};">Hasil Prediksi</div>
            <div style="font-family:Fraunces,serif;font-size:42px;color:{color};margin:8px 0;">{risk_percent:.2f}%</div>
            <div style="font-size:22px;font-weight:800;">{status}</div>
            <p style="line-height:1.7;margin-top:12px;">Probabilitas menunjukkan peluang pasien masuk kelas HeartDisease = 1 berdasarkan pola pada dataset. Hasil ini adalah simulasi machine learning untuk pendukung keputusan, bukan diagnosis medis final.</p>
        </div>
        """, unsafe_allow_html=True)
        st.progress(min(max(probability, 0), 1))
        st.dataframe(pd.DataFrame([patient]), use_container_width=True, hide_index=True)


def page_conclusion():
    hero("Kesimpulan Akhir", "<b>Ringkasan hasil penelitian.</b> Gaussian Naive Bayes menjadi model utama karena unggul di seluruh metrik evaluasi.", ["NB Terbaik", "Accuracy 87.50%", "Recall 89.22%", "ROC AUC 0.9225"])
    st.markdown("""
    <div class="glass-panel" style="border-left:8px solid #315F38;line-height:1.8;">
    Penelitian ini berhasil membangun model klasifikasi penyakit jantung menggunakan Logistic Regression dan Gaussian Naive Bayes. Setelah preprocessing berupa penanganan anomali, encoding, scaling, dan train-test split 80:20, kedua model mampu memberikan performa yang baik.<br><br>
    <b>Gaussian Naive Bayes dipilih sebagai model terbaik</b> karena memperoleh Accuracy 87,50%, Precision 88,35%, Recall 89,22%, F1-Score 88,78%, dan ROC AUC 0,9225. Dashboard Streamlit ini digunakan untuk menyajikan informasi dataset, tahapan penelitian, hasil evaluasi, confusion matrix, ROC curve, dan kalkulator prediksi pasien baru.
    </div>
    """, unsafe_allow_html=True)


inject_css()
st.sidebar.markdown("""
<div style="font-size:76px;line-height:1;margin:24px 0 8px;">🫀</div>
<h2 style="font-family:Fraunces,serif;">Menu Navigasi</h2>
<p style="line-height:1.7;font-weight:700;">Heart Failure Prediction Dataset</p>
""", unsafe_allow_html=True)
page = st.sidebar.radio(
    "",
    ["🏠 Overview", "📊 Data Understanding", "🤖 Model Evaluation", "🧮 Kalkulator Prediksi", "✅ Kesimpulan"],
    label_visibility="collapsed",
)
st.sidebar.markdown("""
<div style="margin-top:28px;padding:18px;border-radius:22px;background:rgba(255,253,247,.7);border:1px solid rgba(49,95,56,.16);">
<b>Dataset</b><br>918 pasien, 12 atribut<br><br><b>Model utama</b><br>Gaussian Naive Bayes<br><br><b>Halaman interaktif</b><br>Kalkulator Prediksi
</div>
""", unsafe_allow_html=True)

if page == "🏠 Overview":
    page_overview()
elif page == "📊 Data Understanding":
    page_data()
elif page == "🤖 Model Evaluation":
    page_model()
elif page == "🧮 Kalkulator Prediksi":
    page_calculator()
else:
    page_conclusion()
