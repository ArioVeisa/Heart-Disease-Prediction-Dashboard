import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Heart Disease Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@700;800&family=Manrope:wght@400;600;700;800&display=swap');

:root {
    --cream: #FFF8EA;
    --paper: #FFFDF7;
    --ink: #17362F;
    --leaf: #315F38;
    --mint: #EAF4E7;
    --sage: #9DC08B;
    --gold: #F2C14E;
    --coral: #E86F51;
    --line: rgba(49, 95, 56, .16);
}

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
}

.stApp {
    color: var(--ink);
    background:
        radial-gradient(circle at 8% 12%, rgba(242, 193, 78, .24), transparent 28%),
        radial-gradient(circle at 92% 18%, rgba(157, 192, 139, .38), transparent 30%),
        linear-gradient(135deg, #FFF8EA 0%, #F6F1DE 48%, #E9F3E2 100%);
}

.block-container {
    max-width: 1220px;
    padding-top: 2.2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #EAF4E7 0%, #F9F1D7 100%);
    border-right: 1px solid var(--line);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: var(--leaf);
}

section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    color: var(--leaf);
    font-family: 'Fraunces', serif;
}

.hero {
    position: relative;
    overflow: hidden;
    padding: 44px 48px;
    border-radius: 34px;
    background:
        linear-gradient(115deg, rgba(255,253,247,.96), rgba(234,244,231,.88)),
        repeating-linear-gradient(45deg, rgba(49,95,56,.06) 0 1px, transparent 1px 14px);
    border: 1px solid var(--line);
    box-shadow: 0 24px 80px rgba(23, 54, 47, .12);
}

.hero:after {
    content: "";
    position: absolute;
    width: 280px;
    height: 280px;
    right: -80px;
    top: -90px;
    border-radius: 999px;
    background: linear-gradient(135deg, rgba(242,193,78,.65), rgba(232,111,81,.25));
}

.kicker {
    display: inline-flex;
    gap: 8px;
    align-items: center;
    padding: 8px 14px;
    border-radius: 999px;
    background: rgba(49, 95, 56, .10);
    color: var(--leaf);
    font-weight: 800;
    letter-spacing: .08em;
    text-transform: uppercase;
    font-size: 12px;
}

.hero h1 {
    position: relative;
    z-index: 1;
    margin: 18px 0 12px;
    max-width: 900px;
    color: var(--leaf);
    font-family: 'Fraunces', serif;
    font-size: clamp(42px, 6vw, 76px);
    line-height: .96;
    letter-spacing: -.04em;
}

.hero p {
    position: relative;
    z-index: 1;
    max-width: 860px;
    color: #304F3F;
    font-size: 17px;
    line-height: 1.8;
}

.metric-card {
    min-height: 155px;
    padding: 24px;
    border-radius: 26px;
    background: rgba(255, 253, 247, .88);
    border: 1px solid var(--line);
    box-shadow: 0 18px 45px rgba(23, 54, 47, .09);
    transition: transform .18s ease, box-shadow .18s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 24px 55px rgba(23, 54, 47, .14);
}

.metric-icon {
    width: 44px;
    height: 44px;
    display: grid;
    place-items: center;
    border-radius: 16px;
    background: var(--mint);
    font-size: 23px;
    margin-bottom: 16px;
}

.metric-title {
    color: #5D715F;
    font-size: 14px;
    font-weight: 800;
    margin-bottom: 6px;
}

.big-number {
    color: var(--leaf);
    font-family: 'Fraunces', serif;
    font-size: 42px;
    line-height: 1;
    font-weight: 800;
    margin-bottom: 8px;
}

.metric-desc {
    color: #55705C;
    font-size: 14px;
    font-weight: 600;
}

.section-title {
    margin: 14px 0 10px;
    color: var(--leaf);
    font-family: 'Fraunces', serif;
    font-size: 34px;
    letter-spacing: -.03em;
}

.glass-panel {
    padding: 24px;
    border-radius: 28px;
    background: rgba(255, 253, 247, .82);
    border: 1px solid var(--line);
    box-shadow: 0 20px 60px rgba(23, 54, 47, .10);
}

.insight-box {
    background: linear-gradient(135deg, #315F38, #17362F);
    color: #FFF8EA;
    padding: 24px;
    border-radius: 24px;
    line-height: 1.75;
    box-shadow: 0 22px 48px rgba(23,54,47,.20);
}

.matrix-box {
    background: #FFFDF7;
    color: var(--ink);
    padding: 22px;
    border-radius: 22px;
    border: 1px solid var(--line);
}

.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 18px;
}

.badge {
    padding: 9px 13px;
    border-radius: 999px;
    background: rgba(242, 193, 78, .24);
    color: var(--leaf);
    font-weight: 800;
    font-size: 13px;
}

[data-testid="stMetric"] {
    background: rgba(255, 253, 247, .82);
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 16px 18px;
    box-shadow: 0 12px 30px rgba(23,54,47,.07);
}

[data-testid="stMetricValue"] {
    color: var(--leaf);
    font-family: 'Fraunces', serif;
}

[data-testid="stDataFrame"] {
    border-radius: 22px;
    overflow: hidden;
    border: 1px solid var(--line);
}

hr {
    border-color: rgba(49, 95, 56, .14) !important;
    margin: 2rem 0 !important;
}

@media (max-width: 768px) {
    .hero { padding: 30px 24px; border-radius: 24px; }
    .hero h1 { font-size: 40px; }
    .metric-card { min-height: auto; }
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="font-size:76px; line-height:1; margin: 24px 0 8px;">🫀</div>
<h2>Menu Navigasi</h2>
<p style="line-height:1.7; font-weight:700;">Heart Failure Prediction Dataset</p>
""", unsafe_allow_html=True)
menu = st.sidebar.radio(
    "",
    ["🔴 Overview", "📊 Model Performance", "🧪 Evaluation Visuals", "ℹ️ About Project"],
    label_visibility="collapsed",
)
st.sidebar.markdown("""
<div style="margin-top:28px; padding:18px; border-radius:22px; background:rgba(255,253,247,.7); border:1px solid rgba(49,95,56,.16);">
<b>Dataset</b><br>918 pasien<br><br>
<b>Model</b><br>Logistic Regression<br>Gaussian Naive Bayes
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="kicker">❤️ Data Mining Dashboard</div>
    <h1>Heart Disease Prediction Dashboard</h1>
    <p><b>Selamat datang di Dashboard Analisis Penyakit Jantung!</b> Dashboard ini menyajikan ringkasan dataset, evaluasi model, dan perbandingan performa Logistic Regression dengan Gaussian Naive Bayes untuk membantu melihat model terbaik secara cepat.</p>
    <div class="badge-row">
        <span class="badge">918 Total Data</span>
        <span class="badge">Naive Bayes Terbaik</span>
        <span class="badge">ROC AUC 0.9225</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.write("")
st.write("")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-icon">👥</div>
        <div class="metric-title">Total Data</div>
        <div class="big-number">918</div>
        <div class="metric-desc">Jumlah seluruh data pasien</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-icon">🫀</div>
        <div class="metric-title">Sakit Jantung</div>
        <div class="big-number">508</div>
        <div class="metric-desc">55,3% dari total data</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-icon">✅</div>
        <div class="metric-title">Tidak Sakit</div>
        <div class="big-number">410</div>
        <div class="metric-desc">44,7% dari total data</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

st.markdown('<div class="section-title">Perbandingan Performa Model</div>', unsafe_allow_html=True)

metrics_df = pd.DataFrame({
    "Metrik": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC AUC"],
    "Logistic Regression": ["84,78%", "84,91%", "88,24%", "86,54%", "0,8989"],
    "Gaussian Naive Bayes": ["87,50%", "88,35%", "89,22%", "88,78%", "0,9225"],
    "Model Terbaik": ["Naive Bayes", "Naive Bayes", "Naive Bayes", "Naive Bayes", "Naive Bayes"]
})

st.dataframe(metrics_df, use_container_width=True, hide_index=True)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accuracy NB", "87.50%", "+2.72%")
c2.metric("Precision NB", "88.35%", "+3.44%")
c3.metric("Recall NB", "89.22%", "+0.98%")
c4.metric("F1-Score NB", "88.78%", "+2.24%")
c5.metric("ROC AUC NB", "0.9225", "+0.0236")

st.divider()

left, right = st.columns([1.15, 1])

with left:
    st.markdown('<div class="section-title">Grafik Perbandingan Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.image("output/model_comparison.png", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-title">Insight Utama</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-box">
        <b>Gaussian Naive Bayes menjadi model terbaik.</b><br><br>
        Model ini unggul pada seluruh metrik evaluasi, yaitu Accuracy, Precision, Recall,
        F1-Score, dan ROC AUC. Dalam kasus prediksi penyakit jantung, nilai Recall penting
        karena menunjukkan kemampuan model mendeteksi pasien yang benar-benar sakit.
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.markdown("""
    <div class="matrix-box">
        <b>Confusion Matrix Naive Bayes</b><br><br>
        True Negative: 70<br>
        False Positive: 12<br>
        False Negative: 11<br>
        True Positive: 91
    </div>
    """, unsafe_allow_html=True)

st.divider()

col_cm, col_roc = st.columns(2)

with col_cm:
    st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.image("output/confusion_matrix_nb.png", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_roc:
    st.markdown('<div class="section-title">ROC Curve</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.image("output/roc_curve_nb.png", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

st.markdown('<div class="section-title">Kesimpulan Dashboard</div>', unsafe_allow_html=True)
st.markdown("""
<div class="glass-panel" style="border-left: 8px solid #315F38;">
    <b>Berdasarkan hasil evaluasi, Gaussian Naive Bayes memiliki performa lebih baik dibandingkan Logistic Regression.</b><br><br>
    Model Naive Bayes memperoleh Accuracy 87,50%, Precision 88,35%, Recall 89,22%, F1-Score 88,78%, dan ROC AUC 0,9225.
    Oleh karena itu, Gaussian Naive Bayes dipilih sebagai model terbaik dalam penelitian ini.
</div>
""", unsafe_allow_html=True)
