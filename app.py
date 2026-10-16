"""
Heart Attack Risk Prediction — Streamlit demo app

Model: Logistic Regression + StandardScaler (konfigurasi yang sama dengan notebook
`notebooks/heart_attack_analysis.ipynb`), dilatih pada seluruh data bersih (302 pasien).

⚠️ Aplikasi ini hanya untuk tujuan edukasi / portofolio, BUKAN alat diagnosis medis.
"""
import os

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "heart.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "outputs", "model_comparison_summary.csv")
FIG_DIR = os.path.join(BASE_DIR, "images", "figures")

FEATURES = ["age", "sex", "cp", "trtbps", "chol", "fbs", "restecg",
            "thalachh", "exng", "oldpeak", "slp", "caa", "thall"]

FEATURE_LABELS = {
    "age": "Usia",
    "sex": "Jenis kelamin",
    "cp": "Tipe nyeri dada (cp)",
    "trtbps": "Tekanan darah istirahat (trtbps)",
    "chol": "Kolesterol (chol)",
    "fbs": "Gula darah puasa > 120 mg/dl (fbs)",
    "restecg": "Hasil EKG istirahat (restecg)",
    "thalachh": "Denyut jantung maksimum (thalachh)",
    "exng": "Angina akibat olahraga (exng)",
    "oldpeak": "ST depression (oldpeak)",
    "slp": "Kemiringan segmen ST (slp)",
    "caa": "Jumlah pembuluh darah utama (caa)",
    "thall": "Hasil tes thalium (thall)",
}

SHORT_LABELS = {
    "age": "age · usia", "sex": "sex · jenis kelamin", "cp": "cp · nyeri dada",
    "trtbps": "trtbps · tekanan darah", "chol": "chol · kolesterol", "fbs": "fbs · gula darah",
    "restecg": "restecg · EKG", "thalachh": "thalachh · denyut maks", "exng": "exng · angina olahraga",
    "oldpeak": "oldpeak · ST depression", "slp": "slp · slope ST", "caa": "caa · pembuluh darah",
    "thall": "thall · tes thalium",
}

# Keterangan kode kategori mengikuti dokumentasi dataset di Kaggle
CHOICES = {
    "sex": {1: "1 · Laki-laki", 0: "0 · Perempuan"},
    "cp": {0: "0 · Typical angina", 1: "1 · Atypical angina",
           2: "2 · Non-anginal pain", 3: "3 · Asymptomatic"},
    "fbs": {0: "0 · Tidak", 1: "1 · Ya"},
    "restecg": {0: "0 · Normal", 1: "1 · Abnormalitas gelombang ST-T",
                2: "2 · Hipertrofi ventrikel kiri"},
    "exng": {0: "0 · Tidak", 1: "1 · Ya"},
    "slp": {0: "0", 1: "1", 2: "2"},
    "caa": {0: "0", 1: "1", 2: "2", 3: "3", 4: "4"},
    "thall": {0: "0", 1: "1 · Fixed defect", 2: "2 · Normal", 3: "3 · Reversible defect"},
}

st.set_page_config(page_title="Heart Attack Risk Prediction", page_icon="🫀", layout="wide")


# ---------------------------------------------------------------------------
# Data & model
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df.drop_duplicates().reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame) -> Pipeline:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    model.fit(df[FEATURES], df["output"])
    return model


df = load_data()
model = train_model(df)

# Contoh pasien diambil langsung dari dataset
PRESETS = {
    "Nilai tengah dataset (median)": df[FEATURES].median().round().astype(float).to_dict(),
    "Contoh pasien A (dari dataset, output = 1)": df.loc[0, FEATURES].astype(float).to_dict(),
    "Contoh pasien B (dari dataset, output = 0)": df[df["output"] == 0].iloc[0][FEATURES].astype(float).to_dict(),
}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🫀 Heart Attack Risk Prediction")
st.caption(
    "Prediksi risiko serangan jantung dari 13 data klinis menggunakan **Logistic Regression** "
    "(akurasi ≈ 84%, AUC ≈ 0,91 pada evaluasi Repeated Holdout)."
)
st.warning(
    "Aplikasi ini dibuat untuk **tujuan edukasi dan portofolio data mining**, bukan alat diagnosis medis. "
    "Konsultasikan kondisi kesehatan dengan tenaga medis profesional.",
    icon="⚠️",
)

tab_pred, tab_eval, tab_data = st.tabs(["🔮 Prediksi", "📊 Performa Model", "🗂️ Tentang Data"])

with tab_pred:
    preset_name = st.selectbox("Isi otomatis dengan contoh", list(PRESETS.keys()))
    p = PRESETS[preset_name]

    with st.form("input_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input(FEATURE_LABELS["age"], 18, 100, int(p["age"]))
            sex = st.selectbox(FEATURE_LABELS["sex"], list(CHOICES["sex"]),
                               index=list(CHOICES["sex"]).index(int(p["sex"])),
                               format_func=CHOICES["sex"].get)
            cp = st.selectbox(FEATURE_LABELS["cp"], list(CHOICES["cp"]),
                              index=list(CHOICES["cp"]).index(int(p["cp"])),
                              format_func=CHOICES["cp"].get)
            trtbps = st.number_input(FEATURE_LABELS["trtbps"] + " — mmHg", 80, 220, int(p["trtbps"]))
            chol = st.number_input(FEATURE_LABELS["chol"] + " — mg/dl", 100, 600, int(p["chol"]))
        with c2:
            fbs = st.selectbox(FEATURE_LABELS["fbs"], list(CHOICES["fbs"]),
                               index=list(CHOICES["fbs"]).index(int(p["fbs"])),
                               format_func=CHOICES["fbs"].get)
            restecg = st.selectbox(FEATURE_LABELS["restecg"], list(CHOICES["restecg"]),
                                   index=list(CHOICES["restecg"]).index(int(p["restecg"])),
                                   format_func=CHOICES["restecg"].get)
            thalachh = st.number_input(FEATURE_LABELS["thalachh"] + " — bpm", 60, 220, int(p["thalachh"]))
            exng = st.selectbox(FEATURE_LABELS["exng"], list(CHOICES["exng"]),
                                index=list(CHOICES["exng"]).index(int(p["exng"])),
                                format_func=CHOICES["exng"].get)
        with c3:
            oldpeak = st.number_input(FEATURE_LABELS["oldpeak"], 0.0, 7.0, float(p["oldpeak"]), step=0.1)
            slp = st.selectbox(FEATURE_LABELS["slp"], list(CHOICES["slp"]),
                               index=list(CHOICES["slp"]).index(int(p["slp"])),
                               format_func=CHOICES["slp"].get)
            caa = st.selectbox(FEATURE_LABELS["caa"], list(CHOICES["caa"]),
                               index=list(CHOICES["caa"]).index(int(p["caa"])),
                               format_func=CHOICES["caa"].get)
            thall = st.selectbox(FEATURE_LABELS["thall"], list(CHOICES["thall"]),
                                 index=list(CHOICES["thall"]).index(int(p["thall"])),
                                 format_func=CHOICES["thall"].get)
        submitted = st.form_submit_button("Prediksi Risiko", type="primary", width="stretch")

    if submitted:
        x = pd.DataFrame([{
            "age": age, "sex": sex, "cp": cp, "trtbps": trtbps, "chol": chol, "fbs": fbs,
            "restecg": restecg, "thalachh": thalachh, "exng": exng, "oldpeak": oldpeak,
            "slp": slp, "caa": caa, "thall": thall,
        }])[FEATURES]
        proba = float(model.predict_proba(x)[0, 1])
        high = proba >= 0.5

        st.divider()
        r1, r2 = st.columns([1, 2])
        with r1:
            st.metric("Probabilitas risiko tinggi", f"{proba:.1%}")
            st.progress(proba)
            if high:
                st.error("**Prediksi: RISIKO TINGGI** (output = 1)")
            else:
                st.success("**Prediksi: RISIKO RENDAH** (output = 0)")
        with r2:
            # Kontribusi tiap fitur terhadap log-odds = koefisien × nilai terstandardisasi
            scaler = model.named_steps["scaler"]
            coef = model.named_steps["clf"].coef_[0]
            z = (x.values[0] - scaler.mean_) / scaler.scale_
            contrib = pd.DataFrame({
                "Fitur": [SHORT_LABELS[f] for f in FEATURES],
                "Kontribusi": coef * z,
            }).sort_values("Kontribusi", key=abs, ascending=False).head(6)
            st.markdown("**Faktor paling berpengaruh pada prediksi ini**")
            st.bar_chart(contrib.set_index("Fitur"), horizontal=True, height=260)
            st.caption("Nilai positif mendorong ke risiko tinggi, nilai negatif ke risiko rendah "
                       "(koefisien Logistic Regression × nilai fitur terstandardisasi).")

with tab_eval:
    st.subheader("Perbandingan Logistic Regression vs Random Forest")
    res = pd.read_csv(RESULTS_PATH)
    st.dataframe(
        res.style.format({c: "{:.3f}" for c in ["Akurasi", "Sensitivitas", "Spesifisitas", "AUC"]}),
        hide_index=True, width="stretch",
    )
    e1, e2 = st.columns(2)
    e1.image(os.path.join(FIG_DIR, "07_roc_curve_comparison.png"), caption="Kurva ROC")
    e2.image(os.path.join(FIG_DIR, "09_metric_comparison_bar.png"), caption="Perbandingan metrik")
    e3, e4 = st.columns(2)
    e3.image(os.path.join(FIG_DIR, "06_feature_importance_rf.png"), caption="Feature importance (Random Forest)")
    e4.image(os.path.join(FIG_DIR, "08_confusion_matrices.png"), caption="Confusion matrix")

with tab_data:
    st.subheader("Heart Attack Analysis & Prediction Dataset")
    d1, d2, d3 = st.columns(3)
    d1.metric("Jumlah pasien (setelah hapus duplikat)", len(df))
    d2.metric("Fitur prediktor", len(FEATURES))
    d3.metric("Proporsi risiko tinggi", f"{df['output'].mean():.1%}")
    st.dataframe(df, hide_index=True, width="stretch", height=320)
    st.image(os.path.join(FIG_DIR, "04_correlation_heatmap.png"), caption="Correlation heatmap", width=700)
