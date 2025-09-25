# eda_telco.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import base64
from pathlib import Path

st.set_page_config(page_title="Telco Churn EDA", layout="wide")

# ========== Global plotting config ==========
mpl.rcParams.update({
"figure.figsize": (4.0, 2.5),
    "figure.dpi": 100,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})
sns.set_theme(context="notebook", style="whitegrid", font_scale=0.9)

def _newfig(w: float = 4.0, h: float = 2.5):
    return plt.figure(figsize=(w, h), dpi=100)

def _center_plot(fig):
    """Convert Matplotlib figure to PNG and display in center."""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    st.markdown(
        f"<div style='text-align: center;'><img src='data:image/png;base64,{img_base64}' style='max-width:100%; height:auto;'></div>",
        unsafe_allow_html=True
    )
    plt.close(fig)

@st.cache_data(show_spinner=False)
def load_telco(csv_path: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    if "Churn" in df.columns:
        df = df.rename(columns={"Churn": "churn"})
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df

def _has(cols, name):
    if name in cols:
        return name
    low = {c.lower(): c for c in cols}
    return low.get(name.lower(), None)

def run(csv_path: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    # Judul
    st.markdown("<h1 style='text-align: center;'>Telco Customer Churn</h1>", unsafe_allow_html=True)

    # Gambar churn_gambar di tengah
    img_path = Path("churn_gambar.jpg")
    if img_path.exists():
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        img_base64 = base64.b64encode(img_bytes).decode()
        st.markdown(
            f"""
            <div style="text-align: center;">
                <img src="data:image/jpeg;base64,{img_base64}" 
                    alt="churn_gambar" style="width:60%; border-radius:8px;">
                <p style="text-align: center; color: gray; font-size: 14px;">churn_gambar</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("Gambar 'churn_gambar.jpg' tidak ditemukan.")

    # Tujuan & Konteks
    st.markdown("""
# **Tujuan dan Konteks**

Proyek ini bertujuan untuk memprediksi **customer churn** dalam industri **telekomunikasi** menggunakan model **K-Nearest Neighbors (KNN)**. Tujuan utama adalah untuk mengidentifikasi pelanggan yang berisiko berhenti berlangganan, yang sangat penting bagi perusahaan untuk merancang **strategi retensi pelanggan** yang lebih baik.

Melalui analisis churn, perusahaan dapat memfokuskan upaya pada pelanggan yang berisiko untuk mengurangi tingkat **customer churn**.
""")

    # Load data
    try:
        df = load_telco(csv_path)
    except FileNotFoundError:
        st.error(f"File tidak ditemukan: {csv_path}")
        return
    except Exception as e:
        st.error(f"Gagal membaca CSV: {e}")
        return

    st.dataframe(df, use_container_width=True)

    # RINGKASAN DATA
    with st.expander("Ringkasan Statistik (describe)"):
        num_df = df.select_dtypes(include=["number"])
        cat_df = df.select_dtypes(include=["object", "category", "bool"])

        tabs = st.tabs(["Numerik", "Kategorik/Boolean"])

        with tabs[0]:
            if not num_df.empty:
                desc_num = num_df.describe().T
                desc_num["missing"]   = num_df.isna().sum().values
                desc_num["missing_%"] = (desc_num["missing"] / len(df) * 100).round(2)
                st.dataframe(desc_num.round(4), use_container_width=True, height=360)
            else:
                st.info("Tidak ada kolom numerik.")

        with tabs[1]:
            if not cat_df.empty:
                def cat_summary(s: pd.Series):
                    vc = s.value_counts(dropna=True)
                    top_val  = vc.index[0] if not vc.empty else "-"
                    top_freq = int(vc.iloc[0]) if not vc.empty else 0
                    return pd.Series({
                        "count": int(s.notna().sum()),
                        "unique": int(s.nunique(dropna=True)),
                        "top": top_val,
                        "freq": top_freq,
                        "top_%": round((top_freq / len(df) * 100), 2) if len(df) else 0.0,
                        "missing": int(s.isna().sum()),
                        "missing_%": round(s.isna().mean() * 100, 2)
                    })
                desc_cat = cat_df.apply(cat_summary, axis=0).T
                st.dataframe(desc_cat, use_container_width=True, height=360)
            else:
                st.info("Tidak ada kolom kategorik/boolean.")

    # DISTRIBUSI TARGET
    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'>Distribusi Target (Churn)</h2>", unsafe_allow_html=True)

    churn_col = _has(df.columns, "churn")
    if churn_col is None:
        st.error("Kolom 'churn' tidak ditemukan di dataset.")
        return

    fig = _newfig()
    sns.countplot(x=churn_col, data=df)
    plt.title("Distribusi Kelas Target (Churn)", loc='center')
    plt.xlabel("churn")
    plt.ylabel("Jumlah")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    _center_plot(fig)

    churn_percentage = (df[churn_col].value_counts(normalize=True) * 100).round(2)
    st.caption("Persentase kelas (%):")
    st.dataframe(churn_percentage.to_frame(name="percent"), use_container_width=False, height=100)

    st.markdown("""
**Hasil pada Dataset Ini:**
- Kelas **No** jauh lebih banyak daripada **Yes**.
- Mengindikasikan risiko **false negative** tinggi jika model tidak ditangani khusus.
""")

    # CONTRACT vs CHURN
    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'>Hubungan Kolom Contract dengan Churn</h2>", unsafe_allow_html=True)

    contract_col = _has(df.columns, "Contract")
    if contract_col is not None:
        order = ["Month-to-month", "One year", "Two year"]
        churn_rate = (
            df.groupby(contract_col)[churn_col]
            .apply(lambda s: (s == "Yes").mean() * 100)
            .reindex(order)
        )

        fig = _newfig()
        plt.plot(churn_rate.index, churn_rate.values, marker="o", linestyle="-")
        plt.title("Churn Rate berdasarkan Durasi Kontrak", loc='center')
        plt.xlabel("Durasi Kontrak")
        plt.ylabel("Churn Rate (%)")
        plt.grid(True, alpha=0.4)
        _center_plot(fig)

        fig = _newfig()
        plt.bar(churn_rate.index, churn_rate.values)
        plt.title("Churn Rate berdasarkan Durasi Kontrak", loc='center')
        plt.xlabel("Durasi Kontrak")
        plt.ylabel("Churn Rate (%)")
        plt.grid(axis="y", alpha=0.4)
        _center_plot(fig)

        st.markdown("""
### Visualisasi Tren Monotonic Durasi Kontrak terhadap Churn Rate
Berdasarkan data, churn rate menurun secara konsisten dari **Month-to-month** → **One year** → **Two year**.  
Semakin panjang durasi kontrak, semakin kecil kemungkinan pelanggan melakukan churn.
        """)

    # BOXPLOT OUTLIER
    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'>Pemeriksaan Outlier dengan Boxplot</h2>", unsafe_allow_html=True)

    num_cols_wishlist = ["tenure", "MonthlyCharges", "TotalCharges"]
    for col in [c for c in num_cols_wishlist if c in df.columns]:
        fig = _newfig()
        plt.boxplot(df[col].dropna(), vert=False, patch_artist=True)
        plt.title(f"Boxplot - {col}", loc='center')
        plt.xlabel(col)
        _center_plot(fig)

    # Penjelasan setelah boxplot
    st.markdown("""
Berdasarkan hasil visualisasi boxplot dan histogram pada tiga fitur numerik utama:

- **`tenure`** memiliki rentang 0–72 bulan (6 tahun), yang sesuai dengan batas lama kontrak dan masuk akal secara bisnis.
- **`MonthlyCharges`** berada pada kisaran ±20–120, wajar untuk variasi tarif paket telekomunikasi mulai dari paket dasar hingga premium.
- **`TotalCharges`** proporsional terhadap `tenure × MonthlyCharges`, di mana nilai tinggi mencerminkan pelanggan dengan lama berlangganan besar dan tarif bulanan tinggi, sehingga bukan merupakan kesalahan data.
""")

    # =========================
    # CONFUSION MATRIX – KNN with SMOTE
    # =========================
    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'>Confusion Matrix – KNN with SMOTE</h2>", unsafe_allow_html=True)

    cm_path = Path("Confussion Matrix.png")  # nama file yang kamu sebutkan
    if cm_path.exists():
        cm_bytes = cm_path.read_bytes()
        cm_b64 = base64.b64encode(cm_bytes).decode()
        st.markdown(
            f"""
            <div style="text-align: center;">
                <img src="data:image/png;base64,{cm_b64}" 
                    alt="Confusion Matrix – KNN with SMOTE" style="width:60%; border-radius:8px;">
                <p style="text-align:center;color:gray;font-size:14px;margin-top:4px;">
                    Confusion Matrix – KNN with SMOTE
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("Gambar 'Confussion Matrix.png' tidak ditemukan di direktori saat ini.")

    # Penjelasan di bawah gambar confusion matrix
    st.markdown("""
**Confusion Matrix KNN dengan SMOTE**

Confusion matrix di atas menunjukkan hasil prediksi model KNN berdasarkan data yang digunakan. Berikut adalah penjelasan elemen-elemen dari confusion matrix tersebut:

**Confusion Matrix Breakdown:**
1. **True Negative (TN)**: 679  
- **Interpretasi**: Model dengan benar memprediksi 679 pelanggan yang tidak churn (kelas "No").
2. **False Positive (FP)**: 357  
- **Interpretasi**: Model salah memprediksi 357 pelanggan yang sebenarnya tidak churn, namun diprediksi sebagai churn (kelas "Yes").
3. **False Negative (FN)**: 50  
- **Interpretasi**: Model salah memprediksi 50 pelanggan yang sebenarnya churn, namun diprediksi sebagai tidak churn (kelas "No").
4. **True Positive (TP)**: 323  
- **Interpretasi**: Model dengan benar memprediksi 323 pelanggan yang benar-benar churn (kelas "Yes").

**Penggunaan Confusion Matrix - Fokus pada `Recall`:**
- **Recall untuk kelas "Yes"**:  
- Recall mengukur kemampuan model dalam menemukan semua pelanggan yang benar-benar churn.  
- Model ini memiliki recall yang baik, karena dapat mengidentifikasi 323 pelanggan yang benar-benar churn dengan sedikit kesalahan.  
- **False Negative (FN)** yang rendah, yaitu 50, menunjukkan bahwa model tidak terlalu sering melewatkan pelanggan yang churn, sehingga dapat membantu dalam menangani pelanggan yang berisiko berhenti berlangganan.

**Kesimpulan:**
- Model sangat baik dalam mengidentifikasi pelanggan yang berisiko churn (Recall tinggi untuk kelas "Yes"). Ini dapat digunakan sebagai langkah antisipatif untuk mencegah pelanggan pergi dan berhenti berlangganan.
""")

if __name__ == "__main__":
    run()
