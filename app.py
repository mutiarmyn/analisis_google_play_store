import streamlit as st
import pandas as pd
from google_play_scraper import reviews, Sort, app as play_app
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# ==============================
# CSS TAMPILAN
# ==============================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #F5F7FA;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 12em;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# CLEANING
# ==============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return text.strip()

# ==============================
# SCRAPING ULASAN
# ==============================
def get_reviews(app_id, n=200):
    result, _ = reviews(app_id, lang="id", country="id",
                        sort=Sort.NEWEST, count=n)
    df = pd.DataFrame(result)

    if "content" not in df.columns or "score" not in df.columns:
        return pd.DataFrame()

    df = df[["content", "score"]]
    df.columns = ["ulasan", "rating"]
    df["clean"] = df["ulasan"].apply(clean_text)
    return df

# ==============================
# LABEL SENTIMEN
# ==============================
def rating_to_label(score):
    if score <= 2:
        return "negatif"
    elif score == 3:
        return "netral"
    return "positif"

# ==============================
# TRAINING MODEL
# ==============================
def train_model(df):
    df["label"] = df["rating"].apply(rating_to_label)
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("nb", MultinomialNB())
    ])
    model.fit(df["clean"], df["label"])
    return model

# ==============================
# SESSION STATE DEFAULT
# ==============================
if "selected_app" not in st.session_state:
    st.session_state.selected_app = None

# ==============================
# SIDEBAR MENU
# ==============================
menu = st.sidebar.selectbox("📌 MENU",
    ["Analisis Sentimen", "Daftar Aplikasi Populer"])

st.sidebar.info("Developed by Mutia Rahmayani 😊")

# ========================================================
# 1️⃣ ANALISIS SENTIMEN
# ========================================================
def show_analysis(app_id):
    st.title("📱 Analisis Sentimen Google Play Store")

    with st.spinner("Mengambil ulasan dan menganalisis..."):
        df = get_reviews(app_id, 300)

        if df.empty:
            st.error("Gagal mengambil ulasan. App ID mungkin salah.")
            return

        model = train_model(df)
        df["sentimen_pred"] = model.predict(df["clean"])

    st.success("Analisis selesai!")

    counts = df["sentimen_pred"].value_counts().reindex(
        ["positif", "netral", "negatif"], fill_value=0
    )

    st.subheader("📊 Statistik Sentimen")
    st.write(f"👍 Positif : **{counts['positif']}**")
    st.write(f"😐 Netral : **{counts['netral']}**")
    st.write(f"👎 Negatif : **{counts['negatif']}**")

    # Bar Chart
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x=counts.index, y=counts.values, palette="Set2", ax=ax)
    ax.set_title(f"Distribusi Sentimen untuk {app_id}")
    st.pyplot(fig)

    # Pie Chart
    fig2, ax2 = plt.subplots(figsize=(6,6))
    ax2.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
            colors=["#4CAF50","#FFC107","#F44336"])
    st.pyplot(fig2)

    # Wordcloud Positif
    st.subheader("☁ WordCloud Positif")
    wc_pos = WordCloud(width=800, height=300).generate(
        " ".join(df[df["sentimen_pred"]=="positif"]["clean"]))
    st.image(wc_pos.to_array())

    # Wordcloud Negatif
    st.subheader("☁ WordCloud Negatif")
    wc_neg = WordCloud(width=800, height=300).generate(
        " ".join(df[df["sentimen_pred"]=="negatif"]["clean"]))
    st.image(wc_neg.to_array())

    # Tabel hasil
    st.subheader("📄 Tabel Ulasan")
    st.dataframe(df)

    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        "hasil_sentimen.csv",
        "text/csv"
    )

# ========================================================
# 2️⃣ TAMPILKAN HALAMAN BERDASARKAN MENU
# ========================================================
if menu == "Analisis Sentimen":

    # Jika user klik dari menu lain → langsung analisis
    if st.session_state.selected_app:
        show_analysis(st.session_state.selected_app)

    else:
        st.title("📱 Analisis Sentimen Google Play Store")
        app_id = st.text_input("Masukkan App ID:")

        if st.button("Analisis Sekarang"):
            if app_id.strip() == "":
                st.warning("Masukkan App ID dulu!")
            else:
                st.session_state.selected_app = app_id
                st.rerun()

# ========================================================
elif menu == "Daftar Aplikasi Populer":

    st.session_state.selected_app = None  # reset agar input manual tidak ikut

    st.title("📱 Daftar Aplikasi Populer")

    daftar_app = {
        "Instagram": "com.instagram.android",
        "Facebook": "com.facebook.katana",
        "TikTok": "com.zhiliaoapp.musically",
        "Shopee": "com.shopee.id",
        "Tokopedia": "com.tokopedia.tkpd",
        "WhatsApp": "com.whatsapp",
        "Mobile Legends": "com.mobile.legends",
        "PUBG Mobile": "com.tencent.ig",
    }

    pilihan = st.selectbox("Pilih aplikasi:", list(daftar_app.keys()))
    app_id = daftar_app[pilihan]

    app_info = play_app(app_id, lang="id", country="id")

    st.subheader(app_info["title"])
    st.image(app_info["icon"])
    st.write("**Deskripsi:**", app_info["description"][:500] + " ...")
    st.write("**Kategori:**", app_info["genre"])
    st.write("**Rating:**", app_info["score"])
    st.write("**Installs:**", app_info.get("installs", "N/A"))

