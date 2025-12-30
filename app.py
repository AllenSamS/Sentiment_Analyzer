import streamlit as st
import joblib
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)

# ======================================================
# CUSTOM CSS (FIXED VISIBILITY + MODERN UI)
# ======================================================
st.markdown("""
<style>

/* Global background */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    font-family: "Segoe UI", sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #141e30, #243b55);
}
section[data-testid="stSidebar"] * {
    color: #ffffff !important;
    font-size: 16px;
}

/* General text */
p, li, span {
    color: #e5e7eb !important;
    font-size: 16px;
}

/* Headings */
h1, h2, h3, h4 {
    color: #ffffff !important;
    font-weight: 800;
}

/* Custom titles */
.main-title {
    font-size: 46px;
    font-weight: 800;
    text-align: center;
    color: #ffffff;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #d1d5db;
    margin-bottom: 30px;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.96);
    padding: 30px;
    border-radius: 18px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.25);
    margin-bottom: 30px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #ff512f, #dd2476);
    color: white;
    border-radius: 14px;
    font-size: 18px;
    padding: 12px 26px;
    font-weight: 700;
    border: none;
}
.stButton > button:hover {
    transform: scale(1.05);
}

/* Text input */
textarea {
    border-radius: 14px !important;
    border: 2px solid #93c5fd !important;
    background-color: #f8fafc !important;
    color: #111827 !important;
    font-size: 16px !important;
}

/* Info boxes */
.success-box {
    background: #ecfdf5;
    border-left: 8px solid #10b981;
    padding: 16px;
    border-radius: 12px;
    font-size: 20px;
    color: #065f46;
    font-weight: 600;
}

.info-box {
    background: #eff6ff;
    border-left: 8px solid #2563eb;
    padding: 16px;
    border-radius: 12px;
    font-size: 18px;
    color: #1e3a8a;
}

/* Footer */
.footer {
    text-align: center;
    color: #d1d5db;
    font-size: 14px;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD MODEL
# ======================================================
model = joblib.load("lr_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ======================================================
# SESSION STATE
# ======================================================
if "history" not in st.session_state:
    st.session_state.history = []

# ======================================================
# TEXT CLEANING
# ======================================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return " ".join([w for w in text.split() if w not in stop_words])

# ======================================================
# KEYWORD EXPLANATION
# ======================================================
def explain_prediction(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    feature_names = vectorizer.get_feature_names_out()
    scores = vec.toarray()[0]

    return [w for w, _ in sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)[:6]]

# ======================================================
# EMOTION LEVEL
# ======================================================
def emotion_label(conf):
    if conf > 85:
        return "🔥 Very Strong Emotion"
    elif conf > 70:
        return "🙂 Strong Emotion"
    elif conf > 55:
        return "😐 Moderate Emotion"
    else:
        return "🤔 Weak Emotion"

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Choose a section",
    ["🎯 Sentiment Analyzer", "📜 Prediction History", "ℹ️ About Project"]
)

# ======================================================
# SENTIMENT ANALYZER
# ======================================================
if page == "🎯 Sentiment Analyzer":

    st.markdown("<div class='main-title'>🎬 Movie Review Sentiment Analyzer</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Analyze emotions behind movie reviews using Machine Learning</div>", unsafe_allow_html=True)

    review = st.text_area("✍️ Enter your movie review here:")

    if st.button("🔍 Analyze Sentiment"):
        if review.strip() == "":
            st.warning("Please enter a review.")
        else:
            cleaned = clean_text(review)
            vec = vectorizer.transform([cleaned])
            probs = model.predict_proba(vec)[0]

            sentiment = "Positive 😊" if probs[1] > probs[0] else "Negative 😞"
            confidence = max(probs) * 100

            st.markdown(f"<div class='success-box'>Sentiment: <b>{sentiment}</b></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='info-box'>Confidence Score: <b>{confidence:.2f}%</b></div>", unsafe_allow_html=True)

            st.progress(confidence / 100)

            st.markdown(f"### 🎭 Emotion Level: {emotion_label(confidence)}")

            keywords = explain_prediction(review)
            st.markdown("### 🔍 Key Influencing Words")
            st.write(", ".join(keywords))

            st.session_state.history.append({
                "Review": review[:80] + "...",
                "Sentiment": sentiment,
                "Confidence": round(confidence, 2)
            })

# ======================================================
# HISTORY
# ======================================================
elif page == "📜 Prediction History":

    st.markdown("<div class='main-title'>📜 Prediction History</div>", unsafe_allow_html=True)

    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist)

        csv = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download History (CSV)", csv, "sentiment_history.csv")
    else:
        st.info("No predictions yet.")

# ======================================================
# ABOUT
# ======================================================
elif page == "ℹ️ About Project":

    st.markdown("<div class='main-title'>📘 About This Project</div>", unsafe_allow_html=True)

    st.markdown("""
### 🎯 Project Objective
Classify movie reviews into **Positive** or **Negative** sentiments using Machine Learning.

### 🧠 Technologies Used
- Python  
- Scikit-learn  
- NLTK  
- Streamlit  
- Pandas  

### ⚙️ ML Workflow
- Text preprocessing  
- TF-IDF vectorization  
- Logistic Regression  
- Model evaluation  

### 🌟 Features
✔ Real-time prediction  
✔ Confidence score  
✔ Emotion intensity  
✔ Keyword explanation  
✔ Prediction history  
✔ CSV export  
✔ Interactive UI  

This project demonstrates how **NLP + ML** can be applied to real-world text analytics.
""")
