import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# =====================================================
# Project Paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "outputs" / "fake_news_model.pkl"
TFIDF_PATH = BASE_DIR / "outputs" / "tfidf.pkl"


# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="AI Fake News Detection",
    page_icon="📰",
    layout="wide"
)


# =====================================================
# Load Trained Model
# =====================================================

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)

    return model, tfidf


model, tfidf = load_model()

# =====================================================
# Text Cleaning
# =====================================================

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove HTML Tags
    text = re.sub(r"<.*?>", "", text)

    # Remove Numbers
    text = re.sub(r"\d+", "", text)

    # Remove Punctuation
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenization
    words = word_tokenize(text)

    # Remove Stopwords
    words = [word for word in words if word not in stop_words]

    # Lemmatization
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)


# =====================================================
# Title
# =====================================================

st.title("📰 AI Fake News Detection")

st.write(
    "An NLP and Machine Learning based system for detecting "
    "whether a news article is likely to be Fake or Real."
)

st.success("Trained model and TF-IDF vectorizer loaded successfully!")

# =====================================================
# News Prediction
# =====================================================

st.header("🔎 Check a News Article")

news_text = st.text_area(
    "Paste the news article below:",
    height=250,
    placeholder="Enter or paste a news article here..."
)

if st.button("Predict News"):
    if not news_text.strip():
        st.warning("Please enter a news article first.")
    else:
        # Convert the article using the trained TF-IDF vectorizer
        cleaned_news = clean_text(news_text)
        news_vector = tfidf.transform([cleaned_news])
        # Make prediction
        prediction = model.predict(news_vector)[0]

        # Get prediction probabilities
        probabilities = model.predict_proba(news_vector)[0]

        fake_probability = probabilities[0] * 100
        real_probability = probabilities[1] * 100

        # Display result
        st.subheader("Prediction")

        if prediction == 1:
          st.success("🟢 REAL NEWS")
        else:
          st.error("🔴 FAKE NEWS")

        # Display probabilities
        st.write(f"**Fake Probability:** {fake_probability:.2f}%")
        st.write(f"**Real Probability:** {real_probability:.2f}%")

        # Probability bar
        st.progress(int(real_probability))