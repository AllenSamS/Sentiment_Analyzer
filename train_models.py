import pandas as pd
import re
import nltk
import joblib

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# -----------------------------
# Text cleaning
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    return " ".join([w for w in words if w not in stop_words])


# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("IMDB Dataset.csv")
df["clean_review"] = df["review"].apply(clean_text)

X_text = df["clean_review"]
y = df["sentiment"].map({"positive": 1, "negative": 0})

# -----------------------------
# Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X_text)

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train Logistic Regression
# -----------------------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# -----------------------------
# Train SVM
# -----------------------------
svm_model = LinearSVC()
svm_model.fit(X_train, y_train)

# -----------------------------
# Save models and vectorizer
# -----------------------------
joblib.dump(lr_model, "lr_model.pkl")
joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✅ Models and vectorizer saved successfully!")
