import pandas as pd
import re
import nltk
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# -----------------------------
# Text cleaning
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return " ".join([w for w in text.split() if w not in stop_words])


# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("IMDB Dataset.csv")
df["clean_review"] = df["review"].apply(clean_text)

X_text = df["clean_review"]
y = df["sentiment"].map({"positive": 1, "negative": 0})

# -----------------------------
# Load vectorizer & models
# -----------------------------
vectorizer = joblib.load("tfidf_vectorizer.pkl")
lr_model = joblib.load("lr_model.pkl")
svm_model = joblib.load("svm_model.pkl")

# Transform text
X = vectorizer.transform(X_text)

# Train-test split (same split logic)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Predictions
# -----------------------------
lr_pred = lr_model.predict(X_test)
svm_pred = svm_model.predict(X_test)

# -----------------------------
# Accuracy comparison
# -----------------------------
results = pd.DataFrame({
    "Model": ["Logistic Regression", "SVM"],
    "Accuracy": [
        accuracy_score(y_test, lr_pred),
        accuracy_score(y_test, svm_pred)
    ]
})

print("\nAccuracy Comparison:")
print(results)

# -----------------------------
# Classification report
# -----------------------------
print("\nLogistic Regression Report:\n")
print(classification_report(y_test, lr_pred))

print("\nSVM Report:\n")
print(classification_report(y_test, svm_pred))

# -----------------------------
# Confusion Matrix (Logistic Regression)
# -----------------------------
cm = confusion_matrix(y_test, lr_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
