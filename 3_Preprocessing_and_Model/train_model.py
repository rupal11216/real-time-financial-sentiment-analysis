import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

nltk.download("stopwords")

# -------- Load dataset (IMPORTANT: latin1 encoding) --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "1_Dataset", "financial_news.csv")

df = pd.read_csv(DATASET_PATH, header=None, encoding="latin1")
df.columns = ["sentiment", "text"]

# -------- Preprocessing --------
stop_words = set(stopwords.words("english"))

df["clean_text"] = df["text"].str.lower()
df["clean_text"] = df["clean_text"].apply(
    lambda x: " ".join(w for w in x.split() if w not in stop_words)
)

# -------- Train-test split --------
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["sentiment"],
    test_size=0.2,
    random_state=42
)

# -------- TF-IDF --------
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# -------- Model --------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# -------- Evaluation --------
y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -------- Save evaluation output --------
os.makedirs("model_outputs", exist_ok=True)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.savefig("model_outputs/confusion_matrix.png")
plt.close()

# -------- Save model & vectorizer --------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf_vectorizer.pkl", "wb"))

print("Model training completed. Files saved.")
