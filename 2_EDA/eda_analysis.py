import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "1_Dataset", "financial_news.csv")

df = pd.read_csv(DATASET_PATH, header=None, encoding="latin1")
df.columns = ["sentiment", "text"]

print(df.head())
print(df["sentiment"].value_counts())

# Create output folder
os.makedirs("eda_outputs", exist_ok=True)

# Sentiment distribution
plt.figure()
sns.countplot(x="sentiment", data=df)
plt.title("Sentiment Distribution")
plt.savefig("eda_outputs/sentiment_distribution.png")
plt.close()

# Text length distribution
df["length"] = df["text"].apply(len)
plt.figure()
sns.histplot(df["length"], bins=30)
plt.title("Text Length Distribution")
plt.savefig("eda_outputs/text_length_distribution.png")
plt.close()

print("EDA completed. Plots saved.")
