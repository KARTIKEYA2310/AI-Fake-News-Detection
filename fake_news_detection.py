# ============================================================
# AI-Powered Fake News Detection Using Text Classification
# Phase 1 : Data Loading, Preprocessing and Exploratory Data Analysis
# ============================================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from wordcloud import WordCloud

# Download NLTK Resources (Run only once)
# try:
#     nltk.data.find("tokenizers/punkt")
# except LookupError:
#     nltk.download("punkt")

# try:
#     nltk.data.find("corpora/stopwords")
# except LookupError:
#     nltk.download("stopwords")

# try:
#     nltk.data.find("corpora/wordnet")
# except LookupError:
#     nltk.download("wordnet")

# ============================================================
# Load Dataset
# ============================================================
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

fake = pd.read_csv(BASE_DIR / "datasets" / "Fake.csv" / "Fake.csv")
true = pd.read_csv(BASE_DIR / "datasets" / "True.csv" / "True.csv")

fake["label"] = 0      # Fake News
true["label"] = 1      # Real News

df = pd.concat([fake, true], ignore_index=True)
df = df.drop_duplicates(subset="text")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print("="*50)
print("First Five Rows")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

# ============================================================
# Dataset Information
# ============================================================

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nStatistical Summary")
print(df.describe(include="all"))

# ============================================================
# Keep Required Columns
# ============================================================

df = df[['title','text','label']]

# Merge title and full text
df["text"] = df["title"].fillna('') + " " + df["text"].fillna('')

df = df[['text','label']]
df = df.dropna()

print("\nNew Dataset")
print(df.head())

# ============================================================
# Text Cleaning
# ============================================================

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

print("\nCleaning Text...")

df["clean_text"] = df["text"].apply(clean_text)

print("Cleaning Completed!")

# ============================================================
# Save Cleaned Dataset
# ============================================================

df.to_csv("cleaned_fake_news.csv", index=False)

# ============================================================
# Exploratory Data Analysis (EDA)
# ============================================================

print("\nLabel Distribution")
print(df["label"].value_counts())

# Label Count Plot
plt.figure(figsize=(6,5))
sns.countplot(x="label", data=df)
plt.title("Fake vs Real News")
plt.xlabel("Label")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ============================================================
# Text Length Distribution
# ============================================================

df["text_length"] = df["clean_text"].apply(len)

plt.figure(figsize=(8,5))

sns.histplot(x=df["text_length"], bins=40)
plt.title("Distribution of Text Length")
plt.xlabel("Text Length")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ============================================================
# Most Frequent Words
# ============================================================

all_words = " ".join(
    df["clean_text"].sample(10000, random_state=42)
).split()

counter = Counter(all_words)

print("\nTop 20 Most Frequent Words\n")

for word, count in counter.most_common(20):
    print(f"{word:20} {count}")

# ============================================================
# Word Cloud
# ============================================================

print("\nGenerating Word Cloud.")

# Use a sample instead of the entire dataset
sample_text = " ".join(
    df["clean_text"].sample(n=5000, random_state=42)
)

wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color="white"
).generate(sample_text)

print("Word Cloud Generated!")

plt.figure(figsize=(14,7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud")
plt.tight_layout()
plt.show()

print("Word Cloud Displayed Successfully!")

# ============================================================
# Display Final Dataset
# ============================================================

print("\nFinal Dataset")

print(df.head())

print("\nDataset Shape")

print(df.shape)

print("\nPreprocessing Completed Successfully!")


# ============================================================
# AI-Powered Fake News Detection
# Phase 2 : Feature Engineering & Model Training
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# Encode Labels
# ============================================================

print("="*60)
print("Checking Labels")
print(df["label"].value_counts())
print(df[["clean_text", "label"]].sample(10))

print("\nUnique Labels:")
print(df["label"].unique())

print("\nData Type:")
print(df["label"].dtype)

# Convert labels if they are strings
if df["label"].dtype == object:

    df["label"] = df["label"].str.lower()

    df["label"] = df["label"].replace({
        "real":1,
        "fake":0,
        "true":1,
        "false":0
    })

print("\nEncoded Labels")
print(df["label"].value_counts())

# ============================================================
# Feature Engineering
# ============================================================

print("\nSplitting Dataset...")

y = df["label"]

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["clean_text"],
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2)
)

X_train = tfidf.fit_transform(X_train_text)
X_test = tfidf.transform(X_test_text)

print("Training Matrix :", X_train.shape)
print("Testing Matrix :", X_test.shape)

# ============================================================
# Train Test Split
# ============================================================



print("\nTraining Samples :", X_train.shape[0])
print("Testing Samples  :", X_test.shape[0])

# ============================================================
# Models
# ============================================================

models = {

    "KNN":
    KNeighborsClassifier(
        n_neighbors=5
    ),

    "Logistic Regression":
    LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Random Forest":
    RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
    ),

    "Neural Network":
    MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=150,
        random_state=42
    )

}

# ============================================================
# Training & Evaluation
# ============================================================

results = []

for name, model in models.items():

    print("\n"+"="*60)
    print(name)
    print("="*60)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(y_test, predictions)

    recall = recall_score(y_test, predictions)

    f1 = f1_score(y_test, predictions)

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    print("\nClassification Report\n")

    print(classification_report(
        y_test,
        predictions
    ))

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])

    # Confusion Matrix

    cm = confusion_matrix(
        y_test,
        predictions
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Fake","Real"]
    )

    disp.plot(cmap="Blues")

    plt.title(name)

    plt.show()

# ============================================================
# Comparison Table
# ============================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

print("\n")
print(results_df)

# ============================================================
# Accuracy Comparison
# ============================================================

plt.figure(figsize=(8,5))

sns.barplot(
    data=results_df,
    x="Model",
    y="Accuracy"
)

plt.title("Model Accuracy Comparison")

plt.xticks(rotation=15)

plt.show()

# ============================================================
# Precision Comparison
# ============================================================

plt.figure(figsize=(8,5))

sns.barplot(
    data=results_df,
    x="Model",
    y="Precision"
)

plt.title("Precision Comparison")

plt.xticks(rotation=15)

plt.show()

# ============================================================
# Recall Comparison
# ============================================================

plt.figure(figsize=(8,5))

sns.barplot(
    data=results_df,
    x="Model",
    y="Recall"
)

plt.title("Recall Comparison")

plt.xticks(rotation=15)

plt.show()

# ============================================================
# F1 Score Comparison
# ============================================================

plt.figure(figsize=(8,5))

sns.barplot(
    data=results_df,
    x="Model",
    y="F1 Score"
)

plt.title("F1 Score Comparison")

plt.xticks(rotation=15)

plt.show()

# ============================================================
# Save Results
# ============================================================

results_df.to_csv(
    "model_comparison.csv",
    index=False
)


print("\nProject Phase 2 Completed Successfully!")

# ============================================================
# AI-Powered Fake News Detection
# Phase 3 : Advanced Analysis & Prediction
# ============================================================

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ============================================================
# BAG OF WORDS
# ============================================================

print("="*60)
print("BAG OF WORDS")
print("="*60)

bow = CountVectorizer(
    max_features=5000,
    ngram_range=(1,2)
)

X_bow = bow.fit_transform(df["clean_text"])

X_train_bow, X_test_bow, y_train_bow, y_test_bow = train_test_split(
    X_bow,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

bow_model = LogisticRegression(max_iter=1000)

bow_model.fit(X_train_bow, y_train_bow)

bow_predictions = bow_model.predict(X_test_bow)

bow_accuracy = accuracy_score(y_test_bow, bow_predictions)

print("\nBag of Words Accuracy :", bow_accuracy)

# ============================================================
# TF-IDF Accuracy
# ============================================================

tfidf_accuracy = results_df.loc[
    results_df["Model"]=="Logistic Regression",
    "Accuracy"
].values[0]

print("TF-IDF Accuracy :", tfidf_accuracy)

# ============================================================
# Comparison
# ============================================================

comparison = pd.DataFrame({

    "Feature Extraction":[
        "Bag of Words",
        "TF-IDF"
    ],

    "Accuracy":[
        bow_accuracy,
        tfidf_accuracy
    ]

})

print("\n")
print(comparison)

plt.figure(figsize=(6,5))

sns.barplot(
    data=comparison,
    x="Feature Extraction",
    y="Accuracy"
)

plt.title("Bag of Words vs TF-IDF")

plt.show()

# ============================================================
# Feature Importance
# ============================================================

print("\n")
print("="*60)
print("Random Forest Feature Importance")
print("="*60)

feature_names = tfidf.get_feature_names_out()

importances = models["Random Forest"].feature_importances_

importance_df = pd.DataFrame({

    "Word":feature_names,

    "Importance":importances

})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 20 Important Words\n")

print(importance_df.head(20))

plt.figure(figsize=(10,7))

sns.barplot(

    data=importance_df.head(20),

    x="Importance",

    y="Word"

)

plt.title("Top 20 Important Words")

plt.show()

# ============================================================
# Save Best Model
# ============================================================

best_model_name = results_df.sort_values(
    by="Accuracy",
    ascending=False
).iloc[0]["Model"]

best_model = models[best_model_name]

print("Best Model :", best_model_name)

joblib.dump(best_model,"fake_news_model.pkl")

joblib.dump(tfidf,"tfidf_vectorizer.pkl")

print("\nModel Saved Successfully")

# ============================================================
# Prediction Function
# ============================================================

print("\n")
print("="*60)
print("Prediction Function")
print("="*60)

def predict_news(news):

    news = clean_text(news)

    news_vector = tfidf.transform([news])

    prediction = best_model.predict(news_vector)[0]

    probabilities = best_model.predict_proba(news_vector)[0]

    print(f"\nFake Probability : {probabilities[0]*100:.2f}%")
    print(f"Real Probability : {probabilities[1]*100:.2f}%")

    if prediction == 1:
        return "REAL NEWS"
    else:
        return "FAKE NEWS"


# ============================================================
# Project Finished
# ============================================================

print("\n")
print("="*60)
print("="*60)

print("\nFiles Generated")

print("1. cleaned_fake_news.csv")
print("2. model_comparison.csv")
print("3. fake_news_model.pkl")
print("4. tfidf_vectorizer.pkl")

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve
)

plt.figure(figsize=(8,6))

for name, model in models.items():

    if hasattr(model, "predict_proba"):

        probs = model.predict_proba(X_test)[:,1]

        fpr, tpr, _ = roc_curve(y_test, probs)

        auc = roc_auc_score(y_test, probs)

        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

plt.figure(figsize=(8,6))

for name, model in models.items():

    if hasattr(model,"predict_proba"):

        probs=model.predict_proba(X_test)[:,1]

        precision, recall, _ = precision_recall_curve(y_test, probs)

        plt.plot(recall, precision, label=name)

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("Precision-Recall Curve")

plt.legend()

plt.show()

from sklearn.model_selection import cross_val_score

print("="*50)

# Create TF-IDF features for the entire dataset (only for cross-validation)
tfidf_cv = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2)
)

X_cv = tfidf_cv.fit_transform(df["clean_text"])

for name, model in models.items():

    score = cross_val_score(

        model,

        X_cv,

        y,

        cv=3,

        scoring="accuracy"

    )

    print(name)

    print("Fold Scores :", score)

    print("Average :", score.mean())

    print()

from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators":[100],
    "max_depth":[20,None],
    "min_samples_split":[2]
}

grid = GridSearchCV(

    RandomForestClassifier(
    random_state=42,
    n_jobs=-1
),

    param_grid,

    cv=3,

    scoring="accuracy",

    n_jobs=-1

)

grid.fit(X_train,y_train)

print("Best Parameters")

print(grid.best_params_)

print("Best Accuracy")

print(grid.best_score_)
plt.close('all')
while True:

    news = input("\nEnter News Article (type 'exit' to quit): ")

    if news.lower() == "exit":
        print("\nExiting Prediction System...")
        break

    print("\nPrediction:")
    print(predict_news(news))

import os

os.makedirs("outputs",exist_ok=True)

results_df.to_csv(

    "outputs/model_results.csv",

    index=False

)

importance_df.to_csv(

    "outputs/feature_importance.csv",

    index=False

)

joblib.dump(

    best_model,

    "outputs/fake_news_model.pkl"

)

joblib.dump(

    tfidf,

    "outputs/tfidf.pkl"

)