
import os
import re
import gzip
import time
import requests
import certifi
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# TASK 3: CLASSICAL NLP BASELINE AND WORD EMBEDDINGS
# Compatible with Python 3.14; Gensim is not required.
# ============================================================

RANDOM_STATE = 42
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

GLOVE_PATH = os.path.join(
    ARTIFACT_DIR, "glove-wiki-gigaword-50.gz"
)

GLOVE_URL = (
    "https://github.com/RaRe-Technologies/gensim-data/"
    "releases/download/glove-wiki-gigaword-50/"
    "glove-wiki-gigaword-50.gz"
)

# ============================================================
# 1. CREATE A LABELED DATASET (360 SHORT TEXTS)
# ============================================================

def build_dataset():
    templates = {
        "positive": [
            "I love the {thing}, it is excellent",
            "Great {thing}, very happy with the service",
            "The {thing} works perfectly and the team was helpful",
            "Amazing experience with {thing}",
            "Thank you, the {thing} is fast and reliable",
            "Very satisfied with the {thing}",
            "Wonderful support and a great {thing}",
            "The {thing} exceeded my expectations",
        ],
        "negative": [
            "I hate the {thing}, it is terrible",
            "Poor {thing}, very unhappy with the service",
            "The {thing} does not work and support was unhelpful",
            "Awful experience with {thing}",
            "The {thing} is slow and unreliable",
            "Very disappointed with the {thing}",
            "Bad support and a broken {thing}",
            "The {thing} failed and wasted my time",
        ],
        "neutral": [
            "I have a question about the {thing}",
            "Please explain how the {thing} works",
            "I would like information about the {thing}",
            "The {thing} was delivered today",
            "Can you update me about the {thing}",
            "I am checking the status of the {thing}",
            "Please confirm the details for the {thing}",
            "I need help understanding the {thing}",
        ],
    }

    things = [
        "app", "order", "payment", "delivery", "account",
        "website", "product", "subscription", "refund",
        "booking", "service", "ticket", "update", "invoice",
        "login",
    ]

    suffixes = [
        "", " today", " please", " for my account", " again"
    ]

    rows = []

    for label, phrases in templates.items():
        for i in range(120):
            phrase = phrases[i % len(phrases)].format(
                thing=things[(i * 7 + len(label)) % len(things)]
            )
            phrase += suffixes[i % len(suffixes)]
            rows.append({"text": phrase, "label": label})

    return pd.DataFrame(rows).sample(
        frac=1, random_state=RANDOM_STATE
    ).reset_index(drop=True)


# ============================================================
# 2. CLEAN TEXT
# ============================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df = build_dataset()
df["text"] = df["text"].apply(clean_text)

df.to_csv(
    os.path.join(ARTIFACT_DIR, "customer_feedback.csv"),
    index=False,
)

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
print("Total texts:", len(df))
print("\nClass distribution:")
print(df["label"].value_counts())
print("\nSample records:")
print(df.head())

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df["label"],
)

# ============================================================
# 3. TF-IDF + LOGISTIC REGRESSION
# ============================================================

tfidf_model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    ),
])

start = time.perf_counter()
tfidf_model.fit(X_train, y_train)
tfidf_train_time = time.perf_counter() - start

start = time.perf_counter()
tfidf_predictions = tfidf_model.predict(X_test)
tfidf_inference_time = (
    time.perf_counter() - start
) / len(X_test)

tfidf_accuracy = accuracy_score(
    y_test, tfidf_predictions
)
tfidf_macro_f1 = f1_score(
    y_test, tfidf_predictions, average="macro"
)

print("\n" + "=" * 60)
print("TF-IDF BASELINE RESULTS")
print("=" * 60)
print(f"Accuracy: {tfidf_accuracy:.4f}")
print(f"Macro F1: {tfidf_macro_f1:.4f}")
print(f"Training time: {tfidf_train_time:.4f} seconds")
print(
    "Inference time per text: "
    f"{tfidf_inference_time:.6f} seconds"
)

print("\nConfusion matrix:")
print("Class order:", list(tfidf_model.classes_))
print(confusion_matrix(
    y_test, tfidf_predictions,
    labels=tfidf_model.classes_,
))

print("\nClassification report:")
print(classification_report(
    y_test, tfidf_predictions, zero_division=0
))

# ============================================================
# 4. PRINT TOP 10 IMPORTANT WORDS PER CLASS
# ============================================================

vectorizer = tfidf_model.named_steps["tfidf"]
classifier = tfidf_model.named_steps["classifier"]
feature_names = np.array(vectorizer.get_feature_names_out())

print("\n" + "=" * 60)
print("TOP 10 IMPORTANT WORDS FOR EACH CLASS")
print("=" * 60)

for index, label in enumerate(classifier.classes_):
    top_indices = np.argsort(
        classifier.coef_[index]
    )[-10:][::-1]

    print(f"\n{label.upper()}:")
    print(", ".join(feature_names[top_indices]))

# Save TF-IDF model and metrics.
import joblib

joblib.dump(
    tfidf_model,
    os.path.join(ARTIFACT_DIR, "tfidf_pipeline.joblib"),
)

tfidf_results = {
    "model": "TF-IDF + Logistic Regression",
    "accuracy": tfidf_accuracy,
    "macro_f1": tfidf_macro_f1,
    "training_time_seconds": tfidf_train_time,
    "inference_time_seconds_per_text": tfidf_inference_time,
}

# ============================================================
# 5. DOWNLOAD GLOVE VECTORS SECURELY
# ============================================================

def download_glove():
    if os.path.exists(GLOVE_PATH) and os.path.getsize(GLOVE_PATH) > 0:
        print("\nGloVe archive already exists. Using local file.")
        return

    print("\nDownloading pretrained GloVe vectors...")
    print("The compressed file is approximately 66 MB.")

    temporary_path = GLOVE_PATH + ".download"

    try:
        with requests.get(
            GLOVE_URL,
            stream=True,
            timeout=(30, 120),
            verify=certifi.where(),
            allow_redirects=True,
        ) as response:
            response.raise_for_status()

            with open(temporary_path, "wb") as file:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        file.write(chunk)

        if os.path.getsize(temporary_path) == 0:
            raise RuntimeError("Downloaded file is empty.")

        os.replace(temporary_path, GLOVE_PATH)
        print("GloVe downloaded successfully.")

    except Exception:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)

        print("\nGloVe download failed.")
        print("Check your internet connection and SSL certificates.")
        print("You can download the file manually from:")
        print(
            "https://github.com/RaRe-Technologies/"
            "gensim-data/releases/tag/glove-wiki-gigaword-50"
        )
        raise


# ============================================================
# 6. LOAD PRETRAINED GLOVE EMBEDDINGS
# ============================================================

download_glove()

glove = {}

print("\nLoading GloVe vocabulary...")

with gzip.open(
    GLOVE_PATH, "rt", encoding="utf-8"
) as file:
    for line in file:
        parts = line.rstrip().split()

        if len(parts) != 51:
            continue

        word = parts[0]

        try:
            vector = np.asarray(
                parts[1:], dtype=np.float32
            )
        except ValueError:
            continue

        glove[word] = vector

print("Loaded GloVe words:", len(glove))
print("Embedding dimensions: 50")

# ============================================================
# 7. FIND FIVE NEAREST WORDS FOR TWO WORDS
# ============================================================

def nearest_words(word, top_n=5):
    if word not in glove:
        print(f"\n'{word}' is not in the GloVe vocabulary.")
        return

    target = glove[word]
    target_norm = np.linalg.norm(target)
    results = []

    for candidate, vector in glove.items():
        if candidate == word:
            continue

        vector_norm = np.linalg.norm(vector)

        if vector_norm == 0 or target_norm == 0:
            continue

        similarity = np.dot(
            target, vector
        ) / (target_norm * vector_norm)

        results.append((candidate, float(similarity)))

    results.sort(key=lambda item: item[1], reverse=True)

    print(f"\nFive nearest words to '{word}':")
    for candidate, score in results[:top_n]:
        print(f"{candidate}: {score:.4f}")


nearest_words("king")
nearest_words("computer")

# ============================================================
# 8. WORD ANALOGY: KING - MAN + WOMAN
# ============================================================

print("\n" + "=" * 60)
print("WORD ANALOGY")
print("=" * 60)

if all(word in glove for word in ["king", "man", "woman"]):
    analogy_vector = (
        glove["king"] - glove["man"] + glove["woman"]
    )

    analogy_norm = np.linalg.norm(analogy_vector)
    results = []

    if analogy_norm > 0:
        for word, vector in glove.items():
            if word in {"king", "man", "woman"}:
                continue

            norm = np.linalg.norm(vector)

            if norm == 0:
                continue

            similarity = np.dot(
                analogy_vector, vector
            ) / (analogy_norm * norm)

            results.append((word, float(similarity)))

        results.sort(
            key=lambda item: item[1], reverse=True
        )

        print("king - man + woman ->", results[:5])
else:
    print("One or more analogy words were not found.")

# ============================================================
# 9. AVERAGE WORD VECTORS INTO SENTENCE VECTORS
# ============================================================

def sentence_vector(sentence):
    words = clean_text(sentence).split()

    vectors = [
        glove[word]
        for word in words
        if word in glove
    ]

    if not vectors:
        return np.zeros(50, dtype=np.float32)

    return np.mean(vectors, axis=0)


X_train_embeddings = np.vstack([
    sentence_vector(text) for text in X_train
])

X_test_embeddings = np.vstack([
    sentence_vector(text) for text in X_test
])

print("\nSentence-vector shapes:")
print("Training:", X_train_embeddings.shape)
print("Testing:", X_test_embeddings.shape)

# ============================================================
# 10. TRAIN CLASSIFIER ON AVERAGED GLOVE SENTENCE VECTORS
# ============================================================

embedding_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)

start = time.perf_counter()
embedding_model.fit(X_train_embeddings, y_train)
glove_train_time = time.perf_counter() - start

start = time.perf_counter()
glove_predictions = embedding_model.predict(
    X_test_embeddings
)
glove_inference_time = (
    time.perf_counter() - start
) / len(X_test)

glove_accuracy = accuracy_score(
    y_test, glove_predictions
)
glove_macro_f1 = f1_score(
    y_test, glove_predictions, average="macro"
)

print("\n" + "=" * 60)
print("GLOVE SENTENCE-VECTOR RESULTS")
print("=" * 60)
print(f"Accuracy: {glove_accuracy:.4f}")
print(f"Macro F1: {glove_macro_f1:.4f}")
print(f"Training time: {glove_train_time:.4f} seconds")
print(
    "Inference time per text: "
    f"{glove_inference_time:.6f} seconds"
)

# ============================================================
# 11. COMPARE TF-IDF AND GLOVE
# ============================================================

glove_results = {
    "model": "GloVe averaged vectors + Logistic Regression",
    "accuracy": glove_accuracy,
    "macro_f1": glove_macro_f1,
    "training_time_seconds": glove_train_time,
    "inference_time_seconds_per_text": glove_inference_time,
}

comparison = pd.DataFrame([
    tfidf_results,
    glove_results,
])

comparison.to_csv(
    os.path.join(ARTIFACT_DIR, "comparison.csv"),
    index=False,
)

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)
print(comparison.to_string(index=False))

print("\nResults saved to:", ARTIFACT_DIR)
print("Task 3 completed successfully.")
