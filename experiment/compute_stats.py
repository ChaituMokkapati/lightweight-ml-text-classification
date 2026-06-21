"""Recompute per-run metrics and Wilcoxon tests from benchmark logic."""

import io
import json
import re
import time
import zipfile
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

RANDOM_SEED = 42
TEST_SIZE = 0.2
N_RUNS = 3
AG_NEWS_SUBSET = 20000
MAX_FEATURES = 20000
NGRAM_RANGE = (1, 2)
MIN_DF = 2
OUT = Path(__file__).resolve().parents[1] / "results" / "content" / "results"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_sms_spam():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    with urllib.request.urlopen(url) as response:
        with zipfile.ZipFile(io.BytesIO(response.read())) as zf:
            with zf.open("SMSSpamCollection") as f:
                df = pd.read_csv(f, sep="\t", header=None, names=["label", "text"], encoding="latin-1")
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df["text"] = df["text"].apply(clean_text)
    return df


def load_ag_news(subset=AG_NEWS_SUBSET):
    import json as json_mod

    base = "https://huggingface.co/datasets/sh0416/ag_news/resolve/main"
    rows = []
    for split in ("train", "test"):
        req = urllib.request.Request(f"{base}/{split}.jsonl", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            for line in resp:
                item = json_mod.loads(line)
                rows.append(
                    {
                        "label": item["label"] - 1,
                        "text": clean_text(f"{item['title']} {item['description']}"),
                    }
                )
    df = pd.DataFrame(rows)
    if subset and subset < len(df):
        per_class = subset // df["label"].nunique()
        df = (
            df.groupby("label", group_keys=False)
            .sample(n=per_class, random_state=RANDOM_SEED)
            .reset_index(drop=True)
        )
    return df


MODELS = {
    "Logistic Regression": LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED),
    "Linear SVM": LinearSVC(C=1.0, random_state=RANDOM_SEED),
    "Naive Bayes": MultinomialNB(alpha=1.0),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_SEED,
        eval_metric="mlogloss", verbosity=0, n_jobs=-1,
    ),
}


def run_benchmark(df, dataset_name, average="binary"):
    all_runs = []
    X, y = df["text"].values, df["label"].values
    for model_name, clf in MODELS.items():
        for run in range(N_RUNS):
            seed = RANDOM_SEED + run
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
            )
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE,
                                          min_df=MIN_DF, stop_words="english")),
                ("clf", clf),
            ])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            all_runs.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Run": run + 1,
                "F1": f1_score(y_test, y_pred, average=average, zero_division=0),
            })
    return pd.DataFrame(all_runs)


def compare_top_models(runs_df, dataset_name, model_a, model_b):
    a = runs_df[(runs_df["Dataset"] == dataset_name) & (runs_df["Model"] == model_a)]["F1"].values
    b = runs_df[(runs_df["Dataset"] == dataset_name) & (runs_df["Model"] == model_b)]["F1"].values
    stat, p = wilcoxon(a, b)
    return {"dataset": dataset_name, "model_a": model_a, "model_b": model_b, "p_value": p, "stat": stat}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    sms_df = load_sms_spam()
    ag_df = load_ag_news()
    sms_runs = run_benchmark(sms_df, "SMS Spam", average="binary")
    ag_runs = run_benchmark(ag_df, "AG News", average="macro")
    all_runs = pd.concat([sms_runs, ag_runs], ignore_index=True)
    all_runs.to_csv(OUT / "all_runs.csv", index=False)

    tests = [
        compare_top_models(all_runs, "SMS Spam", "Linear SVM", "Random Forest"),
        compare_top_models(all_runs, "AG News", "Logistic Regression", "Linear SVM"),
    ]
    with open(OUT / "statistical_tests.json", "w") as f:
        json.dump(tests, f, indent=2)
    print(json.dumps(tests, indent=2))
