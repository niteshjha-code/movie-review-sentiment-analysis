"""Train and compare sentiment models on IMDB, save the best one.

Model selection uses a validation split carved out of the training data.
The official test set is only used for the final report.
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.preprocess import clean_text

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"


def make_pipeline(clf):
    vec = TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), min_df=3, max_df=0.9,
                          sublinear_tf=True, max_features=300_000)
    return Pipeline([("tfidf", vec), ("clf", clf)])


def main():
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    train = pd.read_parquet(RAW / "train-00000-of-00001.parquet")
    test = pd.read_parquet(RAW / "test-00000-of-00001.parquet")
    X_tr, X_val, y_tr, y_val = train_test_split(
        train["text"], train["label"], test_size=0.1, stratify=train["label"], random_state=42)

    candidates = {
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(C=10, max_iter=2000),
        "Linear SVM": LinearSVC(C=0.5),
    }

    rows, fitted = [], {}
    for name, clf in candidates.items():
        print(f"Training {name} ...", flush=True)
        pipe = make_pipeline(clf).fit(X_tr, y_tr)
        val_acc = accuracy_score(y_val, pipe.predict(X_val))
        pred = pipe.predict(test["text"])
        rows.append({
            "Model": name,
            "Val Accuracy": round(val_acc, 4),
            "Test Accuracy": round(accuracy_score(test["label"], pred), 4),
            "Precision": round(precision_score(test["label"], pred), 4),
            "Recall": round(recall_score(test["label"], pred), 4),
            "F1": round(f1_score(test["label"], pred), 4),
        })
        fitted[name] = (pipe, pred)
        print(rows[-1], flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(REPORTS / "model_comparison.csv", index=False)
    print("\n", results.to_string(index=False))

    # Pick best by VALIDATION accuracy (not test) to avoid leakage.
    best_name = results.sort_values("Val Accuracy", ascending=False).iloc[0]["Model"]
    best_pipe, best_pred = fitted[best_name]
    print(f"\nBest model (by validation): {best_name}")
    print(classification_report(test["label"], best_pred, target_names=["Negative", "Positive"]))

    # Refit the winner on all 25k training reviews for the shipped model.
    final = make_pipeline(candidates[best_name].__class__(**candidates[best_name].get_params()))
    final.fit(train["text"], train["label"])
    joblib.dump(final, MODELS / "sentiment_model.joblib", compress=3)
    final_pred = final.predict(test["text"])
    final_acc = accuracy_score(test["label"], final_pred)
    print(f"Final model test accuracy: {final_acc:.4f}")
    (MODELS / "metadata.json").write_text(json.dumps(
        {"model": best_name, "test_accuracy": round(final_acc, 4)}, indent=2))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(test["label"], final_pred,
                                            display_labels=["Negative", "Positive"], cmap="Blues", ax=ax)
    ax.set_title(f"{best_name} - confusion matrix (test)")
    plt.tight_layout()
    plt.savefig(REPORTS / "confusion_matrix.png", dpi=150)
    plt.close()

    # Misclassified reviews for error analysis
    wrong = test.assign(predicted=final_pred)
    wrong = wrong[wrong["label"] != wrong["predicted"]].copy()
    wrong["text"] = wrong["text"].str.slice(0, 400)
    wrong.sample(40, random_state=1).to_csv(REPORTS / "misclassified_examples.csv", index=False)

    # Most influential words (linear models only)
    clf = final.named_steps["clf"]
    if hasattr(clf, "coef_"):
        vocab = final.named_steps["tfidf"].get_feature_names_out()
        coefs = pd.Series(clf.coef_.ravel(), index=vocab).sort_values()
        top = pd.concat([coefs.head(15), coefs.tail(15)])
        plt.figure(figsize=(7, 7))
        top.plot.barh(color=["#d9534f"] * 15 + ["#5cb85c"] * 15)
        plt.title("Top words driving the prediction")
        plt.tight_layout()
        plt.savefig(REPORTS / "top_words.png", dpi=150)
        plt.close()
    print("Done. Model saved to", MODELS / "sentiment_model.joblib")


if __name__ == "__main__":
    main()
