"""Exploratory data analysis: class balance, review length, word clouds."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

from src.preprocess import clean_text

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "reports"


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_parquet(RAW / "train-00000-of-00001.parquet")
    df["sentiment"] = df["label"].map({0: "Negative", 1: "Positive"})
    df["words"] = df["text"].str.split().str.len()

    print("Shape:", df.shape)
    print(df["sentiment"].value_counts())
    print(df["words"].describe())

    # 1. Class balance
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x="sentiment", hue="sentiment", palette=["#d9534f", "#5cb85c"], legend=False)
    plt.title("Class balance (training set)")
    plt.tight_layout()
    plt.savefig(OUT / "class_balance.png", dpi=150)
    plt.close()

    # 2. Review length
    plt.figure(figsize=(7, 4))
    sns.histplot(data=df, x="words", hue="sentiment", bins=60, palette=["#d9534f", "#5cb85c"], element="step")
    plt.xlim(0, 1200)
    plt.title("Review length (words)")
    plt.tight_layout()
    plt.savefig(OUT / "review_length.png", dpi=150)
    plt.close()

    # 3. Word clouds on a sample (fast) of cleaned text
    sample = df.sample(5000, random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (label, colour) in zip(axes, [("Positive", "Greens"), ("Negative", "Reds")]):
        text = " ".join(sample.loc[sample["sentiment"] == label, "text"].map(clean_text))
        wc = WordCloud(width=800, height=450, background_color="white", colormap=colour, max_words=120).generate(text)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"{label} reviews")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT / "wordclouds.png", dpi=150)
    plt.close()
    print("Saved plots to", OUT)


if __name__ == "__main__":
    main()
