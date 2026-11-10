# Movie Review Sentiment Analysis

Classifies IMDB movie reviews as **positive** or **negative** using TF-IDF features and classical ML models,
with a Streamlit web app for live predictions.

## Results (IMDB official 25k test set)

| Model | Val Accuracy | Test Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Naive Bayes | 0.8804 | 0.8609 | 0.8828 | 0.8323 | 0.8568 |
| Logistic Regression | 0.8976 | 0.8930 | 0.8962 | 0.8890 | 0.8926 |
| **Linear SVM** (selected) | **0.8996** | **0.8933** | 0.8959 | 0.8900 | 0.8930 |

The best model is chosen on a **validation split** (10% of train), not on the test set, so the test score is an
honest estimate. The shipped model is then refit on all 25k training reviews (final test accuracy: **89.4%**).

## Project structure

```
src/preprocess.py   text cleaning (keeps negations like "not")
src/eda.py          class balance, review length, word clouds -> reports/
src/train.py        trains 3 models, compares, saves best -> models/
app/app.py          Streamlit demo
reports/            plots, model_comparison.csv, misclassified_examples.csv
data/raw/           IMDB parquet files (Hugging Face: stanfordnlp/imdb)
```

## How to run

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

python -m src.eda                 # optional: make EDA plots
python -m src.train               # train + save model
streamlit run app\app.py          # launch the web app
```

Dataset: download `train-00000-of-00001.parquet` and `test-00000-of-00001.parquet` from
https://huggingface.co/datasets/stanfordnlp/imdb/tree/main/plain_text into `data/raw/`.

## Key design choices

- **Negations are kept** in preprocessing. Standard stopword lists drop "not", which flips sentiment.
- **Unigrams + bigrams** let the model capture phrases such as "not good" and "waste of".
- **Model selection on validation data** to avoid test-set leakage.

## Known limitations

- **Sarcasm** ("Oh great, another wasted two hours. Wonderful.") is often misclassified, since TF-IDF is bag-of-words.
  See `reports/misclassified_examples.csv` for 40 real error cases.
- Trained on movie reviews, so accuracy drops on other domains (products, tweets).
- App confidence is a rough margin-based score, not a calibrated probability.

## Next steps

- Fine-tune **DistilBERT** (Hugging Face) for ~92-93% accuracy and better sarcasm handling.
- Try a product-review dataset (Amazon/Flipkart) to test domain transfer.
- Add 3-class sentiment (positive / neutral / negative).
