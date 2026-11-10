"""Streamlit demo: type a review, get its sentiment."""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so the pickled pipeline can import src.preprocess

import joblib
import streamlit as st

st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎬")


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "sentiment_model.joblib")


def predict(model, text: str):
    """Return (label, confidence in 0-1)."""
    if hasattr(model, "predict_proba"):
        p_pos = float(model.predict_proba([text])[0][1])
    else:  # e.g. LinearSVC: squash the margin into a pseudo-probability
        p_pos = 1 / (1 + math.exp(-2 * float(model.decision_function([text])[0])))
    return ("Positive" if p_pos >= 0.5 else "Negative"), max(p_pos, 1 - p_pos)


st.title("🎬 Movie Review Sentiment Analyzer")
st.caption("TF-IDF + linear model trained on 25,000 IMDB reviews.")

model = load_model()
meta_path = ROOT / "models" / "metadata.json"
if meta_path.exists():
    meta = json.loads(meta_path.read_text())
    st.sidebar.metric("Model", meta["model"])
    st.sidebar.metric("Test accuracy", f"{meta['test_accuracy']:.1%}")
st.sidebar.info("Confidence is a rough estimate, not a calibrated probability. "
                "Sarcasm and mixed reviews can fool the model.")

EXAMPLES = {
    "Positive example": "An absolute masterpiece. Brilliant acting, a gripping story, and a score that stays with you.",
    "Negative example": "Boring, predictable and far too long. I wanted my two hours back.",
    "Tricky (negation)": "This movie is not good at all, I would never recommend it.",
}
choice = st.selectbox("Try an example or write your own:", ["(write my own)"] + list(EXAMPLES))
default = EXAMPLES.get(choice, "")
review = st.text_area("Review text", value=default, height=160, key=f"review_{choice}")

if st.button("Analyze", type="primary"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        label, conf = predict(model, review)
        if label == "Positive":
            st.success(f"😊 Positive  ·  confidence {conf:.0%}")
        else:
            st.error(f"😞 Negative  ·  confidence {conf:.0%}")
        st.progress(conf)
