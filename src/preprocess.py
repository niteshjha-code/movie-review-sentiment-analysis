"""Text cleaning shared by training and the Streamlit app."""
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Negations flip sentiment ("not good"), so never treat them as stopwords.
NEGATIONS = {"not", "no", "nor", "never", "none", "nothing", "cannot", "neither", "nobody", "nowhere"}
STOPWORDS = frozenset(ENGLISH_STOP_WORDS) - NEGATIONS

_HTML = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_NON_LETTER = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip HTML/URLs/punctuation/numbers, keep negations, drop stopwords."""
    text = str(text).lower()
    text = _HTML.sub(" ", text)
    text = _URL.sub(" ", text)
    text = text.replace("n't", " not").replace("’t", " not")
    text = _NON_LETTER.sub(" ", text)
    tokens = [t for t in _SPACES.split(text) if t and t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)
