import re
import nltk
from nltk.corpus import stopwords


nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('portuguese'))

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zà-ú\s]", " ", text)
    tokens = text.split()
 
    tokens = [t for t in tokens if t not in STOP_WORDS]
    return " ".join(tokens)