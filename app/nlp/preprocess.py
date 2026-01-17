import re

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zà-ú\s]", " ", text)
    tokens = text.split()
    return " ".join(tokens)