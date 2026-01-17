from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

LABELS_MAP = {
    "solicitação de cliente ou dúvida técnica": "Produtivo",
    "mensagem irrelevante, agradecimento ou saudação": "Improdutivo"
}

def classify_email(text: str) -> dict:
    result = classifier(text, list(LABELS_MAP.keys()))
    best_label = result["labels"][0]
    categoria_final = LABELS_MAP[best_label]

    return {
        "categoria": categoria_final,
        "confianca": round(result["scores"][0], 2)
    }