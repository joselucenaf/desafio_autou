from transformers import pipeline


classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

LABELS_MAP = {
    "solicitação de serviço, problema financeiro, boleto, segunda via, suporte técnico ou dúvida": "Produtivo",
    "apenas agradecimento, apenas saudação, elogio sem pedido ou conversa informal": "Improdutivo"
}

def classify_email(text: str) -> dict:

    result = classifier(text, list(LABELS_MAP.keys()))

    best_label = result["labels"][0]
    score = result["scores"][0]
    
    categoria_final = LABELS_MAP[best_label]

    return {
        "categoria": categoria_final,
        "confianca": round(score, 2)
    }