from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import fitz
from app.schemas import EmailRequest, EmailResponse
from app.nlp.preprocess import preprocess_text
from app.nlp.classifier import classify_email
from app.nlp.responder import generate_response

app = FastAPI(title="Email Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def process_logic(text: str):
    email_limpo = preprocess_text(text)
    classificacao = classify_email(email_limpo)
    resposta = generate_response(text, classificacao["categoria"])

    return {
        "categoria": classificacao["categoria"],
        "confianca": classificacao["confianca"],
        "resposta_sugerida": resposta
    }

@app.post("/analyze-email", response_model=EmailResponse)
def analyze_email(request: EmailRequest):
    return process_logic(request.email)

@app.post("/analyze-file", response_model=EmailResponse)
async def analyze_file(file: UploadFile = File(...)):
    content = ""
    
    if file.filename.endswith(".txt"):
        content = (await file.read()).decode("utf-8")
    elif file.filename.endswith(".pdf"):
        pdf_data = await file.read()
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        for page in doc:
            content += page.get_text()
    else:
        raise HTTPException(status_code=400, detail="Formato não suportado")

    if not content.strip():
        raise HTTPException(status_code=400, detail="Arquivo vazio")

    return process_logic(content)