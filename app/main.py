from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import fitz 
import os
from app.schemas import EmailRequest, EmailResponse
from app.nlp.preprocess import preprocess_text
from app.nlp.classifier import classify_email
from app.nlp.responder import generate_response

app = FastAPI(title="Email Classification API", docs_url="/api/docs")

#Config. do CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir arquivos estáticos do frontend
FRONTEND_PATH = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(FRONTEND_PATH):
    app.mount("/static", StaticFiles(directory=FRONTEND_PATH), name="static")

def process_logic(text: str):
    classificacao = classify_email(text) 
    
    resposta = generate_response(text, classificacao["categoria"])

    return {
        "categoria": classificacao["categoria"],
        "confianca": classificacao["confianca"],
        "resposta_sugerida": resposta
    }

@app.get("/")
async def serve_frontend():
    if os.path.exists(os.path.join(FRONTEND_PATH, "index.html")):
        return FileResponse(os.path.join(FRONTEND_PATH, "index.html"))
    return {"message": "Backend está rodando, frontend não encontrado"}

@app.post("/analyze-email", response_model=EmailResponse)
def analyze_email(request: EmailRequest):
    return process_logic(request.email)

@app.post("/analyze-file", response_model=EmailResponse)
async def analyze_file(file: UploadFile = File(...)):
    content = ""
    file_bytes = await file.read()
    print(f"\n--- DEBUG UPLOAD ---")
    print(f"Arquivo recebido: {file.filename}")
    print(f"Tamanho lido: {len(file_bytes)} bytes")

    try:
        if file.filename.endswith(".txt"):
            content = file_bytes.decode("utf-8")
        
        elif file.filename.endswith(".pdf"):
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            print(f"Número de páginas no PDF: {len(doc)}")
            
            for page_num, page in enumerate(doc):
                text_extracted = page.get_text()
                print(f"Página {page_num + 1}: {len(text_extracted)} caracteres extraídos.")
                content += text_extracted
            doc.close()
        
        else:
            raise HTTPException(status_code=400, detail="Formato não suportado. Use .txt ou .pdf")
        
    except Exception as e:
        print(f"ERRO NA LEITURA: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao processar arquivo: {str(e)}")

    if not content.strip():
        print("RESULTADO: O conteúdo extraído está vazio.")
        print("DICA: Se for um PDF, verifique se não é uma imagem/escaneado sem OCR.")
        print("--------------------\n")
        raise HTTPException(status_code=400, detail="Arquivo vazio ou sem texto legível")

    print(f"SUCESSO: {len(content)} caracteres enviados para processamento.")
    print("--------------------\n")
    
    return process_logic(content)

#Render
@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "Desafio Autou NLP API"}