import os
import nltk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict

# --- CORREÇÃO DE AMBIENTE RENDER ---
# Forçamos o NLTK a usar uma pasta local para evitar erros de permissão
nltk_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_path, exist_ok=True)
nltk.data.path.append(nltk_path)

# Downloads essenciais (adicionado 'stopwords' que o Sumy exige)
nltk.download('punkt', download_dir=nltk_path, quiet=True)
nltk.download('punkt_tab', download_dir=nltk_path, quiet=True)
nltk.download('stopwords', download_dir=nltk_path, quiet=True)

# Importações após o setup do NLTK
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.utils import get_stop_words

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SummarizeRequest(BaseModel):
    text: str
    num_sentences: int = 3
    language: str = "portuguese"

@app.get("/")
def home():
    return {"status": "online", "docs": "/docs"}

@app.post("/summarize")
async def summarize_text(request: SummarizeRequest):
    if not request.text.strip():
        return {"summary": "Texto vazio", "stats": {}}

    try:
        # Criamos o parser e o summarizer dentro do try para capturar o erro exato
        parser = PlaintextParser.from_string(request.text, Tokenizer(request.language))
        summarizer = TextRankSummarizer()
        summarizer.stop_words = get_stop_words(request.language)

        resumo_frases = summarizer(parser.document, request.num_sentences)
        resumo_texto = " ".join(str(frase).strip() for frase in resumo_frases)

        return {
            "summary": resumo_texto,
            "original_len": len(request.text.split()),
            "summary_len": len(resumo_texto.split())
        }
    except Exception as e:
        # Se der erro 500, agora ele retornará o TEXTO do erro para sabermos o que é
        return {"error": str(e), "type": str(type(e))}
