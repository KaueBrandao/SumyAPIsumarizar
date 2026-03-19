import os
import nltk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict

# --- CONFIGURAÇÃO NLTK PARA PRODUÇÃO ---
# Garante que o download seja feito no diretório do projeto no Render
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

# Adiciona o caminho ao NLTK para ele não se perder
nltk.data.path.append(nltk_data_path)

# Realiza os downloads necessários
nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
nltk.download('punkt_tab', download_dir=nltk_data_path, quiet=True)

# Só importamos o Sumy APÓS configurar o NLTK
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.utils import get_stop_words
# ---------------------------------------

app = FastAPI(title="API de Resumo TextRank")

# Configuração de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SummarizeRequest(BaseModel):
    text: str
    num_sentences: int = 3
    language: str = "portuguese"

@app.get("/")
def home():
    return {"status": "online", "message": "Use o endpoint /summarize via POST"}

@app.post("/summarize")
async def summarize_text(request: SummarizeRequest) -> Dict[str, object]:
    if not request.text.strip():
        return {"summary": "", "stats": {"original_words": 0, "summary_words": 0}}

    try:
        # Processamento com Sumy (TextRank)
        parser = PlaintextParser.from_string(request.text, Tokenizer(request.language))
        summarizer = TextRankSummarizer()
        summarizer.stop_words = get_stop_words(request.language)

        # Gera o resumo
        resumo_frases = summarizer(parser.document, request.num_sentences)
        resumo_texto = " ".join(str(frase).strip() for frase in resumo_frases)

        return {
            "summary": resumo_texto,
            "stats": {
                "original_words": len(request.text.split()),
                "summary_words": len(resumo_texto.split()),
                "sentences_requested": request.num_sentences
            }
        }
    except Exception as e:
        return {"error": str(e), "message": "Erro interno ao processar o resumo"}
