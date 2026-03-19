from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import nltk

# Importações do Sumy (TextRank)
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.utils import get_stop_words

# Download dos recursos necessários do NLTK (executa uma vez no deploy)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

app = FastAPI(title="API de Resumo TextRank")

# Configuração de CORS para seu site conseguir acessar a API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # No Render, você pode trocar "*" pela URL do seu site
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SummarizeRequest(BaseModel):
    text: str
    num_sentences: int = 3
    language: str = "portuguese"

@app.post("/summarize")
async def summarize_text(request: SummarizeRequest) -> Dict[str, object]:
    if not request.text.strip():
        return {"summary": "", "stats": {"original_words": 0, "summary_words": 0}}

    # Processamento com Sumy (TextRank)
    parser = PlaintextParser.from_string(request.text, Tokenizer(request.language))
    summarizer = TextRankSummarizer()
    summarizer.stop_words = get_stop_words(request.language)

    # Gera o resumo com base no número de frases solicitado
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