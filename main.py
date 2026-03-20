from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# 1. Configuração do Modelo e Tokenizer (Carregados globalmente para eficiência)
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

app = FastAPI(title="Summarization API com T5")

# 2. Definição do esquema de dados de entrada
class TextRequest(BaseModel):
    text: str

# 3. Endpoint de Resumo
@app.post("/summarize")
async def summarize_text(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="O texto não pode estar vazio.")

    try:
        # Prepara o input com o prefixo do T5
        input_text = "summarize: " + request.text
        
        # Tokenização
        inputs = tokenizer.encode(
            input_text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        )

        # Geração do resumo
        outputs = model.generate(
            inputs,
            max_length=150,
            min_length=40,
            length_penalty=1.0,
            num_beams=4,
            early_stopping=True
        )

        # Decodificação
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            "original_length": len(request.text),
            "summary": summary,
            "summary_length": len(summary)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Para rodar: uvicorn nome_do_arquivo:app --reload
