from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
from dotenv import load_dotenv
import os
import numpy as np
from rate_limiter import RateLimiter
import requests
import time
load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

rate_limiter = RateLimiter(requests_per_minute=5, requests_per_second=2)

def get_question_embedding(input_text:str, max_retries: int = 3) -> list[float]:
    
    for attempt in range(max_retries):
        try:
            rate_limiter.wait_if_needed()
            url = f'{os.getenv("OPENAI_PROXY_BASE_URL")}/embeddings'
            headers = {
                'Authorization': f'Bearer {os.getenv("OPENAI_PROXY_API_KEY")}',
                'Content-Type': 'application/json'
            }
            payload = {
                "input": input_text,
                "model": "text-embedding-3-small"
            }
            response = requests.post(url, headers=headers, json=payload, timeout=10)
        
            return response.json()["data"][0]["embedding"]

        except Exception as e:
            if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                #Exponential backoff for rate limit errors
                wait_time = 2 ** attempt
                print(f'Rate limit exceeded, retrying in {wait_time} seconds...')
                time.sleep(wait_time)
            elif attempt == max_retries - 1:
                print(f'Failed to get embeddings after {max_retries} attempts: {e}')
                raise
            else:
                print(f'attempt {attempt + 1} failed with error: {e}, retrying...')
                time.sleep(1)

    raise Exception("Max retries excedded")


system_prompt = ""
def get_llm_response(context:str, question:str):
    content = {"context":context, "userQuestion":question}
    url = os.getenv("OPENAI_PROXY_BASE_URL") + "/responses"
    API_KEY = os.getenv("OPENAI_PROXY_API_KEY")
    headers = {
        'Authorization':f'Bearer {API_KEY}',
        'Content-Type':"application/json"
    }
    payload = {
        "model":"gpt-5-nano",
        "reasoning":{"effort":"low"},
        "instructions":system_prompt,
        "input":content
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        return response.json()[0]["content"][0]["text"]

    except Exception as e:
        error_message = f'error occured while getting llm response. please try again. Error:{str(e)}'
        print(error_message)
        raise Exception(error_message)

def load_files():
    entries = os.listdir("embeddings/")
    embedding_files = [f for f in entries if os.path.isfile(os.path.join("embeddings/", f))]
    embeddings = []
    chunks = []
    for f in embedding_files:
        data = np.load(f)
        embeddings.extend(data['embeddings'])
        chunks.extend(data['chunks'])
    return embeddings, chunks

def answer(question:str):
    question_embedding = get_question_embedding(question)
    embeddings, chunks = load_files()
    similarities = np.dot(embeddings, question_embedding)/(np.linalg.norm(embeddings, axis=1) * np.linalg.norm(question_embedding))
    top_indices = np.argsort(similarities)[-10:][::-1]
    top_chunks = [chunks[i] for i in top_indices]
    context = '\n\n'.join(top_chunks)
    answer = get_llm_response(context, question)
    return answer


@app.post('/api/naval/talk')
async def talk(input:str):
    answer = answer(input)
    return {"message":answer}