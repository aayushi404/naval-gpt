import os
import requests
import time
from dotenv import load_dotenv
from ..rate_limiter import RateLimiter
from ..utils.load_files import load_files
import numpy as np

load_dotenv()
rate_limiter = RateLimiter(requests_per_minute=5, requests_per_second=2)

def get_question_embedding(input_text:str, max_retries: int = 3) -> list[float]:
    
    for attempt in range(max_retries):
        try:
            rate_limiter.wait_if_needed()
            url = f'{os.getenv("OPENAI_BASE_URL")}/embeddings'
            headers = {
                'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}',
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

def get_chunks(question:str):
    print(question)
    print("loading embedding files")
    embeddings, chunks = load_files()
    print("getting question embeddings")
    question_embedding = get_question_embedding(question)
    embeddings, chunks = load_files()
    similarities = np.dot(embeddings, question_embedding)/(np.linalg.norm(embeddings, axis=1) * np.linalg.norm(question_embedding))
    top_indices = np.argsort(similarities)[-10:][::-1]
    top_chunks = [chunks[i] for i in top_indices]

    return top_chunks
   