from rate_limiter import RateLimiter
import numpy as np
import sys
import os
from tqdm import tqdm
import requests
import time
from dotenv import load_dotenv

load_dotenv()

rate_limiter = RateLimiter(requests_per_minute=5, requests_per_second=1)

def getEmbedding(chunk:str, max_retries:int=3):
    
    for attempt in range(max_retries):
        try:
            rate_limiter.wait_if_needed()

            url = f'{os.getenv("OPENAI_PROXY_BASE_URL")}/embeddings'
            headers = {
                'Authorization': f'Bearer {os.getenv("OPENAI_PROXY_API_KEY")}',
                'Content-Type': 'application/json'
            }
            payload = {
                "input": chunk,
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

def run():
    chunk_file = sys.argv[1]
    if not chunk_file:
        print("please provide chunk file!")
        return
    chunks_file_path = "chunks/" + chunk_file
    if not os.path.exists(chunks_file_path):
        print("there is no chunk file!")
        return

    remaining_chunk = np.load(f'chunks/{chunk_file}').tolist()
    embedding_file_path = "embeddings/" + chunk_file.replace(".npy", ".npz")
    if os.path.exists(embedding_file_path):
        embeddings_data = np.load(embedding_file_path)
        all_embeddings = embeddings_data["embeddings"].tolist()
        processed_chunks = embeddings_data["chunks"].tolist()
    else:
        all_embeddings = []
        processed_chunks = []

    chunks_to_process = sys.argv[2]
    if not chunks_to_process:
        print("please proved the number of chunks you want to process!")

    chunks_to_process = min(int(chunks_to_process), len(remaining_chunk))

    with tqdm(total=chunks_to_process, desc="Processing chunks") as pbar:
        chunks = remaining_chunk[:chunks_to_process]
        for i, chunk in enumerate(chunks):
            try:
                embeddings = getEmbedding(chunk)
                all_embeddings.append(embeddings)
                processed_chunks.append(chunk)
                pbar.set_postfix({"chunks_processed":i, "target":chunks_to_process,"total":len(remaining_chunk) - i})
                pbar.update(1)
            except Exception as e:
                print(f'error occured while processing chunk: {chunk}, so skipping it. Error: {str(e)}')
                remaining_chunk.append(chunk)
                pbar.update(1)
                continue

        np.save(chunks_file_path, remaining_chunk[chunks_to_process:])
        np.savez(embedding_file_path, embeddings=all_embeddings, chunks=processed_chunks)
        print(f'{len(remaining_chunk) - chunks_to_process} remaining to process')
if __name__ == "__main__":
    load_dotenv()
    run()