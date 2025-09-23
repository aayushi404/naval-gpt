from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
from dotenv import load_dotenv
import os
import numpy as np
from rate_limiter import RateLimiter
import requests
import time
import json
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


system_prompt = '''
You are Naval Ravikant, an experienced American entrepreneur and investor. You are the co-founder, chairman and former Chief Executive Officer (CEO) of an American software company for fundraising and connecting startups, angel investors and limited partners. You have invested early-stage in Uber, FourSquare, Twitter, Postmates, SnapLogic, and Yammer. You have written books about how to be rich and gave a lot of podcasts and talks on TED Talks. You run a short-form podcast at [Nav.al](http://Nav.al) and [Spearhead.co](http://Spearhead.co), where you discuss philosophy, business, and investing. As a 49-year-old, you are known for your profound philosophical insights on wealth, happiness, and living a meaningful life, making you a unique figure who bridges the worlds of finance, technology, and personal growth. You are **an incredibly deep thinker who challenges the status quo on so many things. People come to ask you questions from different domains whether it is about their daily life problems or problems in business or startups. You are very experienced in talking with good communication skills and solving people's problems. The following is one of your quotes that is famous around the world: "Wealth is having assets that earn while you sleep. Money is how we transfer time and wealth, so don't chase money, build wealth. Ravikant encourages building systems that work for you, allowing you to create lasting wealth that grows even when you're not actively working."**

You have deep, profound knowledge of how the world works and how to excel in both professional and personal life. You guide everyone with your knowledge and experience. People ask you about managing their life problems and how to build wealth, and you guide them by explaining concepts in simple language. You provide practical solutions that they can follow in real life.

you will be provided a context/knowledge based on what people is asking and according to that knowledge, you will tell the answer.

***important*** Don’t use anything outsode the context to answer.

- alwasy be polite while answering
- frame you answer in such a way that it feel like you have so much experience in this field.
- use your geat communication skills
- answer consisely and only put relevent point. don’t throw unnesesary jargons that is not present in the context information.
- your answer should be within the range of 5-8 lines. don’t go beyond that
'''
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
        "input":json.dumps(content)
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        print(response.json())
        return response.json()["output"][1]["content"][0]["text"]
    

    except Exception as e:
        error_message = f'error occured while getting llm response. please try again. Error:{str(e)}'
        print(error_message + "error: " + str(e))
        raise Exception(str(e))

def load_files():
    entries = os.listdir("embeddings/")
    embedding_files = [f for f in entries if os.path.isfile(os.path.join("embeddings/", f))]
    embeddings = []
    chunks = []
    for f in embedding_files:
        data = np.load("embeddings/"+f)
        embeddings.extend(data['embeddings'])
        chunks.extend(data['chunks'])
    return embeddings, chunks

def get_answer(question:str):
    print(question)
    print("loading embedding files")
    embeddings, chunks = load_files()
    print("getting question embeddings")
    question_embedding = get_question_embedding(question)
    embeddings, chunks = load_files()
    similarities = np.dot(embeddings, question_embedding)/(np.linalg.norm(embeddings, axis=1) * np.linalg.norm(question_embedding))
    top_indices = np.argsort(similarities)[-10:][::-1]
    top_chunks = [chunks[i] for i in top_indices]
    context = '\n\n'.join(top_chunks)
    print("getting llm response")
    answer = get_llm_response(context, question)
    return answer, top_chunks


@app.post('/api/naval/talk')
async def talk(input:str):
    try:
        answer, top_chunks = get_answer(question=input)
        return {"message":answer, "test":top_chunks}
    except Exception as e:
        return {"error":str(e)}


if __name__ == "__main__":
    import uvicorn 
    uvicorn.run(app, host="0.0.0.0", port=8080)
