from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .controller.agentic_retrieval import agentic_retrieval
from .controller.simple_retrieval import simple_retrieval
load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get('/api/naval/talk')
async def talk(input:str):
    try:
        answer, top_chunks = simple_retrieval(question=input)
        print(answer)
        return {"message":answer, "test":top_chunks}
    except Exception as e:
        return {"error":str(e)}


@app.get('/api/agent/talk')
async def answer(input:str):
    try:
        print()
        answer = agentic_retrieval(input)
        return{"message":answer}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn 
    uvicorn.run(app, host="0.0.0.0", port=8080)
