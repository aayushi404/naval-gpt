import os
import json
import requests
from dotenv import load_dotenv

from ..services.retrieve import get_chunks

load_dotenv()

def get_sys_prompt():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    with open(f'{BASE_DIR}/data/prompts/sys_prompt.txt', 'r') as f:
        sys_prompt = f.read()

    return sys_prompt


def get_llm_response(context:str, question:str):
    content = {"context":context, "userQuestion":question}
    url = os.getenv("OPENAI_BASE_URL") + "/responses"
    API_KEY = os.getenv("OPENAI_API_KEY")
    system_prompt = get_sys_prompt()
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

def simple_retrieval(question:str):
    chunks = get_chunks(question)

    context = "\n\n".join(chunks)

    print("Getting llm Response")

    answer = get_llm_response(context, question)

    return answer, chunks
