from langchain.messages import HumanMessage
from ..services.retrieval_agent import graph

def agentic_retrieval(question:str):
    initial_state = {
        "messages": [HumanMessage(content=question)]
    }

    print(initial_state)
    response = graph.invoke(initial_state)

    print(response)
    return response