from langgraph.graph import state, StateGraph, END, START
from langchain.messages import HumanMessage
from langgraph.types import Command
from langchain.chat_models import init_chat_model
from typing import TypedDict, Literal

from .retrieve import get_chunks

class Classification(TypedDict):
    topic: Literal["General", "Tech", "Wealth"]
    search_query: str | None

class State(TypedDict):
    classify: Classification | None
    docSearch: list[str] | None
    
    messages: list[str] | None
    
llm = init_chat_model('groq:llama-3.3-70b-versatile')

def classify(State:state):
    """Classify the intend of the user query"""
    structured_llm = llm.with_structured_output(Classification)

    classification_prompt = f"""
        Analyse and understand the user query and classify the topic.
        
        General:
        - If the topic is "General" then there is no need to retrieve any kind of document so keep the search_query param as "None".
        - If the topic is not "General" then write a brief search query according to the user requirements.The search query should be comprihensive enough so that it could do the similarity search based on the query and retrieve the best context.

        user: {State['messages']}
        Classify the topic
    """

    classification = structured_llm.invoke(classification_prompt)
    print(classification)

    if classification['topic'] == 'Wealth':
        goto = "wealth_docs"

    elif classification['topic'] == 'Tech':
        goto = "tech_docs"

    else:
        goto = "draft_response"

    return Command(
        update={"classify": classification},
        goto=goto
    )

def wealth_docs(State:state):
    """Retrieve the wealth documents for context"""
    classification = State.get('classify', {})
    search_query = f"{classification.get('search_query', "")}"

    search_result = get_chunks(search_query)

    return {"docSearch": search_result}

def tech_docs(State:state):
    """Retrieve the tech documents for context"""
    classification = State.get('classfy', {})
    search_query = f"{classification.get('search_query', "")}"

    search_result = get_chunks(search_query)

    return {"docSearch": search_result}

def draft_response(State:state):
    """Draft the final response to user"""

    prompt = """
        You are an excellent speaker and advisor.You give advise and help people in their carrier and personal development.
        You have a very good communication skils experience in guiding people in tech and business.
    """
    classification = State.get('classify', {})
    if classification.get('topic', 'General') == 'General':
        general_prompt = f"""
            {prompt}
            user: {State['messages'][0]}
            Talk and engage with the user.
        """
        response = llm.invoke(general_prompt)

    else:
        enhanced_prompt = f""" 
            {prompt}
            To help the user with better advice use the context to frame your question. The context data came from some famous and sucessfull people from the industry so answer the user according to the context.
            context: {State['docSearch']}
            user: {State['messages'][0]}

            Guidelines:
            - Be professional and helpful
            - Address their specific concern
            - Use the provided documentation
        """
        response = llm.invoke(enhanced_prompt)
    
    return {"messages":[response]}

def send_reply(State:state):
    """Sends the reply to the user"""

    return {"messages":State['messages'][0].content}

workflow = StateGraph(State)

workflow.add_node("classify", classify)
workflow.add_node("wealth_docs", wealth_docs)
workflow.add_node("tech_docs", tech_docs)
workflow.add_node("draft_response", draft_response)
workflow.add_node("send_reply", send_reply)

workflow.add_edge(START, "classify")
workflow.add_edge("wealth_docs", "draft_response")
workflow.add_edge("tech_docs", "draft_response")
workflow.add_edge("draft_response", "send_reply")
workflow.add_edge("send_reply", END)

graph = workflow.compile()

