import json
from typing import List, Tuple
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import gradio as gr
import env
from log import get_logger
from vectordb import VectorDB

LOGGER = get_logger(__name__)
LOGGER.info('Startup sequence initiated')

KNOWLEDGEBASE = VectorDB()
CHAT_MODEL = ChatOpenAI(model=env.OPENAI_CHAT_MODEL, temperature=0, api_key=env.OPENAI_API_KEY, streaming=True)

SYSTEM_TEMPLATE = """
    You are a succinct conversational agent who provides Buddhist insight and perspective into the user's life.
    Whatever is troubling them, respond in a manner to keep the conversation running, but ground yourself in the provided context.
    This context is Buddhist philosophy from a slightly oversimplified book.
    Each element of the context starts with a theoretical question, then an answer to that question.
    When replying, ignore context that is irrelevant. It may have been retrieved mistakenly.
    You must always cite a source from the context at the end of your reply or state when it may contradict itself.
    If you are ever unsure of the Buddhist perspective, you may still respond, but be transparent about your lack of knowledge.
    Context is provided below.
    ```
""".replace('\t', '').strip()


def chat_function(message:str, history:List[Tuple[str, str]]):
    messages = []
    # retrieve context
    retrieved_docs = KNOWLEDGEBASE.search(query=message)
    retrieved_docs.sort(key=lambda t:t[1], reverse=True)
    context = '\n---\n'.join([f"Document: {d[0].page_content}\n(Metadata JSON: {json.dumps(d[0].metadata)})" for d in retrieved_docs])
    # prepare prompt
    messages.append(SystemMessage(content = '\n'.join([SYSTEM_TEMPLATE, context])))
    history_to_keep = min(len(history), 2)
    for u,ai in history[:history_to_keep]:
        messages.append(HumanMessage(content=u))
        messages.append(AIMessage(content=ai))
    messages.append(HumanMessage(content=message))
    # feed into chatgpt for response
    full_answer = ''
    for chunk in CHAT_MODEL.stream(input=messages):
        full_answer += chunk.content
        yield full_answer


INTERFACE = gr.ChatInterface(
    chat_function,
    title='Buddhism For Beginners',
)
INTERFACE.launch(
    server_name=env.SERVER_HOST,
    server_port=env.SERVER_PORT,
)
