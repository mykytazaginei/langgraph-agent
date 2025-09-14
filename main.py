import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
import operator
from langgraph.graph import StateGraph, END
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import WebBaseLoader
import trafilatura

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

base_llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    task="conversational",  
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
)

llm = ChatHuggingFace(llm=base_llm)
@tool
def read_file(path: str) -> str:
    """Reads the content of a file from a given path."""
    try:
        with open(path, 'r', encoding='uft-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: The file at path '{path}' was not found."
    except Exception as e:
        return f"An error occurred while reading the file: {e}"

@tool
def write_file(path:str, content:str)->str:
    """Write content to a file at given path"""
    try:
        with open(path, 'w', encoding='uft-8') as f:
            f.write(content) 
    except FileNotFoundError:
        return f"Error: The file at path '{path}' was not found."
    except Exception as e:
        return f"An error occurred while reading the file: {e}"

@tool
def load_web_text(url:str)->str :
    '''web'''
    docs = WebBaseLoader(url).load()
    html = trafilatura.featch_url(url)
    return trafilatura.extract(html) or "\n\n".join(d.page_content for d in docs)

@tool 
def web_search(query: str) -> str:
    '''Web Search by query using DuckDuckGo.'''
    search = DuckDuckGoSearchRun()
    return str(search.invoke(query))

tools = [web_search, load_web_text, read_file, write_file]
llm_with_tools = llm.bind_tools(tools)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a useful agent. If necessary, use tools to search for information. Answer briefly."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | llm_with_tools

def call_model(state: AgentState):
    messages = state['messages']
    response = chain.invoke(messages)  
    return {'messages': [response]}  

def call_tool(state: AgentState):
    last_message = state['messages'][-1]  # Последний ответ модели
    if last_message.tool_calls:  # Проверяем, есть ли вызовы
        tool_call = last_message.tool_calls[0]  # Берем первый
        # Находим инструмент по имени
        tool = next(t for t in tools if t.name == tool_call['name'])
        # Вызываем с аргументами
        tool_output = tool.invoke(tool_call['args'])
        # Добавляем результат как новое сообщение
        return {"messages": [AIMessage(content=tool_output)]}
    return state  # Если нет вызова, ничего не делаем

def should_continue(state: AgentState):
    last_message = state['messages'][-1]
    if last_message.tool_calls:  
        return "tool" 
    return END  

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tool", call_tool) 
workflow.set_entry_point("agent")

workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tool", "agent")

app = workflow.compile()

if __name__ == "__main__":
    inputs = {"messages": [HumanMessage(content="Какова столица Франции?")]}
    result = app.invoke(inputs)
    print(result['messages'][-1].content)
