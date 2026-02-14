from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated,Literal
from langgraph.graph.message import add_messages
from pydantic import BaseModel,Field
from langchain_groq import ChatGroq
from dotenv import load_dotenv

#tools imports
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_community.tools import DuckDuckGoSearchResults,WikipediaQueryRun,YouTubeSearchTool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
# save memory
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import SystemMessage,BaseMessage
from datetime import datetime
import operator
import sqlite3

# rag imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

# async support
load_dotenv()

def load_model():
   
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    return llm

# embedding model 
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")


# ingest pdf
def ingest_pdf(file_path:str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    # save in store
    store = FAISS.from_documents(texts, embedding_model)
    return store

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages]

llm = load_model()
# tools 
## web search tool
search_tool = DuckDuckGoSearchResults()
## wikipedia tool
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
## youtube search tool
youtube_tool = YouTubeSearchTool()
## current time tool
@tool
def current_time() -> dict:
    """
    Get the current date and time.
    """
    now = datetime.now()
    return {"current_time": now.strftime("%Y-%m-%d %H:%M:%S")}
# custom calculator tool
# @tool
# def calculator(first_num : float , second_num : float , action : Literal['add','sub','mul','div'])-> dict:
#     """
#     Perform a basic arithmetic operation on two numbers.
#     supported actions are add, sub, mul, and div.
#     """
#     try:
#         if action == 'add':
#             result = operator.add(first_num, second_num)
#         elif action == 'sub':
#             result = operator.sub(first_num, second_num)
#         elif action == 'mul':
#             result = operator.mul(first_num, second_num)
#         elif action == 'div':
#             if second_num == 0:
#                 return {"error": "Division by zero is not allowed."}
#             result = operator.truediv(first_num, second_num)
#         else:
#             return {"error": "Unsupported action. Use 'add', 'sub', 'mul', or 'div'."}
        
#         return {'first_num': first_num, 'second_num': second_num, 'action': action, 'result': result}
#     except Exception as e:
#         return {"error": str(e)}

rag_store = None  # global

#rag tool
@tool
def search_rag(query: str) -> dict:
    """Search RAG store."""
    global rag_store

    if rag_store is None:
        return {"error": "No document uploaded yet"}

    results = rag_store.similarity_search(query, k=3)

    return {
        "results": [doc.page_content for doc in results]
    }

# use more tools as needed, and remember to bind them to the llm before using
# to not use much tools in a simple llm
tools = [search_tool,wiki_tool,youtube_tool,current_time,search_rag]
llm_with_tools = llm.bind_tools(tools)


# nodes 
        
def chat_node(state:ChatState):
    # take user query from state
    messages = state['messages']
    system_msg = SystemMessage(
    content="""
    Use search_rag when questions relate to uploaded documents.
    Tool calls must be JSON.
    """
    )
    messages = [system_msg] + messages
    
    # send to llm 
    response = llm_with_tools.invoke(messages)
    # response store state
    return {
        'messages': [response]
    }
    
tool_node = ToolNode(tools)
    
# checkpointer
conn = sqlite3.connect(database='chatbot.db',check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)
# create a graph
graph = StateGraph(ChatState)

# add nodes
graph.add_node('chat_node',chat_node)
graph.add_node('tools',tool_node)

# add edges
graph.add_edge(START,'chat_node')
graph.add_conditional_edges('chat_node',tools_condition)
graph.add_edge('tools','chat_node')

# compile graph
chatbot = graph.compile(checkpointer=checkpointer)



def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)
