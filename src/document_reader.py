import os
from typing_extensions import TypedDict
from typing import Annotated

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from langgraph.graph import StateGraph, END, add_messages, START
from langgraph.types import interrupt

import pdf2image
import pytesseract
from PIL import Image
from pymongo import MongoClient
import datetime
from dotenv import load_dotenv
load_dotenv()

client = MongoClient(os.getenv("connection_string"))
db = client["pdf_project"]

ceased_collection = db["ceased_docs"]
archive_collection = db["archived_docs"]
audit_collection = db["audit_logs"]

# Define the tool to read a single PDF file
@tool
def read_pdf(file_path: str) -> str:
    """
    Read a single PDF file and return its text content.
    Handles both text-based and image-based (scanned) PDFs using OCR for the latter.
    """
    if not os.path.exists(file_path):
        return f"File {file_path} does not exist."
    
    if not file_path.endswith('.pdf'):
        return f"File {file_path} is not a PDF."
    
    try:
        # First, try to extract text directly
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        extracted_text = ""
        for doc in docs:
            extracted_text += doc.page_content + "\n"
        
        # Check if meaningful text was extracted (more than just whitespace)
        if extracted_text.strip():
            return f"--- Text from {os.path.basename(file_path)} ---\n{extracted_text}"
        else:
            # Fallback to OCR for image-based PDFs
            images = pdf2image.convert_from_path(file_path)
            ocr_text = ""
            for image in images:
                ocr_text += pytesseract.image_to_string(image) + "\n"
            return f"--- OCR Text from {os.path.basename(file_path)} ---\n{ocr_text}"
    except Exception as e:
        return f"Error reading {file_path}: {str(e)}"

# Define the agent state
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    classification: str

system_prompt = """You are a helpful agent that can read PDF files and classify their content. 
    When asked to classify a PDF, first use the read_pdf tool to get the content, then classify it as either
    'ceased' Valid cease & desist request
    or 'irrelevant' Not a cease request, 
    or 'uncertain' (if it's unclear). Provide the text and the classification."""

agent = create_agent("gpt-5", tools = [read_pdf], system_prompt=system_prompt)

# Define the agent node
def document_loader_agent(state: AgentState):

    result = agent.invoke({"messages": state["messages"]})

    last_msg = result["messages"][-1].content.lower()

    if "ceased" in last_msg:
        classification = "ceased"
    elif "irrelevant" in last_msg:
        classification = "irrelevant"
    else:
        classification = "uncertain"
    
    return {"messages": result["messages"],
            "classification":classification}

def route(state:AgentState):
    return state["classification"]

def cease_node(state: AgentState):
    return state

def irrelevant_node(state: AgentState):
    return state

def uncertain_node(state: AgentState):
    return state

def human_review_node(state: AgentState):
    # Pause execution and ask human
    human_input = interrupt(
        {
            "question": "Classification is uncertain. Please choose:",
            "options": ["ceased", "irrelevant"]
        }
    )

    return {
        "classification": human_input
    }

import re

def extract_doc_name(messages):
    text = messages[0].content
    match = re.search(r"Classify the PDF:\s*(.*\.pdf)", text)
    return match.group(1) if match else "unknown.pdf"

def database_agent(state: AgentState):

    document_name = extract_doc_name(state["messages"])

    record = {
        "document_name": document_name,
        "classification": "ceased",
        "received_date": datetime.datetime.now(),
        "details": state["messages"][-1].content[:1000]
    }

    ceased_collection.insert_one(record)

    return {
        **state,
        "db_record": record
    }

def archiving_agent(state: AgentState):

    document_name = extract_doc_name(state["messages"])

    record = {
        "document_name": document_name,
        "classification": "irrelevant",
        "archived_date": datetime.datetime.now(),
        "reason": "Not relevant document"
    }

    archive_collection.insert_one(record)

    return {
        **state,
        "archive_record": record
    }

def audit_agent(state: AgentState):

    record = {
        "document_name": extract_doc_name(state["messages"]),
        "final_classification": state["classification"],
        "timestamp": datetime.datetime.now(),
        "status": "completed"
    }

    audit_collection.insert_one(record)
    return state


# Build the graph
graph = StateGraph(AgentState)
graph.add_node("document_loader_agent", document_loader_agent)
graph.add_node("ceased", cease_node)
graph.add_node("irrelevant", irrelevant_node)
graph.add_node("human_review", human_review_node)
graph.add_node("database_agent",database_agent)
graph.add_node("archiving_agent",archiving_agent)
graph.add_node("audit_agent",audit_agent)

graph.add_edge(START, "document_loader_agent")

graph.add_conditional_edges("document_loader_agent",route,{
    "ceased":"ceased",
    "irrelevant":"irrelevant",
    "uncertain":"human_review"
}
)

graph.add_conditional_edges("human_review",route,{
    "ceased":"ceased",
    "irrelevant":"irrelevant"
})

graph.add_edge("ceased", "database_agent")
graph.add_edge("irrelevant", "archiving_agent")

graph.add_edge("database_agent","audit_agent")
graph.add_edge("archiving_agent","audit_agent")
graph.add_edge("audit_agent",END)

# Compile the graph
app = graph.compile()

print(app.get_graph().draw_mermaid())

folder_path = "../files"
files = [
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if f.endswith(".pdf")
]
results = []

for file_path in files:

    print(f"Processing {file_path} file")
    invoke_text = f"Classify the PDF: {file_path}"

    result = app.invoke({"messages": [HumanMessage(content=invoke_text)]})
    results.append({
        "file": file_path,
        "classification": result.get("classification", "unknown")
    })
    print(f"Processed {file_path} file")

print(result)