from flask import Flask, request, jsonify
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
import chromadb
from unidecode import unidecode
import re
import os
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Retrieve the Hugging Face API key from environment variables
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if api_key is None:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable not set")

# Example usage of the API key (if needed)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key

# Global variables for vector_db and qa_chain
vector_db = None
qa_chain = None

def load_doc(list_file_path, chunk_size, chunk_overlap):
    loaders = [PyPDFLoader(x) for x in list_file_path]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)
    doc_splits = text_splitter.split_documents(pages)
    return doc_splits

def create_db(splits, collection_name):
    embedding = HuggingFaceEmbeddings()
    new_client = chromadb.EphemeralClient()
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        client=new_client,
        collection_name=collection_name,
    )
    return vectordb

def initialize_llmchain(llm_model, temperature, max_tokens, top_k, vector_db):
    if llm_model == "mistralai/Mistral-7B-Instruct-v0.2":
        llm = HuggingFaceEndpoint(
            repo_id=llm_model,
            temperature=temperature,
            max_new_tokens=max_tokens,
            top_k=top_k,
        )
    # Add more conditions for other LLM models as needed

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key='answer',
        return_messages=True
    )
    retriever = vector_db.as_retriever()

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        chain_type="stuff",
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    return qa_chain

def initialize():
    global vector_db, qa_chain

    # Perform initialization steps if not already initialized
    if vector_db is None or qa_chain is None:
        pdf_files = [Path("testing_pdf.pdf")]  # Replace with actual file paths
        chunk_size = 256
        chunk_overlap = 64

        # Initialize vector database
        collection_name = create_collection_name(pdf_files[0])
        doc_splits = load_doc(pdf_files, chunk_size, chunk_overlap)
        vector_db = create_db(doc_splits, collection_name)

        # Initialize QA chain (choose an LLM model)
        llm_model = "mistralai/Mistral-7B-Instruct-v0.2"
        temperature = 0.7
        max_tokens = 256
        top_k = 5
        qa_chain = initialize_llmchain(llm_model, temperature, max_tokens, top_k, vector_db)

def create_collection_name(filepath):
    collection_name = Path(filepath).stem
    collection_name = collection_name.replace(" ", "-")
    collection_name = unidecode(collection_name)
    collection_name = re.sub('[^A-Za-z0-9]+', '-', collection_name)
    collection_name = collection_name[:50]
    if len(collection_name) < 3:
        collection_name = collection_name + 'xyz'
    if not collection_name[0].isalnum():
        collection_name = 'A' + collection_name[1:]
    if not collection_name[-1].isalnum():
        collection_name = collection_name[:-1] + 'Z'
    return collection_name

def format_chat_history(history):
    """
    Format the chat history into a string or list format that the QA chain expects.
    
    Args:
        history (list): A list of dictionaries with keys "message" and "response".
        
    Returns:
        formatted_history: A formatted chat history suitable for input to the QA chain.
    """
    formatted_history = []
    for entry in history:
        formatted_history.append((entry["message"], entry.get("response", "")))
    return formatted_history

def conversation(qa_chain, message, history):
    formatted_chat_history = format_chat_history(history)
    result = qa_chain(
        {"question": message, "chat_history": formatted_chat_history}
    )
    response = result["answer"]
    response_answer = response
    if response_answer.find("Helpful Answer:") != -1:
        response_answer = response_answer.split("Helpful Answer:")[-1]
    response_sources = result["source_documents"]
    response_source1 = response_sources[0].page_content.strip()
    response_source2 = response_sources[1].page_content.strip()
    response_source3 = response_sources[2].page_content.strip()
    response_source1_page = response_sources[0].metadata["page"] + 1
    response_source2_page = response_sources[1].metadata["page"] + 1
    response_source3_page = response_sources[2].metadata["page"] + 1
    return response_answer, response_source1, response_source2, response_source3, response_source1_page, response_source2_page, response_source3_page

@app.route('/initialize', methods=['POST'])
def initialize_endpoint():
    initialize()
    return 'Initialization complete', 200

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    global vector_db, qa_chain

    message = request.json.get('message')
    history = request.json.get('history', [])

    # Ensure vector_db and qa_chain are initialized before processing chat
    if vector_db is None or qa_chain is None:
        initialize()  # Ensure initialization is done before processing chat

    # Process the message using qa_chain
    response_answer, response_source1, response_source2, response_source3, response_source1_page, response_source2_page, response_source3_page = conversation(qa_chain, message, history)

    return jsonify({
        'response': response_answer,
        'source1': response_source1,
        'source2': response_source2,
        'source3': response_source3,
        'page1': response_source1_page,
        'page2': response_source2_page,
        'page3': response_source3_page
    }), 200

if __name__ == '__main__':
    app.run(debug=True)
