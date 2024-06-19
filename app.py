import os
from dotenv import load_dotenv
import gradio as grimport os
import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from pathlib import Path
from unidecode import unidecode
import re
import dill

# Load environment variables from .env file
load_dotenv()

# Disable Gradio analytics to avoid network timeout errors
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# Retrieve the Together API key from environment variables
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if api_key is None:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable not set")

# Example usage of the API key (if needed)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key

list_llm = [
    "mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.1", "google/gemma-7b-it", "google/gemma-2b-it",
    "HuggingFaceH4/zephyr-7b-beta", "HuggingFaceH4/zephyr-7b-gemma-v0.1",
    "meta-llama/Llama-2-7b-chat-hf", "microsoft/phi-2",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "mosaicml/mpt-7b-instruct", "tiiuae/falcon-7b-instruct",
    "google/flan-t5-xxl"
]
list_llm_simple = [os.path.basename(llm) for llm in list_llm]

def load_doc(list_file_path, chunk_size, chunk_overlap):
    loaders = [PyPDFLoader(x) for x in list_file_path]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    doc_splits = text_splitter.split_documents(pages)
    return doc_splits

def create_db(splits, collection_name, persist_directory):
    embedding = HuggingFaceEmbeddings()
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    return vectordb

def load_db(persist_directory, collection_name):
    embedding = HuggingFaceEmbeddings()
    vectordb = Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embedding,
    )
    return vectordb

def initialize_llmchain(llm_model, temperature, max_tokens, top_k, vector_db, progress=gr.Progress()):
    progress(0.1, desc="Initializing HF tokenizer...")
    progress(0.5, desc="Initializing HF Hub...")
    llm = HuggingFaceEndpoint(
        repo_id=llm_model,
        temperature=temperature,
        max_new_tokens=max_tokens,
        top_k=top_k,
        load_in_8bit=True,
    )
    progress(0.75, desc="Defining buffer memory...")
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key='answer',
        return_messages=True
    )
    retriever = vector_db.as_retriever()
    progress(0.8, desc="Defining retrieval chain...")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        chain_type="stuff",
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    progress(0.9, desc="Done!")
    return qa_chain

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
    print('Filepath:', filepath)
    print('Collection name:', collection_name)
    return collection_name

def initialize_database(list_file_obj, chunk_size, chunk_overlap, persist_directory, progress=gr.Progress()):
    list_file_path = [x.name for x in list_file_obj if x is not None]
    progress(0.1, desc="Creating collection name...")
    collection_name = create_collection_name(list_file_path[0])
    db_path = os.path.join(persist_directory, collection_name)
    if os.path.exists(db_path):
        progress(0.5, desc="Loading existing vector database...")
        vector_db = load_db(persist_directory, collection_name)
    else:
        progress(0.25, desc="Loading document...")
        doc_splits = load_doc(list_file_path, chunk_size, chunk_overlap)
        progress(0.5, desc="Generating vector database...")
        vector_db = create_db(doc_splits, collection_name, persist_directory)
    progress(0.9, desc="Done!")
    return vector_db, collection_name, "Complete!"

def initialize_LLM(llm_option, llm_temperature, max_tokens, top_k, vector_db, progress=gr.Progress()):
    llm_name = list_llm[list_llm_simple.index(llm_option)]
    print("llm_name:", llm_name)
    qa_chain = initialize_llmchain(llm_name, llm_temperature, max_tokens, top_k, vector_db, progress)
    return qa_chain, "Complete!"

def format_chat_history(chat_history):
    formatted_chat_history = []
    for user_message, bot_message in chat_history:
        formatted_chat_history.append(f"User: {user_message}")
        formatted_chat_history.append(f"Assistant: {bot_message}")
    return formatted_chat_history

def conversation(qa_chain, user_input, chat_history):
    formatted_chat_history = format_chat_history(chat_history)
    result = qa_chain({
        "question": user_input,
        "chat_history": formatted_chat_history
    })
    return result['answer'], result['source_documents']

def save_llm_chain_state(qa_chain, filepath):
    with open(filepath, 'wb') as f:
        dill.dump(qa_chain, f)

def load_llm_chain_state(filepath):
    with open(filepath, 'rb') as f:
        qa_chain = dill.load(f)
    return qa_chain

# Example Gradio app
persist_directory = 'persist_directory'
llm_state_file = 'llm_chain_state.pkl'

with gr.Blocks() as demo:
    gr.Markdown("# PDF Chatbot")
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload PDF")
            chunk_size = gr.Slider(minimum=100, maximum=1000, step=50, label="Chunk Size", value=500)
            chunk_overlap = gr.Slider(minimum=0, maximum=100, step=10, label="Chunk Overlap", value=50)
            initialize_db_button = gr.Button("Initialize Database")
        with gr.Column():
            llm_option = gr.Dropdown(label="Choose LLM", choices=list_llm_simple, value=list_llm_simple[0])
            llm_temperature = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label="LLM Temperature", value=0.7)
            max_tokens = gr.Slider(minimum=1, maximum=1024, step=1, label="Max Tokens", value=512)
            top_k = gr.Slider(minimum=1, maximum=100, step=1, label="Top K", value=50)
            initialize_llm_button = gr.Button("Initialize LLM")
            save_llm_button = gr.Button("Save LLM State")

    vector_db = gr.State(None)
    qa_chain = gr.State(None)

    def load_initial_state():
        if os.path.exists(llm_state_file):
            qa_chain = load_llm_chain_state(llm_state_file)
            vector_db = load_db(persist_directory, qa_chain.vector_db.collection_name)
            return qa_chain, vector_db, "LLM State Loaded!"
        return None, None, "No LLM State Found"

    def initialize_database_fn(file, chunk_size, chunk_overlap, progress=gr.Progress()):
        vector_db, collection_name, status = initialize_database([file], chunk_size, chunk_overlap, persist_directory, progress)
        save_llm_chain_state(qa_chain, llm_state_file)
        return vector_db, status

    initialize_db_button.click(
        initialize_database_fn,
        inputs=[file_input, chunk_size, chunk_overlap],
        outputs=[vector_db, gr.Textbox(label="Status")]
    )

    def initialize_llm_fn(llm_option, llm_temperature, max_tokens, top_k, vector_db, progress=gr.Progress()):
        qa_chain, status = initialize_LLM(llm_option, llm_temperature, max_tokens, top_k, vector_db, progress)
        save_llm_chain_state(qa_chain, llm_state_file)
        return qa_chain, status

    initialize_llm_button.click(
        initialize_llm_fn,
        inputs=[llm_option, llm_temperature, max_tokens, top_k, vector_db],
        outputs=[qa_chain, gr.Textbox(label="Status")]
    )

    def save_llm_fn(qa_chain):
        save_llm_chain_state(qa_chain, llm_state_file)  # Access the file path directly
        return "LLM State Saved!"


    save_llm_button.click(
        save_llm_fn,
        inputs=[qa_chain],  # Only pass the qa_chain state as input
        outputs=[gr.Textbox(label="Save Status")]
)

    def conversation_fn(qa_chain, user_input):
        if qa_chain is None:
            return "LLM not initialized. Please initialize LLM.", []
        chat_history = gr.State([])
        result, source_docs = conversation(qa_chain, user_input, chat_history)
        return result, source_docs

    demo.launch()

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceEndpoint
from pathlib import Path
import chromadb
from unidecode import unidecode
from transformers import AutoTokenizer
import transformers
import torch
import tqdm 
import accelerate
import re

# Load environment variables from .env file
load_dotenv()

# Retrieve the Together API key from environment variables
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if api_key is None:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable not set")

# Example usage of the API key (if needed)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key

# The rest of your code...
list_llm = ["mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mixtral-8x7B-Instruct-v0.1", "mistralai/Mistral-7B-Instruct-v0.1", \
    "google/gemma-7b-it","google/gemma-2b-it", \
    "HuggingFaceH4/zephyr-7b-beta", "HuggingFaceH4/zephyr-7b-gemma-v0.1", \
    "meta-llama/Llama-2-7b-chat-hf", "microsoft/phi-2", \
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "mosaicml/mpt-7b-instruct", "tiiuae/falcon-7b-instruct", \
    "google/flan-t5-xxl"
]
list_llm_simple = [os.path.basename(llm) for llm in list_llm]

def load_doc(list_file_path, chunk_size, chunk_overlap):
    loaders = [PyPDFLoader(x) for x in list_file_path]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size, 
        chunk_overlap = chunk_overlap)
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

def load_db():
    embedding = HuggingFaceEmbeddings()
    vectordb = Chroma(
        embedding_function=embedding)
    return vectordb

def initialize_llmchain(llm_model, temperature, max_tokens, top_k, vector_db, progress=gr.Progress()):
    progress(0.1, desc="Initializing HF tokenizer...")
    progress(0.5, desc="Initializing HF Hub...")
    if llm_model == "mistralai/Mixtral-8x7B-Instruct-v0.1":
        llm = HuggingFaceEndpoint(
            repo_id=llm_model, 
            temperature = temperature,
            max_new_tokens = max_tokens,
            top_k = top_k,
            load_in_8bit = True,
        )
    elif llm_model in ["HuggingFaceH4/zephyr-7b-gemma-v0.1","mosaicml/mpt-7b-instruct"]:
        raise gr.Error("LLM model is too large to be loaded automatically on free inference endpoint")
        llm = HuggingFaceEndpoint(
            repo_id=llm_model, 
            temperature = temperature,
            max_new_tokens = max_tokens,
            top_k = top_k,
        )
    elif llm_model == "microsoft/phi-2":
        llm = HuggingFaceEndpoint(
            repo_id=llm_model, 
            temperature = temperature,
            max_new_tokens = max_tokens,
            top_k = top_k,
            trust_remote_code = True,
            torch_dtype = "auto",
        )
    elif llm_model == "TinyLlama/TinyLlama-1.1B-Chat-v1.0":
        llm = HuggingFaceEndpoint(
            repo_id=llm_model, 
            temperature = temperature,
            max_new_tokens = 250,
            top_k = top_k,
        )
    elif llm_model == "meta-llama/Llama-2-7b-chat-hf":
        raise gr.Error("Llama-2-7b-chat-hf model requires a Pro subscription...")
        llm = HuggingFaceEndpoint(
            repo_id=llm_model, 
            temperature = temperature,
            max_new_tokens = max_tokens,
            top_k = top_k,
        )
    else:
        llm = HuggingFaceEndpoint(
            repo_id=llm_model, 
            temperature = temperature,
            max_new_tokens = max_tokens,
            top_k = top_k,
        )
    
    progress(0.75, desc="Defining buffer memory...")
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key='answer',
        return_messages=True
    )
    retriever=vector_db.as_retriever()
    progress(0.8, desc="Defining retrieval chain...")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        chain_type="stuff", 
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    progress(0.9, desc="Done!")
    return qa_chain

def create_collection_name(filepath):
    collection_name = Path(filepath).stem
    collection_name = collection_name.replace(" ","-") 
    collection_name = unidecode(collection_name)
    collection_name = re.sub('[^A-Za-z0-9]+', '-', collection_name)
    collection_name = collection_name[:50]
    if len(collection_name) < 3:
        collection_name = collection_name + 'xyz'
    if not collection_name[0].isalnum():
        collection_name = 'A' + collection_name[1:]
    if not collection_name[-1].isalnum():
        collection_name = collection_name[:-1] + 'Z'
    print('Filepath: ', filepath)
    print('Collection name: ', collection_name)
    return collection_name

def initialize_database(list_file_obj, chunk_size, chunk_overlap, progress=gr.Progress()):
    list_file_path = [x.name for x in list_file_obj if x is not None]
    progress(0.1, desc="Creating collection name...")
    collection_name = create_collection_name(list_file_path[0])
    progress(0.25, desc="Loading document...")
    doc_splits = load_doc(list_file_path, chunk_size, chunk_overlap)
    progress(0.5, desc="Generating vector database...")
    vector_db = create_db(doc_splits, collection_name)
    progress(0.9, desc="Done!")
    return vector_db, collection_name, "Complete!"

def initialize_LLM(llm_option, llm_temperature, max_tokens, top_k, vector_db, progress=gr.Progress()):
    llm_name = list_llm[llm_option]
    print("llm_name: ",llm_name)
    qa_chain = initialize_llmchain(llm_name, llm_temperature, max_tokens, top_k, vector_db, progress)
    return qa_chain, "Complete!"

def format_chat_history(chat_history):
    formatted_chat_history = []
    for user_message, bot_message in chat_history:
        formatted_chat_history.append(f"User: {user_message}")
        formatted_chat_history.append(f"Assistant: {bot_message}")
    return formatted_chat_history

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

with gr.Blocks() as demo:
    with gr.Tab("Initialize"):
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    pdf_files = gr.File(label="Upload PDFs", file_types=[".pdf"], file_count="multiple")
                    chunk_size = gr.Slider(64, 2048, value=256, step=64, label="Chunk Size")
                    chunk_overlap = gr.Slider(0, 512, value=64, step=16, label="Chunk Overlap")
                    initialize_db_btn = gr.Button("Initialize Vector Database")
                    db_output = gr.Textbox(label="Database Status")
                with gr.Column():
                    llm_choice = gr.Dropdown(label="Select LLM", choices=list_llm_simple, value=list_llm_simple[0])
                    llm_temperature = gr.Slider(0.0, 1.0, value=0.7, step=0.1, label="Temperature")
                    max_tokens = gr.Slider(16, 1024, value=256, step=16, label="Max Tokens")
                    top_k = gr.Slider(1, 10, value=5, step=1, label="Top K")
                    initialize_llm_btn = gr.Button("Initialize LLM")
                    llm_output = gr.Textbox(label="LLM Status")

    with gr.Tab("Chat"):
        with gr.Column():
            chatbot = gr.Chatbot()
            msg = gr.Textbox()
            submit_btn = gr.Button("Send")
            clear_btn = gr.Button("Clear")

    # Use Gradio's state to store vector_db and qa_chain objects
    vector_db_state = gr.State(None)
    qa_chain_state = gr.State(None)

    def on_initialize_db_btn_click(pdf_files, chunk_size, chunk_overlap, progress=gr.Progress()):
        vector_db, collection_name, status = initialize_database(pdf_files, chunk_size, chunk_overlap, progress)
        return status, vector_db

    initialize_db_btn.click(
        on_initialize_db_btn_click, 
        [pdf_files, chunk_size, chunk_overlap],
        [db_output, vector_db_state]
    )

    def on_initialize_llm_btn_click(llm_choice, llm_temperature, max_tokens, top_k, vector_db, progress=gr.Progress()):
        llm_option = list_llm_simple.index(llm_choice)
        qa_chain, status = initialize_LLM(llm_option, llm_temperature, max_tokens, top_k, vector_db, progress)
        return status, qa_chain

    initialize_llm_btn.click(
        on_initialize_llm_btn_click, 
        [llm_choice, llm_temperature, max_tokens, top_k, vector_db_state],
        [llm_output, qa_chain_state]
    )

    def on_submit_btn_click(msg, history, qa_chain):
        response_answer, response_source1, response_source2, response_source3, response_source1_page, response_source2_page, response_source3_page = conversation(qa_chain, msg, history)
        response = ["User: " + msg, "AI: " + response_answer]
        return history + [response], gr.update(value="")


    def on_clear_btn_click():
        return chatbot.update([]), gr.update(value="")

    submit_btn.click(
        on_submit_btn_click, 
        [msg, chatbot, qa_chain_state],
        [chatbot, msg]
    )

    clear_btn.click(on_clear_btn_click, [], [chatbot, msg])

demo.launch()


