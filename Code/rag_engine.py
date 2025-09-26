# rag_engine.py

import requests
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- API Configuration ---
# IMPORTANT: Place your OpenRouter API key here
OPENROUTER_API_KEY = "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" 
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "x-ai/grok-4-fast:free"

class RAG_Engine:
    def __init__(self, pdf_path):
        """
        Initializes the RAG engine.
        1. Loads the PDF.
        2. Splits it into chunks.
        3. Creates a vector store from the chunks.
        """
        print("Initializing RAG Engine...")
        
        # 1. Load the document
        loader = PyPDFLoader(pdf_path)
        self.documents = loader.load()
        print(f"Loaded {len(self.documents)} pages from the PDF.")

        # 2. Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        self.chunks = text_splitter.split_documents(self.documents)
        print(f"Split the document into {len(self.chunks)} chunks.")

        # 3. Create embeddings and vector store
        # Using a popular, high-quality open-source embedding model
        embedding_model =  HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        print("Creating vector store... (This might take a moment on first run)")
        self.vector_store = FAISS.from_documents(self.chunks, embedding_model)
        print("Vector store created successfully.")

        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant chunks

    def _get_deepseek_response(self, prompt):
        """
        Private method to call the OpenRouter API.
        """
        if not OPENROUTER_API_KEY or "sk-or-v1-..." in OPENROUTER_API_KEY:
            return "Error: Please provide a valid OpenRouter API key in rag_engine.py"

        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        data = json.dumps({"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}]})

        try:
            response = requests.post(API_URL, headers=headers, data=data, timeout=30)
            response.raise_for_status()
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            return f"Error: API connection failed: {e}"

    def query(self, user_question):
        """
        The main query function.
        1. Retrieves relevant context.
        2. Builds a prompt.
        3. Calls the LLM to get an answer.
        """
        print(f"Received query: {user_question}")

        # 1. Retrieve relevant context
        context_docs = self.retriever.invoke(user_question)
        context = "\n\n".join([doc.page_content for doc in context_docs])
        print(f"Retrieved context: {context[:500]}...") # Print first 500 chars of context

        # 2. Build the prompt
        prompt_template = f"""
        You are an expert assistant for 'Project Nova'. Your task is to answer questions accurately based ONLY on the provided context.
        If the answer is not available in the context, clearly state "I do not have information on this topic based on the provided document." Do not make up information.

        CONTEXT:
        ---
        {context}
        ---

        QUESTION: {user_question}

        ANSWER:
        """

        # 3. Get the response from the LLM
        answer = self._get_deepseek_response(prompt_template)
        print(f"Generated answer: {answer}")
        return answer