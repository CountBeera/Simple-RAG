# chatbot.py

import os
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# constants
PERSIST_DIRECTORY = "chroma_db"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_BASE = "https://api.groq.com/openai/v1"
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

# 1️⃣ Load embeddings & vectordb (from previous ingestion)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

# 2️⃣ Configure the Groq-backed chat model
llm = ChatOpenAI(
    model_name=MODEL_NAME,
    temperature=0.0,
    openai_api_key=GROQ_API_KEY,
    openai_api_base=GROQ_API_BASE
)

# 3️⃣ Memory to hold chat history
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 4️⃣ Build a RAG-enabled conversational chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
    memory=memory
)

def chat(query: str) -> str:
    """Given a user query, run the RAG chain and return the LLM’s answer."""
    result = qa_chain({"question": query})
    return result["answer"]
