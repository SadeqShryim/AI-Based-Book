from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub

import os
from prompt_templates import james_clear_prompt

# Load environment variable for HuggingFaceHub
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# === Safe Embedding Loader ===
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    persist_dir = os.path.join("..", "data", "vectorstore")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
except Exception as e:
    import logging
    logging.warning(f"Embedding load failed: {e}")
    
    # Fallback retriever for deployment testing
    class DummyRetriever:
        def invoke(self, question):
            return []
    retriever = DummyRetriever()

# === Conversation Memory ===
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    return_messages=True
)

# === LLM (Free HF Model) ===
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.5, "max_new_tokens": 512},
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)

# === Chain Logic ===
def retrieve_and_format(inputs):
    docs = retriever.invoke(inputs["question"])
    context = "\n\n".join([doc.page_content for doc in docs]) if docs else "No relevant passages found."
    return {
        "context": context,
        "question": inputs["question"],
        "chat_history": inputs["chat_history"]
    }

chain = RunnableLambda(retrieve_and_format) | james_clear_prompt | llm

# === Entry Function ===
def run_chat(user_input: str) -> str:
    chat_history = memory.load_memory_variables({}).get("chat_history", "")
    inputs = {
        "question": user_input,
        "chat_history": chat_history
    }
    output = chain.invoke(inputs)
    memory.save_context(inputs, {"output": output})
    return output
