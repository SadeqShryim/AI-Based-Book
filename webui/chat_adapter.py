from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda
import os

from prompt_templates import james_clear_prompt

# === Vector Store ===
persist_dir = os.path.join("..", "data", "vectorstore")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# === Memory ===
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    return_messages=True
)

# === LLM ===
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=512
)

# === Chain ===
def retrieve_and_format(inputs):
    docs = retriever.invoke(inputs["question"])
    context = "\n\n".join([doc.page_content for doc in docs])
    return {
        "context": context,
        "question": inputs["question"],
        "chat_history": inputs["chat_history"]
    }

chain = RunnableLambda(retrieve_and_format) | james_clear_prompt | llm

def run_chat(user_input: str) -> str:
    chat_history = memory.load_memory_variables({}).get("chat_history", "")
    inputs = {
        "question": user_input,
        "chat_history": chat_history
    }
    output = chain.invoke(inputs)
    memory.save_context(inputs, {"output": output})
    return output
