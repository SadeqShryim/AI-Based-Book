from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from webui.prompt_templates import james_clear_prompt

from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda
import os


def load_vectorstore():
    persist_dir = os.path.join("..", "data", "vectorstore")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)

def build_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = OllamaLLM(model="phi3")

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        return_messages=True
    )

    # Pull docs and inject into {context}
    def retrieve_and_format(inputs):
        docs = retriever.invoke(inputs["question"])  # use `.invoke()` not `.get_relevant_documents()`
        context = "\n\n".join([doc.page_content for doc in docs])
        return {
            "context": context,
            "question": inputs["question"],
            "chat_history": inputs["chat_history"]
        }

    # Create LCEL chain
    chain = (
        RunnableLambda(retrieve_and_format)
        | james_clear_prompt
        | llm
    )

    # Add memory to track history


    return chain, memory

def run_with_memory(chain, memory, question):
    chat_history = memory.load_memory_variables({})["chat_history"]

    inputs = {
        "question": question,
        "chat_history": chat_history
    }

    output = chain.invoke(inputs)
    memory.save_context(inputs, {"output": output})
    return output



def main():
    print("🔁 Loading vectorstore...")
    vectorstore = load_vectorstore()

    print("🤖 Atomic Habits AI is ready. Ask a question (type 'exit' to quit):\n")
    qa_chain, memory = build_qa_chain(vectorstore)

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break

        response = run_with_memory(qa_chain, memory, query)
        print("\nAI:", response, "\n")


if __name__ == "__main__":
    main()
