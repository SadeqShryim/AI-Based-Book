import streamlit as st
from chat_adapter import run_chat


# --- Page Config ---
st.set_page_config(page_title="Atomic Habits AI", page_icon="💬")

# --- App Title ---
st.title("💬 Atomic Habits AI")
st.caption("Ask anything... like James Clear is answering you.")

# --- Chat history state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Chat display ---
for msg in st.session_state.messages:
    role = "🧠 AI" if msg["role"] == "ai" else "🧍 You"
    with st.chat_message(msg["role"]):
        st.markdown(f"**{role}:** {msg['text']}")

# --- Input box ---
user_input = st.chat_input("Say something...")

if user_input:
    # Show your message
    st.chat_message("user").markdown(f"**🧍 You:** {user_input}")
    st.session_state.messages.append({"role": "user", "text": user_input})

    # Get AI response
    ai_response = run_chat(user_input)

    # Show AI message
    st.chat_message("ai").markdown(f"**🧠 AI:** {ai_response}")
    st.session_state.messages.append({"role": "ai", "text": ai_response})
