import os
import boto3
import streamlit as st
from langchain_community.llms import Bedrock
from langchain.callbacks.base import BaseCallbackHandler
import re

# --- AWS Setup ---
os.environ["AWS_PROFILE"] = "robbarto"
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

# --- Mistral 7B Instruct Model ID ---
MISTRAL_MODEL_ID = "mistral.mistral-7b-instruct-v0:2"

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "token_count" not in st.session_state:
    st.session_state.token_count = 0

# --- Streaming Callback (optional) ---
class StreamHandler(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.content = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.content += token
        self.placeholder.markdown(self.content)

# --- Format Prompt for Mistral ---
def format_mistral_prompt(user_prompt):
    return f"[INST] {user_prompt.strip()} [/INST]"

# --- Clean Output ---
def clean_mistral_output(text):
    return re.sub(r"\[/?INST\]", "", text).strip()

# --- Sidebar Controls ---
st.sidebar.title("Settings")
temperature = st.sidebar.slider("Creativity (temperature)", 0.0, 1.5, 0.9, 0.1)

if st.sidebar.button("🔄 Reset Chat"):
    st.session_state.chat_history = []
    st.session_state.token_count = 0

# --- LLM Setup ---
llm = Bedrock(
    model_id=MISTRAL_MODEL_ID,
    client=bedrock_client,
    model_kwargs={"max_tokens": 1000, "temperature": temperature},
    streaming=False  # Enable if streaming is available in your region
)

# --- Page Title ---
st.title("🧠 Mistral 7B Instruct Chatbot (via AWS Bedrock)")

# --- Display Chat History ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Main Interaction ---
if prompt := st.chat_input("Ask me anything..."):
    # Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Format prompt for Mistral
    mistral_prompt = format_mistral_prompt(prompt)

    # Response placeholder
    response_box = st.chat_message("assistant")
    placeholder = response_box.empty()
    handler = StreamHandler(placeholder)

    try:
        if llm.streaming:
            _ = llm.invoke(mistral_prompt, callbacks=[handler])
            response = handler.content
        else:
            response = llm.invoke(mistral_prompt)
            response = clean_mistral_output(response.strip().strip('"'))
            placeholder.markdown(response)
    except Exception as e:
        response = f"(Error: {str(e)})"
        placeholder.markdown(response)

    # Save assistant reply
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Rough token estimate
    estimated_tokens = len(prompt.split()) + len(response.split())
    st.session_state.token_count += estimated_tokens

# --- Token Tracker ---
st.sidebar.markdown(f"🧮 **Estimated tokens used:** `{st.session_state.token_count}`")
