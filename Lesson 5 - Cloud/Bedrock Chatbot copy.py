import os
import json
import boto3
import streamlit as st

# --- AWS Setup ---
os.environ["AWS_PROFILE"] = "robbarto"
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-2")

GUARDRAIL_IDENTIFIER = os.getenv("BEDROCK_GUARDRAIL_IDENTIFIER", "").strip()
GUARDRAIL_VERSION = os.getenv("BEDROCK_GUARDRAIL_VERSION", "DRAFT").strip()

# --- Mistral 7B Instruct Model ID ---
#MISTRAL_MODEL_ID = "mistral.mistral-7b-instruct-v0:2"
CLAUDE_HAIKU_ARN = "arn:aws:bedrock:us-east-2:368661395607:inference-profile/global.anthropic.claude-haiku-4-5-20251001-v1:0"
CLAUDE_HAIKU_GLOBAL_PROFILE = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
MODEL_ID = CLAUDE_HAIKU_ARN

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "token_count" not in st.session_state:
    st.session_state.token_count = 0

# --- Format Prompt for Mistral ---
def format_mistral_prompt(user_prompt):
    return user_prompt.strip()

# --- Clean Output ---
def clean_mistral_output(text):
    return text.strip()


def invoke_claude(messages, max_tokens, temperature):
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "anthropic_version": "bedrock-2023-05-31",
    }

    invoke_kwargs = {
        "modelId": MODEL_ID,
        "contentType": "application/json",
        "accept": "application/json",
        "body": json.dumps(payload),
    }
    if st.session_state.get("guardrails_enabled") and GUARDRAIL_IDENTIFIER:
        invoke_kwargs["guardrailIdentifier"] = GUARDRAIL_IDENTIFIER
        invoke_kwargs["guardrailVersion"] = GUARDRAIL_VERSION
        if st.session_state.get("guardrails_trace"):
            invoke_kwargs["trace"] = "ENABLED"

    response = bedrock_client.invoke_model(**invoke_kwargs)

    result = json.loads(response["body"].read())
    content = result.get("content") or []
    if not content:
        return ""
    return content[0].get("text", "")

# --- Sidebar Controls ---
st.sidebar.title("Settings")
st.sidebar.markdown("**Model:**")
st.sidebar.code(MODEL_ID)
st.sidebar.markdown(f"**Inference profile ARN:** `{CLAUDE_HAIKU_ARN}`")
st.sidebar.markdown(f"**Global inference profile:** `{CLAUDE_HAIKU_GLOBAL_PROFILE}`")
st.sidebar.markdown("**Guardrails:**")
st.session_state.guardrails_enabled = st.sidebar.toggle(
    "Enable Guardrail",
    value=bool(GUARDRAIL_IDENTIFIER),
)
st.sidebar.caption(
    f"Identifier: `{GUARDRAIL_IDENTIFIER or 'Not set (BEDROCK_GUARDRAIL_IDENTIFIER)'}`\n\n"
    f"Version: `{GUARDRAIL_VERSION}`"
)
st.session_state.guardrails_trace = st.sidebar.toggle("Enable Guardrail trace", value=False)
temperature = st.sidebar.slider("Creativity (temperature)", 0.0, 1.5, 0.9, 0.1)

if st.sidebar.button("🔄 Reset Chat"):
    st.session_state.chat_history = []
    st.session_state.token_count = 0

# --- Page Title ---
st.title("🧠 Claude Haiku Chatbot (via AWS Bedrock)")

# --- Display Chat History ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Main Interaction ---
if prompt := st.chat_input("Ask me anything..."):
    # Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    messages = []
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            messages.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "assistant":
            messages.append({"role": "assistant", "content": msg["content"]})

    # Response placeholder
    response_box = st.chat_message("assistant")
    placeholder = response_box.empty()

    try:
        response = invoke_claude(messages=messages, max_tokens=1000, temperature=temperature)
        response = clean_mistral_output(response)
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
