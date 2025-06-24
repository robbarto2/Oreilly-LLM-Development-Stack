import os
from langsmith import traceable
from ollama import Client

# Set LangSmith env vars
os.environ["LANGCHAIN_TRACING_V2"] = "true"  
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Climate Research"  # Optional: specify project name


print("LangSmith tracing is enabled. View your traces at https://smith.langchain.com/")

# --- Custom traceable function for classification ---
@traceable(name="CO2 Category Lookup")
def classify_emission_source(query: str) -> str:
    if "transport" in query.lower():
        return "Transportation"
    elif "coal" in query.lower() or "power plant" in query.lower():
        return "Electric Power Generation"
    elif "beef" in query.lower() or "farming" in query.lower():
        return "Agriculture"
    else:
        return "Other"

# --- Traceable function to query Ollama LLM ---
@traceable(name="LLaMA 3 Response")
def query_llama(prompt: str) -> str:
    client = Client(host="http://localhost:11434")
    response = client.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# --- Run the demo ---
prompt = "What causes the most CO2 emissions?"
print(f"🧠 Prompt: {prompt}")

# Call traced functions
llm_response = query_llama(prompt)
print(f"💬 LLM Response:\n{llm_response}\n")

category = classify_emission_source(prompt)
print(f"📊 Emission Category: {category}")

# --- Done ---
print("\n" + "="*50)
print("🎉 Demo complete! Check your LangSmith project for both traced functions.")
print("https://smith.langchain.com/")
print("="*50)
