# --- LangChain + Ollama RAG Pipeline with WORKING LangSmith Tracing ---
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set all required environment variables for LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"  
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Agentic Tracing"  # Optional: specify project name

print("LangSmith tracing is enabled. View your traces at https://smith.langchain.com/")

# Modern imports - no deprecation warnings
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langsmith import traceable
from langsmith import Client  # Import the LangSmith Client for feedback logging
from langsmith.run_helpers import trace  # Use the correct tracing context manager

# --- Agentic Pipeline with Tools ---
print("\nSetting up agentic pipeline...")

from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain_community.utilities import WikipediaAPIWrapper
from langchain import hub

# Load and process local document for agent
file_path2 = os.path.join(os.path.dirname(__file__), "Noclimate.txt")
loader2 = TextLoader(file_path2)
docs2 = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(loader2.load())
vs2 = Chroma.from_documents(docs2, OllamaEmbeddings(model="mxbai-embed-large"), persist_directory="lc_agent_chroma")
retriever2 = vs2.as_retriever(search_type="mmr", search_kwargs={"k": 4})

# Build QA chain
llm2 = OllamaLLM(model="llama3")
rag_chain = RetrievalQA.from_chain_type(llm=llm2, retriever=retriever2)

# Add Wikipedia tool
wiki_tool = WikipediaAPIWrapper()

# Define tools
tools = [
    Tool(
        name="LocalDocQA",
        func=lambda query: rag_chain.invoke({"query": query})["result"],
        description="Use this tool to answer questions about climate change using the local document."
    ),
    Tool(
        name="WikipediaSearch",
        func=wiki_tool.run,
        description="Use this when the local document does not seem to contain the information."
    )
]

# Get the react prompt template
try:
    prompt = hub.pull("hwchase17/react")
    print("Successfully loaded ReAct prompt from hub")
except Exception as e:
    print(f"Failed to load prompt from hub: {e}")
    # Fallback prompt if hub is unavailable
    from langchain.prompts import PromptTemplate
    prompt = PromptTemplate(
        input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
        template="""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format exactly (do not skip any lines):

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT: After every 'Action:' line, you must output 'Action Input:' on the next line, even if the input is empty.

Begin!

Question: {input}
{agent_scratchpad}"""
    )

# Create the agent
agent = create_react_agent(llm2, tools, prompt)

# Create agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # Enable verbose to see agent thinking
    handle_parsing_errors=True,
    max_iterations=3
)

# Run the agent with LangSmith tracing context to capture run_id
print("Running agentic query...")
with trace(name="Agentic Query") as run:
    response = agent_executor.invoke({"input": "What causes the most CO2 emissions?"})
    run_id = run.id  # This is the run_id for LangSmith feedback
print("[Agentic LangChain]", response["output"])

# --- Collect and log feedback to LangSmith ---
try:
    # Prompt user for feedback (1-5)
    while True:
        feedback = input("\nPlease rate the response (1-5, where 5=best): ").strip()
        if feedback in {"1", "2", "3", "4", "5"}:
            feedback = int(feedback)
            break
        print("Invalid input. Please enter a number from 1 to 5.")

    if run_id is not None:
        client = Client()
        client.create_feedback(
            run_id=run_id,
            key="user_rating",
            score=feedback,
            comment=f"User rated the response {feedback}/5."
        )
        print(f"\n✅ Feedback ({feedback}/5) logged to LangSmith for run_id: {run_id}")
        print("You can view feedback on your LangSmith project page.")
    else:
        print("\n⚠️ Could not find run_id. Feedback not logged to LangSmith.")
except Exception as e:
    print(f"\n⚠️ Error logging feedback to LangSmith: {e}")

print("\n" + "="*50)
print("🎉 Execution complete!")
print("Check your LangSmith dashboard at: https://smith.langchain.com/")
print("Look for the 'default' project to see your traces.")
print("="*50)