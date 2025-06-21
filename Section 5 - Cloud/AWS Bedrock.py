# Import necessary modules from LangChain, Boto3, OS, and Streamlit
from langchain.chains import LLMChain
from langchain_community.chat_models import BedrockChat
from langchain.prompts import PromptTemplate
import boto3
import os
import streamlit as st

# Set the AWS profile to use for authentication
os.environ["AWS_PROFILE"] = "robbarto"

# Initialize the Bedrock client using boto3 for the specified region
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

# Specify the Claude 3.5 Sonnet model ID for Bedrock
modelID = "anthropic.claude-3-5-sonnet-20240620-v1:0"

# Initialize the BedrockChat LLM with the specified model, client, and generation parameters
llm = BedrockChat(
    model_id=modelID,
    client=bedrock_client,
    model_kwargs={"max_tokens": 2000, "temperature": 0.9}
)

# Define a function for the chatbot that takes language and freeform_text as input
# It uses a prompt template to format the input for the LLM

def my_chatbot(language, freeform_text):
    prompt = PromptTemplate(
        input_variables=["language", "freeform_text"],
        template="You are a chatbot. You are in {language}.\n\n{freeform_text}"
    )

    bedrock_chain = LLMChain(llm=llm, prompt=prompt)

    response=bedrock_chain({'language':language, 'freeform_text':freeform_text})
    return response

#print(my_chatbot("english","who is buddha?"))

st.title("Bedrock Chatbot")

language = st.sidebar.selectbox("Language", ["english", "japanese", "chinese", "korean"])

if language:
    freeform_text = st.sidebar.text_area(label="what is your question?",
    max_chars=100)

if freeform_text:
    response = my_chatbot(language,freeform_text)
    st.write(response['text'])