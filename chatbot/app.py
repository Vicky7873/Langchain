from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM 
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# Validate API key
if "LANGCHAIN_API_KEY" not in os.environ:
    st.error("LANGCHAIN_API_KEY environment variable not set.")
    exit(1)

# Set environment variables
os.environ["LANGCHAIN_TRACING_V2"]=os.getenv('LANGCHAIN_TRACING_V2')
os.environ["LANGCHAIN_API_KEY"]=os.getenv('LANGCHAIN_API_KEY')

# Initialize components
prompt = ChatPromptTemplate.from_template("You are a helpful assistant. Question: {question}")
llm = OllamaLLM(model="llama3.2:latest")
output = StrOutputParser()
chain = prompt|llm|output

# Streamlit app
st.title("Bhiki's Chatbot")
input_text = st.text_input("Ask me anything")

if input_text:  
        response = chain.invoke({"question": input_text})
        st.write(response)
