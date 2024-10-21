from fastapi import FastAPI, logging
from fastapi.middleware.cors import CORSMiddleware
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM 
from langserve import add_routes
import uvicorn
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

model = OllamaLLM(model="llama3.2:latest")
prompt = ChatPromptTemplate.from_template("You are a helpful assistant. Question: {question}") 
output = StrOutputParser()

app = FastAPI(
    title="Bhiki's Chatbot",
    description="Ask me anything",
    version="1.0.0",
    logs_level="debug"
)

add_routes(
    app,
    prompt|model|output,
    path="/chatbot"
)

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8501)