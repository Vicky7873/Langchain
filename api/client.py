import requests
import streamlit as st
import os
from langchain_core.output_parsers import StrOutputParser

def get_chatbot(input_text):
    response = requests.post(
        "http://localhost:8501/chatbot",
        json={'input':{'topic':input_text}}
    )

st.title("Bhiki's Chatbot")
input_text = st.text_input("Ask me anything")



if input_text:  
    st.write(get_chatbot(input_text))