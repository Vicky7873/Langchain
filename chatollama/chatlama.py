import os
import streamlit as st
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
load_dotenv()

# os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
# os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2')\


api_key = os.getenv('LANGCHAIN_API_KEY')
if api_key:
    os.environ['LANGCHAIN_API_KEY'] = api_key
else:
    print("LANGCHAIN_API_KEY environment variable not found.")

# Verify API Tracing Configuration
tracing_v2 = os.getenv('LANGCHAIN_TRACING_V2')
if tracing_v2:
    os.environ['LANGCHAIN_TRACING_V2'] = tracing_v2
else:
    print("LANGCHAIN_TRACING_V2 environment variable not found.")

st.title("ChatOllama Demo")
st.write("this is a simple chatbot using chatollama")

with st.form("llm-form"):
    input_text = st.text_area("Enter your Question")
    submit = st.form_submit_button("Submit")

def genrate_response(text):
    llm = ChatOllama(model='llama3.2:latest')
    response = llm.invoke(text)
    return response

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if submit and input_text:
    with st.spinner("Getting response..."):
        response = genrate_response(input_text)
        st.session_state['chat_history'].append({"User":input_text,"response":response})
        st.write(response)
    
# st.write("## Chat History")
# for chat in reversed(st.session_state['chat_history']):
#     st.write(f"**🧑 User**: {chat['User']}")
#     st.write(f"**🧠 Assistant**: {chat['response']}")
#     st.write("---")