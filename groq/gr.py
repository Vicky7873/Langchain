import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
import time

load_dotenv()

groq_api = os.environ['GROQ_API_KEY']

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings()
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_split = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    st.session_state.final_docs = st.session_state.text_split.split_documents(st.session_state.docs)

    st.session_state.vector = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)


st.title("Groq Demo")
llm = ChatGroq(api_key=groq_api,model='mixtral-8x7b-32768')

prompt = ChatPromptTemplate.from_template(
    """
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
I will tip you $1000 if the user finds the answer helpful. 
Question: {input}"""
)

doc_chain = create_stuff_documents_chain(llm,prompt)
db_ret = st.session_state.vector.as_retriever()
ret_chain = create_retrieval_chain(doc_chain,db_ret)

prompt = st.text_input('Input you prompt here')

if prompt:
    start = time.process_time()
    
    # Get relevant documents if you still want to use retrieval
    context = db_ret.get_relevant_documents(prompt)
    
    # The context is now not being used in the prompt
    response = ret_chain.invoke({"input": prompt})
    
    print("Response time :", time.process_time() - start)
    st.write(response['answer'])

    # Optionally, you can still display the retrieved documents
    with st.expander("Document Similarity Search"):
        for doc in context:
            st.write(doc.page_content)
            st.write("--------------------------------")


