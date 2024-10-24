import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_objectbox.vectorstores import ObjectBox
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')

st.title('Demo with objetbox')

llm = ChatGroq(
    model= 'Llama3-8b-8192',
    api_key=groq_api_key
)

prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}

    """

)

def vector_embeddings():
    if 'db' not in st.session_state:
        st.session_state.loader = PyPDFDirectoryLoader('./census')
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_split = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap = 200)
        st.session_state.final_docs = []
        for doc in st.session_state.docs[:20]:
            text = doc.page_content  # Assuming doc[1] contains the text content
            st.session_state.final_docs.append(Document(page_content=text))
        # st.session_state.final_docs = st.session_state.text_split.split_documents(st.session_state.docs[:20])
        st.session_state.embeddings = HuggingFaceBgeEmbeddings(
            model_name = "BAAI/bge-small-en-v1.5",
            model_kwargs = {"device":"cpu"},
            encode_kwargs = {"normalize_embeddings":True}
        )

        st.session_state.db = ObjectBox.from_documents(st.session_state.embeddings,st.session_state.final_docs,embedding_dimensions=768)


input_text = st.text_input("Enter Your Question From Documents")
if st.button("Documents Embedding"):
    vector_embeddings()
    st.write("ObjectBox Database is ready")

import time

if input_text and 'db' in st.session_state:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.db.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()

    response=retrieval_chain.invoke({'input':input_text})

    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")


