import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_objectbox.vectorstores import ObjectBox
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.documents import Document
import time
from langchain_chains import LLMObjectBoxChain

# Load environment variables
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')

if not groq_api_key:
    st.error("Groq API key is missing! Please provide a valid key.")
    st.stop()

st.title('Demo with ObjectBox')

# Set up the language model (LLM)
llm = ChatGroq(
    model='Llama3-8b-8192',
    api_key=groq_api_key
)

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Function to load and embed documents
def vector_embeddings():
    if 'db' not in st.session_state:
        loader = PyPDFDirectoryLoader('./census')
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        final_docs = []
        for doc in docs[:20]:
            split_texts = text_splitter.split_text(doc.page_content)
            for text in split_texts:
                final_docs.append(Document(page_content=text))
        
        embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

       

input_text = st.text_input("Enter Your Question From Documents")
if st.button("Documents Embedding"):
    vector_embeddings()
    st.write("ObjectBox Database is ready")

if input_text and 'db' in st.session_state:
    retriever = st.session_state.db.as_retriever()
    retrieval_chain = RetrievalQA(llm, retriever)

    start_time = time.process_time()
    response = retrieval_chain.invoke({'input': input_text})
    elapsed_time = time.process_time() - start_time

    st.write(f"Response time: {elapsed_time:.2f} seconds")
    st.write(response['answer'])

    # With a Streamlit expander
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response.get("source_documents", [])):
            # Handle tuple if necessary
            st.write(doc[0] if isinstance(doc, tuple) else doc.page_content)
            st.write("--------------------------------")
