import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_objectbox.vectorstores import ObjectBox
from langchain.chains import LLMChain, RetrievalQA,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from transformers import pipeline

# Initialize Streamlit
st.title('Simple ObjectBox LLM Demo')

# Load Hugging Face model
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="gpt2")

llm_pipeline = load_model()

# Define a simple LLM function
def llm(input_text):
    try:
        output = llm_pipeline(input_text)
        return output[0]['generated_text']
    except Exception as e:
        return f"Error: {str(e)}"

# Function to load and embed documents
def load_and_embed_docs():
    try:
        # Load PDF documents from a directory
        docs = PyPDFDirectoryLoader('./census').load()
        
        # Split text into chunks
        final_docs = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        for doc in docs:
            if doc.page_content:
                final_docs.extend(Document(page_content=chunk) for chunk in text_splitter.split_text(doc.page_content))
        
        # Initialize embeddings and store in ObjectBox
        embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        vector_store = ObjectBox.from_documents(embedding=embeddings, embedding_dimensions=768, documents=final_docs)
        
        # Save vector store in session state
        st.session_state.db = vector_store
    except Exception as e:
        st.error(f"Error loading and embedding documents: {str(e)}")

# Button to load and embed documents
if st.button("Load and Embed Documents"):
    load_and_embed_docs()
    st.success("Documents have been embedded!")

# Input for user questions
input_text = st.text_input("Ask a question")

if input_text and 'db' in st.session_state:
    retriever = st.session_state.db.as_retriever()
    
    prompt = PromptTemplate.from_template("""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer: """)
    
    runnable_lambda = RunnableLambda(lambda input: llm(input))
    qa_chain = create_stuff_documents_chain(runnable_lambda, prompt)
    retrieval_chain = create_retrieval_chain(retriever, qa_chain)
    
    response = retrieval_chain.invoke({"input": input_text,"context": ""})
    
    # Show relevant documents in an expander
    with st.expander("Relevant Documents"):
        for doc in retriever.get_relevant_documents(input_text):
            st.write(doc.page_content)