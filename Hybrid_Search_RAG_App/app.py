import os
import streamlit as st
import torch
from config import *
from retriever import initialize_pinecone, create_hybrid_retriever
from document_processor import convert_pdf_to_markdown, process_and_chunk_document
from llm_chain import initialize_llm, create_retrieval_chain
st.set_page_config(layout="wide")


st.title("Ask PDF")

@st.cache_resource
def setup_resources():
    # Initialize Pinecone
    index = initialize_pinecone(PINECONE_API_KEY, INDEX_NAME)
    # Create retriever
    retriever, embeddings = create_hybrid_retriever(index, MODEL_NAME)
    # Initialize LLM
    llm = initialize_llm(LLM_REPO_ID)
    return retriever, embeddings, llm

retriever, embeddings, llm = setup_resources()
rag_chain = create_retrieval_chain(retriever, llm)

# File upload handling
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file:
    # Initialize processed_files dictionary if it doesn't exist
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = {}

    # Check if this specific file needs processing
    if uploaded_file.name not in st.session_state.processed_files:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            # Save uploaded file
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process document
            markdown_content = convert_pdf_to_markdown(temp_path)
            cache_filename = os.path.splitext(uploaded_file.name)[0] + "_chunks.pkl"
            docs = process_and_chunk_document(markdown_content, embeddings, cache_filename)
            
            # Index documents
            retriever.add_texts([doc.page_content for doc in docs])
            
            # Store in session state
            file_entry = {
                "docs": docs,
                "content": "\n\n".join([doc.page_content for doc in docs])
            }
            
            # Update processed_files with new entry
            st.session_state.processed_files[uploaded_file.name] = file_entry

    # Update current document display
    current_file = st.session_state.processed_files[uploaded_file.name]
    st.session_state.docs = current_file["docs"]
    st.session_state.content = current_file["content"]

    # Question handling
    user_question = st.text_input("Enter your question:")
    if st.button("Ask") and user_question:
        st.write("**Question:**", user_question)
        with st.spinner("Generating answer..."):
            answer = ""
            for chunk in rag_chain.stream(user_question):
                answer += chunk
            st.markdown("**Answer:**")
            st.write(answer)
