import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def initialize_session_state():
    """Initialize the session state variables if they don't exist."""
    if "vectorstores" not in st.session_state:
        st.session_state.vectorstores = []
    
    if "loaded_urls" not in st.session_state:
        st.session_state.loaded_urls = []
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

# Default configuration settings
DEFAULT_SETTINGS = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "temperature": 0.7,
    "model_name": "gpt-4",
    "max_tokens": 1000,
    "persist_directory": "./chroma_db"
}

# Prompt templates
RETRIEVER_PROMPT_TEMPLATE = """
Given the above conversation, generate a search query to look up in order to get 
information relevant to the conversation
"""

ANSWER_PROMPT_TEMPLATE = """
Answer the user's questions based ONLY ON THE CONTEXT THAT IS GIVEN.
YOU ARE NOT SUPPOSED TO USE YOUR GENERAL KNOWLEDGE. IF YOU USE EXTERNAL KNOWLEDGE, THE SYSTEM WILL CRASH. CHECK IF THE QUESTION THAT IS ASKED HAS ANSWERS PRESENT IN THE GIVEN CONTEXT. IF IT IS NOT, REPLY WITH PLEASE ASK CONTEXT-RELATED QUESTIONS:

{context}
"""
