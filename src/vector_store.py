import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

class MergedRetriever:
    """
    A retriever that combines results from multiple vector stores.
    
    This class implements the LangChain retriever interface to allow
    querying multiple vector stores simultaneously and combining their results.
    """
    
    def __init__(self, vectorstores):
        """
        Initialize with a list of vector stores.
        
        Args:
            vectorstores (list): List of LangChain vectorstore objects.
        """
        self.retrievers = [vs.as_retriever() for vs in vectorstores]
    
    def get_relevant_documents(self, query):
        """
        Retrieve relevant documents from all vector stores.
        
        Args:
            query (str): The query to search for.
            
        Returns:
            list: Combined list of relevant documents from all retrievers.
        """
        all_docs = []
        for retriever in self.retrievers:
            docs = retriever.get_relevant_documents(query)
            all_docs.extend(docs)
        return all_docs
    
    def invoke(self, query):
        """
        Invoke the retriever with a query (compatibility with newer LangChain versions).
        
        Args:
            query (str): The query to search for.
            
        Returns:
            list: Combined list of relevant documents from all retrievers.
        """
        return self.get_relevant_documents(query)

def get_vectorstore_from_url(url, persistent_dir=None):
    """
    Create a vector store from the content of a given URL.
    
    Args:
        url (str): The URL to load content from.
        persistent_dir (str, optional): Directory to persist the vector store.
            If None, the vector store will not be persisted.
    
    Returns:
        Chroma: A vector store containing the chunked documents from the URL.
    """
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()
    
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    
    # create a vectorstore from the chunks
    if persistent_dir:
        vector_store = Chroma.from_documents(
            documents=document_chunks, 
            embedding=OpenAIEmbeddings(),
            persist_directory=persistent_dir
        )
    else:
        vector_store = Chroma.from_documents(
            documents=document_chunks, 
            embedding=OpenAIEmbeddings()
        )
    
    return vector_store

def merge_vectorstores(vectorstores):
    """
    Merge multiple vector stores into a single one with a combined retriever.
    
    Args:
        vectorstores (list): List of vector stores to merge
    
    Returns:
        A vector store with a retriever that queries all input vector stores
    """
    # If there's only one vectorstore, return it directly
    if len(vectorstores) == 1:
        return vectorstores[0]
    
    # For multiple vectorstores, we'll combine their retrievers
    combined_retriever = MergedRetriever(vectorstores)
    
    # Return the first vectorstore but with the combined retriever
    vectorstores[0]._retriever = combined_retriever
    return vectorstores[0]

def update_vectorstore_with_url(url):
    """
    Update the vector store with content from a new URL.
    
    Args:
        url (str): The URL to load content from.
        
    Returns:
        str: Success message if the update was successful.
    """
    # Initialize vectorstores if not already done
    if "vectorstores" not in st.session_state:
        st.session_state.vectorstores = []
    
    # Initialize loaded_urls if not already done
    if "loaded_urls" not in st.session_state:
        st.session_state.loaded_urls = []
    
    # Add URL to the list of loaded URLs
    st.session_state.loaded_urls.append(url)
    
    # Create a new persistent directory for each URL
    persistent_dir = f"./chroma_db_{len(st.session_state.vectorstores)}"
    
    # Get vectorstore for the new URL
    new_vectorstore = get_vectorstore_from_url(url, persistent_dir)
    
    # Add to list of vectorstores
    st.session_state.vectorstores.append(new_vectorstore)
    
    # Merge all vectorstores
    st.session_state.vector_store = merge_vectorstores(st.session_state.vectorstores)
    
    # Return success message
    return f"Successfully loaded content from: {url}"