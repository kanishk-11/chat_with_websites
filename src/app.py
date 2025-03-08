__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv

# Import from our modules
from vector_store import update_vectorstore_with_url
from retrieval import get_context_retriever_chain
from conversation import get_conversational_rag_chain
from utils import initialize_session_state

# Load environment variables
load_dotenv()

def get_response(user_input):
    """
    Generate a response to the user's input using the RAG chain.
    
    Args:
        user_input (str): The user's query text
        
    Returns:
        str: The AI's response based on the retrieved context
    """
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']

def main():
    """Main application entry point."""
    # app config
    st.set_page_config(page_title="Chat with Webpages", page_icon="💻")
    st.title("Chat with Webpages")
    
    # Initialize session state
    initialize_session_state()
    
    # sidebar
    with st.sidebar:
        st.header("Manage Websites")
        website_url = st.text_input("Add Primary Website URL")
        
        # Add URL button for the first URL
        if website_url and st.button("Set as Primary URL"):
            if "vectorstores" not in st.session_state or len(st.session_state.vectorstores) == 0:
                with st.spinner("Loading primary website content..."):
                    success_msg = update_vectorstore_with_url(website_url)
                    st.success(success_msg)
            else:
                st.warning("Primary URL already added. Use 'Add Another Website URL' below.")
        
        # Only show the second URL input if the first one has been added
        if "vectorstores" in st.session_state and len(st.session_state.vectorstores) > 0:
            st.divider()
            additional_url = st.text_input("Load Another Website URL")
            
            # Add URL button for additional URLs
            if additional_url and st.button("Add URL to Knowledge"):
                # Check if URL is already added
                if additional_url in st.session_state.loaded_urls:
                    st.warning(f"URL already added: {additional_url}")
                else:
                    with st.spinner(f"Loading additional content from {additional_url}..."):
                        success_msg = update_vectorstore_with_url(additional_url)
                        st.success(success_msg)
            
            # Display currently loaded URLs
            if "loaded_urls" in st.session_state and len(st.session_state.loaded_urls) > 0:
                st.divider()
                st.subheader("Currently Loaded Websites")
                for i, url in enumerate(st.session_state.loaded_urls):
                    st.write(f"{i+1}. {url}")

    if "vectorstores" not in st.session_state or len(st.session_state.vectorstores) == 0:
        st.info("Please enter a primary website URL and click 'Set as Primary URL'")
    else:
        # session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content="Hi! Ask me anything about the websites you've loaded."),
            ]

        # user input
        user_query = st.chat_input("Type your message here...")
        if user_query is not None and user_query != "":
            response = get_response(user_query)
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))
            
        # conversation
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)

if __name__ == "__main__":
    main()
