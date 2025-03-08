from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever

def get_context_retriever_chain(vector_store):
    """
    Create a retriever chain that is aware of conversation history.
    
    This chain takes the conversation history and current input to generate
    a search query, which is then used to retrieve relevant documents.
    
    Args:
        vector_store: The vector store to retrieve documents from
        
    Returns:
        A history-aware retriever chain
    """
    llm = ChatOpenAI()
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain