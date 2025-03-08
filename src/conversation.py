from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

def get_conversational_rag_chain(retriever_chain):
    """
    Create a conversational retrieval-augmented generation chain.
    
    Args:
        retriever_chain: A LangChain retriever chain that gets documents based on conversation history
        
    Returns:
        A chain that combines retrieval with conversation to answer queries based on retrieved documents
    """
    llm = ChatOpenAI()
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based ONLY ON THE BELOW CONTEXT THAT IS GIVEN. "
                "WHATEVER YOU KNOW OUTSIDE OF THIS CONTEXT IS ALSO FALSE, AND YOU MUST REPLY WITH "
                "PLEASE ASK CONTEXT-RELATED QUESTIONS. IF THE QUESTIONS CONTEXT IS NOT IN THIS CONTEXT, "
                "REPLY PLEASE ASK CONTEXT-RELATED QUESTIONS.:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)