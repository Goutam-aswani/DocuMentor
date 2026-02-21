# # chatbot_service.py

# from langchain_google_genai import ChatGoogleGenerativeAI
# # *** CHANGE: Import MessagesPlaceholder to handle a list of messages in the prompt. ***
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.output_parsers import StrOutputParser
# # *** CHANGE: Import message types to structure the chat history. ***
# from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# from config import settings
# from typing import List

# # *** CHANGE: The function now accepts a 'chat_history' list as an argument. ***
# def get_chatbot_response(prompt: str, chat_history: List[BaseMessage]):
#     """
#     Initializes the chatbot model and gets a streaming response,
#     now with conversational context.
#     Args:
#         prompt: The new message from the user.
#         chat_history: A list of previous messages (HumanMessage, AIMessage)
#     Yields:
#         str: Chunks of the bot's response.
#     """
#     if not settings.google_api_key:
#         raise ValueError("GOOGLE_API_KEY is not set in the environment.")

#     llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=settings.google_api_key)
    
#     # *** CHANGE: The prompt template is updated to include a placeholder for chat history. ***
#     # This tells the AI to consider the previous messages before answering the new one.
#     prompt_template = ChatPromptTemplate.from_messages([
#         ("system", (
#             "You are a helpful assistant. You provide concise answers based on the provided context. "
#             "Format your responses using GitHub-flavored Markdown."
#         )),
#         # The placeholder will be filled with the list of previous messages.
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("user", "{input}")
#     ])
    
#     output_parser = StrOutputParser()
    
#     chain = prompt_template | llm | output_parser
    
#     try:
#         # DEBUG: Print the prompt and history being sent.
#         print(f"--- DEBUG: Streaming with prompt: '{prompt}' ---")
#         print(f"--- DEBUG: Streaming with history: {chat_history} ---")

#         # *** CHANGE: The 'stream' call now includes the chat_history. ***
#         response_stream = chain.stream({
#             "input": prompt,
#             "chat_history": chat_history
#         })

#         for chunk in response_stream:
#             yield chunk

#     except Exception as e:
#         print(f"--- DEBUG: Error streaming from LangChain: {e}")
#         yield "Sorry, I encountered an error while processing your request."



# # chatbot_service.py

import os
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from rag_service import get_session_retriever,get_session_retriever_with_scores
from web_search_service import TavilySearchService
from config import settings
from typing import List, Optional
from operator import itemgetter


# Model configurations
MODELS = {
    # Groq models
    "llama-3.1-8b-instant": {
        "provider": "groq",
        "model_name": "llama-3.1-8b-instant"
    },
    "llama-3.3-70b-versatile": {
        "provider": "groq",
        "model_name": "llama-3.3-70b-versatile"
    },
    "meta-llama/llama-4-scout-17b-16e-instruct": {
        "provider": "groq",
        "model_name": "meta-llama/llama-4-scout-17b-16e-instruct"
    },
    "moonshotai/kimi-k2-instruct": {
        "provider": "groq",
        "model_name": "moonshotai/kimi-k2-instruct"
    },
    "qwen/qwen3-32b": {
        "provider": "groq",
        "model_name": "qwen/qwen3-32b"
    },
    # OpenRouter models (free)
    "z-ai/glm-4.5-air:free": {
        "provider": "openrouter",
        "model_name": "z-ai/glm-4.5-air:free"
    },
    "mistralai/mistral-small-3.1-24b-instruct:free": {
        "provider": "openrouter",
        "model_name": "mistralai/mistral-small-3.1-24b-instruct:free"
    },
}


def get_model_instance(model_name: str = "llama-3.3-70b-versatile"):
    """
    Get the appropriate model instance based on model name
    """
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODELS.keys())}")
    
    model_config = MODELS[model_name]
    provider = model_config["provider"]
    actual_model_name = model_config["model_name"]
    
    if provider == "google":
        return ChatGoogleGenerativeAI(
            model=actual_model_name,
            google_api_key=settings.google_api_key
        )
    elif provider == "groq":
        return ChatGroq(
            model=actual_model_name,
            groq_api_key=settings.grok_api_key
        )
    elif provider == "openrouter":
        return ChatOpenAI(
            model=actual_model_name,
            openai_api_key=settings.openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1"
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_chatbot_response(
    prompt: str, 
    chat_history: List[BaseMessage], 
    model_name: str = "llama-3.3-70b-versatile",
    use_web_search: bool = False
):
    """
    Initializes the chatbot model and gets a streaming response,
    now with conversational context, model selection, and optional web search.
    """
    llm = get_model_instance(model_name)
    
    # Initialize web search if enabled
    web_search_context = ""
    if use_web_search:
        search_service = TavilySearchService()
        search_results = search_service.search(prompt)
        if search_results:
            web_search_context = search_service.format_search_results(search_results)
    
    # Create system prompt with optional web search context
    system_message = (
        "You are a helpful assistant. You provide concise answers based on the provided context. "
        "Format your responses using GitHub-flavored Markdown."
    )
    
    if web_search_context:
        system_message += f"\n\nHere is some relevant information from web search:\n\n{web_search_context}"
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
   
    output_parser = StrOutputParser()
    chain = prompt_template | llm | output_parser
   
    try:
        print(f"--- DEBUG: Using model: {model_name} ---")
        print(f"--- DEBUG: Web search enabled: {use_web_search} ---")
        print(f"--- DEBUG: Streaming with prompt: '{prompt}' ---")
        print(f"--- DEBUG: Chat history length: {len(chat_history)} ---")

        response_stream = chain.stream({
            "input": prompt,
            "chat_history": chat_history
        })

        for chunk in response_stream:
            yield chunk

    except Exception as e:
        print(f"--- DEBUG: Error streaming from LangChain: {e}")
        yield "Sorry, I encountered an error while processing your request."




def check_document_relevance(retriever, query: str, min_docs: int = 2) -> tuple[list, bool]:
    """
    Check if enough relevant documents are retrieved for the query.
    Returns: (documents_list, is_sufficient_bool)
    """
    try:
        retrieved_docs = retriever.get_relevant_documents(query)
        print(f"--- DEBUG: Retrieved {len(retrieved_docs)} documents after filtering ---")
        
        is_sufficient = len(retrieved_docs) >= min_docs
        
        if not is_sufficient:
            print(f"--- INFO: Insufficient documents ({len(retrieved_docs)}) for query, minimum required: {min_docs} ---")
            # return retrieved_docs[0], True
        print(retrieved_docs[0].page_content)
        return retrieved_docs, is_sufficient
        
    except Exception as e:
        print(f"--- ERROR: Document retrieval failed: {e} ---")
        return [], False
    



def get_rag_chatbot_response(
    prompt: str, 
    chat_history: List[BaseMessage], 
    session_id: int,
    model_name: str = "llama-3.3-70b-versatile",
    use_web_search: bool = False
):
    """
    Generates a streaming RAG response using conversational context and retrieved documents
    filtered by the session_id with model selection and optional web search.
    """
    model = get_model_instance(model_name)
    
    # Initialize web search if enabled
    web_search_context = ""
    if use_web_search:
        search_service = TavilySearchService()
        search_results = search_service.search(prompt)
        if search_results:
            web_search_context = search_service.format_search_results(search_results)

    retriever = get_session_retriever_with_scores(session_id, similarity_threshold=0.95)

    retrieved_docs, has_sufficient_docs = check_document_relevance(retriever, prompt, min_docs=1)
    if not has_sufficient_docs:
        print(f"--- INFO: Rejecting query due to insufficient relevant documents ---")
        def rejection_generator():
            yield "I cannot answer this question as I don't find sufficient relevant information in the uploaded documents. Please ensure your question is related to the content of the uploaded files."
        return rejection_generator()
    
    def format_docs(docs_list):
        if not docs_list:
            return "No relevant documents found."
        return "\n\n".join(doc.page_content for doc in docs_list)

    # Create system prompt with optional web search context
    system_message = (
        """You are a document analysis assistant. Your ONLY job is to answer questions based STRICTLY on the provided context from uploaded documents.

CRITICAL RULES:
1. You MUST ONLY use information that is explicitly stated in the context provided below
2. If the answer is not in the context, you MUST respond with: "I cannot answer this question as the information is not available in the uploaded documents."
3. Do NOT use any general knowledge, assumptions, or information from outside the provided context
4. Do NOT provide partial answers by mixing document content with general knowledge
5. Do NOT make inferences beyond what is explicitly stated in the documents

BEFORE answering any question:
- First, carefully scan the provided context
- Verify that the specific information needed to answer the question exists in the context
- If you cannot find the specific information, use the rejection response above

FORMAT REQUIREMENTS:
- Use GitHub-flavored Markdown for formatting
- Be concise and direct
- Quote relevant sections when possible
- If answering, cite which part of the document the information comes from

CONTEXT:
{context}

THE CONTEXT IS YOUR KNOWLEDGE BASE THE CONTEXT IS ALL YOU KNOW

Remember: It's better to say "I don't know based on the documents" than to provide information not found in the context."""
    )
    
    if web_search_context:
        system_message += f"\n\nAdditional web search information (use only if relevant to document context):\n\n{web_search_context}"

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    print(f"--- DEBUG: Creating RAG chain for session_id: {session_id} with model: {model_name} ---")
    print(f"--- DEBUG: Web search enabled: {use_web_search} ---")
    
    rag_chain = (
        {
            "context": lambda x: format_docs(retrieved_docs),
            "input": itemgetter("input"),
            "chat_history": itemgetter("chat_history"),
        }
        | rag_prompt
        | model
        | StrOutputParser()
    )
    
    print(f"--- DEBUG: RAG chain created for session_id: {session_id} ---")
    return rag_chain.stream({
        "input": prompt,
        "chat_history": chat_history
    })

