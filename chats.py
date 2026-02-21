from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select
from typing import List, Optional
from pydantic import BaseModel
from fastapi import Body
from fastapi.responses import StreamingResponse
from limiter import limiter
from database import get_session
from models import User, ChatSession, ChatMessage
from dependencies import get_current_active_user
from chatbot_service import get_chatbot_response, get_rag_chatbot_response, MODELS
from slowapi import Limiter
from slowapi.util import get_remote_address
from langchain_core.messages import HumanMessage, AIMessage
from usage_tracker import UsageTracker

CHAT_RATE_LIMIT = "30/minute"
router = APIRouter(
    prefix="/chats",
    tags=["Chats"],
    dependencies=[Depends(get_current_active_user)],
    responses={404: {"description": "Not found"}},
)

# --- Pydantic Models (No changes) ---
class ChatMessageResponse(BaseModel):
    id: int
    content: str
    role: str

class ChatSessionResponse(BaseModel):
    id: int
    title: str

class ChatHistoryResponse(BaseModel):
    id: int
    title: str
    messages: List[ChatMessageResponse]

class NewChatMessageRequest(BaseModel):
    prompt: str
    session_id: Optional[int] = None
    model_name: Optional[str] = "gemini-2.5-flash"
    use_web_search: Optional[bool] = False

class RenameChatRequest(BaseModel):
    new_title: str

# # *** FIX: The function signature is corrected to accept chat_session_id (an int) ***
# async def stream_and_save_response(prompt: str, chat_session_id: int, chat_history: list, db: Session):
#     """
#     Streams the chatbot response and saves the full message.
#     Now accepts chat_session_id directly.
#     """
#     # *** FIX: Use the chat_session_id directly in the debug print statement ***
#     print(f"--- DEBUG: Starting stream for session {chat_session_id} ---")
    
#     # Pass the complete history to the chatbot service.
#     response_generator = get_chatbot_response(prompt, chat_history)
    
#     full_bot_response = ""
#     for chunk in response_generator:
#         full_bot_response += chunk
#         yield chunk
        
#     print(f"--- DEBUG: Finished streaming. Full response: '{full_bot_response}'")
    
#     bot_message_to_save = ChatMessage(
#         content=full_bot_response, 
#         role="model", 
#         # *** FIX: Use the chat_session_id directly when saving the message ***
#         session_id=chat_session_id
#     )
#     db.add(bot_message_to_save)
#     db.commit()
    
#     print(f"--- DEBUG: Saved bot response to DB for session {chat_session_id} ---")

# @router.post("/", summary="Post a new message and get a streaming response")
# def post_new_message(
#     *,
#     limiter: Limiter = Depends(lambda: limiter.limit(CHAT_RATE_LIMIT)),
#     request_data: NewChatMessageRequest,
#     session: Session = Depends(get_session),
#     current_user: User = Depends(get_current_active_user),
# ):
#     chat_session = None
#     if request_data.session_id:
#         chat_session = session.get(ChatSession, request_data.session_id)
#         if not chat_session or chat_session.user_id != current_user.id:
#             raise HTTPException(status_code=404, detail="Chat session not found")
#     else:
#         title = request_data.prompt[:100]
#         chat_session = ChatSession(title=title, user_id=current_user.id)
#         session.add(chat_session)
#         session.commit()
#         session.refresh(chat_session)

#     # *** FIX: Corrected History Assembly Logic ***
    
#     # 1. Fetch previous messages from the database.
#     statement = (
#         select(ChatMessage)
#         .where(ChatMessage.session_id == chat_session.id)
#         .order_by(ChatMessage.created_at)
#         .limit(10)
#     )
#     db_messages = session.exec(statement).all()

#     # 2. Format the fetched history for LangChain.
#     chat_history_for_chain = []
#     for msg in db_messages:
#         if msg.role == "user":
#             chat_history_for_chain.append(HumanMessage(content=msg.content))
#         elif msg.role == "model":
#             chat_history_for_chain.append(AIMessage(content=msg.content))

#     # 3. Save the NEW user message to the database.
#     user_message = ChatMessage(content=request_data.prompt, role="user", session_id=chat_session.id)
#     session.add(user_message)
#     session.commit()
#     print(f"--- DEBUG: User message '{request_data.prompt}' saved for session {chat_session.id}")

#     # *** FIX: Manually add the new user message to the history list AFTER saving it. ***
#     # This ensures the AI gets the absolute latest context.
#     chat_history_for_chain.append(HumanMessage(content=request_data.prompt))

#     # 4. Pass the prompt, session ID, and the correctly assembled history to the streaming function.
#     return StreamingResponse(
#         stream_and_save_response(request_data.prompt, chat_session.id, chat_history_for_chain, session),
#         media_type="text/plain; charset=utf-8"
#     )
# *** CHANGE: Modified to return session ID in response headers ***
def stream_and_save_response_with_headers(
    prompt: str, 
    chat_session_id: int, 
    chat_history: list, 
    db_session: Session,
    model_name: str = "gemini-2.5-flash",
    use_web_search: bool = False,
    current_user:User = Depends(get_current_active_user)
):
    """
    Streams the chatbot response and saves the full message.
    Now yields both content and session metadata with model selection and web search.
    """
    print(f"--- DEBUG: Starting stream for session {chat_session_id} with model {model_name} ---")
    print(f"--- DEBUG: Web search enabled: {use_web_search} ---")
    print(f"--- DEBUG: Chat history length being passed: {len(chat_history)} ---")

    chat_session = db_session.get(ChatSession, chat_session_id)
    if not chat_session:
        # This is a fallback, though the main endpoint should prevent this
        raise HTTPException(status_code=404, detail="Chat session not found during streaming")

    # THE CORE LOGIC: Decide which response generator to use
    if chat_session.has_documents:
        print(f"--- INFO: Using RAG chain for session {chat_session_id} ---")
        response_generator = get_rag_chatbot_response(
            prompt, chat_history, chat_session_id, model_name, use_web_search
        )
    else:
        print(f"--- INFO: Using standard chain for session {chat_session_id} ---")
        response_generator = get_chatbot_response(
            prompt, chat_history, model_name, use_web_search
        )

    full_bot_response = ""
    for chunk in response_generator:
        full_bot_response += chunk
        yield chunk
       
    print(f"--- DEBUG: Finished streaming. Full response: '{full_bot_response[:100]}...' ---")
   
    bot_message_to_save = ChatMessage(
        content=full_bot_response,
        role="model",
        session_id=chat_session_id
    )
    db_session.add(bot_message_to_save)
    
    try:
        db_session.commit()
        print(f"--- DEBUG: Successfully saved bot response to DB for session {chat_session_id} ---")
        
        # Track usage statistics
        try:
            # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
            estimated_tokens = len(full_bot_response) // 4
            UsageTracker.track_message(
                user_id=current_user.id,
                tokens_used=estimated_tokens,
                model_name=model_name,
                web_search_used=use_web_search
            )
            print(f"--- DEBUG: Usage tracked for user {current_user.id} ---")
        except Exception as tracking_error:
            print(f"--- DEBUG: Error tracking usage: {tracking_error} ---")
            
    except Exception as e:
        print(f"--- DEBUG: Error saving bot response to DB: {e} ---")
        db_session.rollback()
        raise


@router.post("/", summary="Post a new message and get a streaming response")
def post_new_message(
    *,
    limiter: Limiter = Depends(lambda: limiter.limit(CHAT_RATE_LIMIT)),
    request_data: NewChatMessageRequest,
    db_session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    chat_session = None
    # *** CHANGE: Track if we created a new session ***
    session_was_created = False
    
    if request_data.session_id:
        # *** ENHANCED DEBUG: Log the session ID being requested ***
        print(f"--- DEBUG: Looking for existing session ID: {request_data.session_id} ---")
        chat_session = db_session.get(ChatSession, request_data.session_id)
        if not chat_session or chat_session.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Chat session not found")
        print(f"--- DEBUG: Using existing session {chat_session.id} ---")
    else:
        # *** ENHANCED DEBUG: Log new session creation ***
        print(f"--- DEBUG: No session_id provided, creating new session ---")
        title = request_data.prompt[:100] if len(request_data.prompt) > 100 else request_data.prompt
        chat_session = ChatSession(title=title, user_id=current_user.id)
        db_session.add(chat_session)
        
        try:
            db_session.commit()
            db_session.refresh(chat_session)
            session_was_created = True  # *** CHANGE: Mark that we created a new session ***
            print(f"--- DEBUG: Created NEW chat session with ID: {chat_session.id} ---")
            
            # Track session creation
            try:
                UsageTracker.track_session_created(current_user.id)
                print(f"--- DEBUG: Session creation tracked for user {current_user.id} ---")
            except Exception as tracking_error:
                print(f"--- DEBUG: Error tracking session creation: {tracking_error} ---")
                
        except Exception as e:
            print(f"--- DEBUG: Error creating chat session: {e} ---")
            db_session.rollback()
            raise HTTPException(status_code=500, detail="Failed to create chat session")

    # Save the user message
    user_message = ChatMessage(
        content=request_data.prompt, 
        role="user", 
        session_id=chat_session.id
    )
    db_session.add(user_message)
    
    try:
        db_session.commit()
        print(f"--- DEBUG: User message saved for session {chat_session.id} ---")
    except Exception as e:
        print(f"--- DEBUG: Error saving user message: {e} ---")
        db_session.rollback()
        raise HTTPException(status_code=500, detail="Failed to save user message")

    # Fetch chat history (excluding current message)
    statement = (
        select(ChatMessage)
        .where(ChatMessage.session_id == chat_session.id)
        .order_by(ChatMessage.created_at)
        .limit(20)
    )
    db_messages = db_session.exec(statement).all()
    print(f"--- DEBUG: Fetched {len(db_messages)} messages from DB for session {chat_session.id} ---")

    # Format history for LangChain (excluding the current user message)
    chat_history_for_chain = []
    for msg in db_messages[:-1]:
        if msg.role == "user":
            chat_history_for_chain.append(HumanMessage(content=msg.content))
        elif msg.role == "model":
            chat_history_for_chain.append(AIMessage(content=msg.content))
    
    print(f"--- DEBUG: Formatted {len(chat_history_for_chain)} messages for LangChain ---")
    print(request_data.prompt, 
            chat_session.id, 
            chat_history_for_chain, 
            db_session
        )
    # *** CHANGE: Create StreamingResponse with custom headers that include session ID ***
    response = StreamingResponse(
        stream_and_save_response_with_headers(
            request_data.prompt, 
            chat_session.id, 
            chat_history_for_chain, 
            db_session,
            request_data.model_name,
            request_data.use_web_search
        ),
        media_type="text/plain; charset=utf-8"
    )
    
    # *** CHANGE: Add session ID to response headers so frontend can capture it ***
    response.headers["X-Session-ID"] = str(chat_session.id)
    # *** CHANGE: Add flag to indicate if this was a new session ***
    response.headers["X-Session-Created"] = str(session_was_created).lower()
    
    print(f"--- DEBUG: Returning response with session ID {chat_session.id} in headers ---")
    return response

# --- Other Endpoints (No changes) ---
@router.get("/models", summary="Get available models")
def get_available_models():
    """Get list of available models grouped by provider"""
    models_by_provider = {}
    for model_name, config in MODELS.items():
        provider = config["provider"]
        if provider not in models_by_provider:
            models_by_provider[provider] = []
        models_by_provider[provider].append({
            "name": model_name,
            "display_name": model_name.replace("models/", "").replace("/", " / ")
        })
    return models_by_provider


@router.get("/", response_model=List[ChatSessionResponse], summary="Get all chat sessions for the current user")
def get_user_chat_sessions(
    *, 
    session: Session = Depends(get_session), 
    current_user: User = Depends(get_current_active_user)
):
    return current_user.sessions

@router.get("/{session_id}", response_model=ChatHistoryResponse, summary="Get the history of a specific chat session")
def get_chat_history(
    *, 
    session_id: int, 
    session: Session = Depends(get_session), 
    current_user: User = Depends(get_current_active_user)
):
    chat_session = session.get(ChatSession, session_id)
    if not chat_session or chat_session.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found")
    return chat_session

@router.put("/{session_id}", response_model=ChatSessionResponse, summary="Rename a chat session")
def rename_chat_session(
    session_id: int,
    request_data: RenameChatRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user)
):
    chat_session = session.get(ChatSession, session_id)
    if not chat_session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found")
    if chat_session.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to rename this chat session")
    chat_session.title = request_data.new_title
    session.add(chat_session)
    session.commit()
    session.refresh(chat_session)
    return chat_session


@router.delete("/all", status_code=status.HTTP_200_OK, summary="Delete all chat sessions for current user")
def delete_all_chat_sessions(
    *,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user)
):
    # Get all chat sessions for the current user
    chat_sessions = session.exec(
        select(ChatSession).where(ChatSession.user_id == current_user.id)
    ).all()
    
    if not chat_sessions:
        return {"message": "No chat sessions found to delete"}
    
    # Delete all sessions (messages will be deleted automatically due to cascade)
    for chat_session in chat_sessions:
        session.delete(chat_session)
    
    session.commit()
    
    return {"message": f"Successfully deleted {len(chat_sessions)} chat sessions"}


@router.delete("/{session_id}", status_code=status.HTTP_200_OK, summary="Delete a chat session")
def delete_chat_session(
    *,
    session_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user)
):
    chat_session = session.get(ChatSession, session_id)
    if not chat_session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found")
    if chat_session.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to delete this chat session")
    session.delete(chat_session)
    session.commit()
    return {"message": "Chat session deleted successfully"}

# @router.post("/")
# async def create_chat_session_or_message(
#     user_message: str = Body(..., embed=True),
#     session_id: int = Body(None, embed=True),
#     db: Session = Depends(get_session),
#     current_user: User = Depends(get_current_active_user),
# ):
#     # --- This part for getting or creating a session is the same ---
#     print(f"--- INFO: Creating or using chat session for user {current_user.id} ---")
#     if session_id:
#         chat_session = db.get(ChatSession, session_id)
#         if not chat_session or chat_session.user_id != current_user.id:
#             raise HTTPException(status_code=404, detail="Chat session not found")
#     else:
#         chat_session = ChatSession(session_name="New Chat", user_id=current_user.id)
#         db.add(chat_session)
#         db.commit()
#         db.refresh(chat_session)
#     print(f"--- INFO: Using chat session ID: {chat_session.id} ---")
#     # --- Saving user message is the same ---
#     user_db_message = ChatMessage(message_text=user_message, is_user=True, session_id=chat_session.id)
#     db.add(user_db_message)
#     db.commit()
#     print(f"--- INFO: User message saved for session {chat_session.id} ---")
#     # --- Preparing chat history is the same ---
#     history = []
#     # Make sure to refresh the session to get the latest message list
#     db.refresh(chat_session) 
#     print(f"--- INFO: Preparing chat history for session {chat_session.id} ---")    
#     for msg in chat_session.messages:
#         if msg.is_user:
#             history.append(HumanMessage(content=msg.message_text))
#         else:
#             history.append(AIMessage(content=msg.message_text))
#     print(f"--- INFO: Chat history prepared with {len(history)} messages ---")
#     async def stream_response():
#         # === THE NEW CORE LOGIC ===
#         # Decide which chain to use based on the session's 'has_documents' flag
#         if chat_session.has_documents:
#             # This is a RAG-enabled session
#             print(f"--- INFO: Routing to RAG chain for session_id: {chat_session.id} ---")
#             g = get_rag_chatbot_response(user_message, history, chat_session.id)
#         else:
#             # This is a standard conversational session
#             print(f"--- INFO: Routing to standard chain for session_id: {chat_session.id} ---")
#             g = get_chatbot_response(user_message, history)
#         # ==========================

#         full_response = ""
#         for chunk in g:
#             full_response += chunk
#             yield chunk
#         print(f"--- INFO: Full response generated for session {chat_session.id} ---")
#         # Save the full AI response to the database
#         ai_db_message = ChatMessage(message_text=full_response, is_user=False, session_id=chat_session.id)
#         db.add(ai_db_message)
#         db.commit()

#     return StreamingResponse(stream_response(), media_type="text/event-stream")


# Usage Statistics Endpoints
@router.get("/usage/stats", summary="Get usage statistics for the current user")
async def get_usage_stats(
    days: int = 30,
    current_user: User = Depends(get_current_active_user)
):
    """Get usage statistics for the current user over the specified number of days"""
    try:
        stats = UsageTracker.get_user_usage_stats(current_user.id, days)
        return {"stats": stats}
    except Exception as e:
        print(f"--- ERROR: Failed to get usage stats: {e} ---")
        raise HTTPException(status_code=500, detail="Failed to retrieve usage statistics")


@router.get("/usage/totals", summary="Get total usage statistics for the current user")
async def get_usage_totals(
    current_user: User = Depends(get_current_active_user)
):
    """Get total usage statistics for the current user"""
    try:
        totals = UsageTracker.get_user_total_stats(current_user.id)
        return totals
    except Exception as e:
        print(f"--- ERROR: Failed to get usage totals: {e} ---")
        raise HTTPException(status_code=500, detail="Failed to retrieve usage totals")


# Usage Statistics Endpoints
@router.get("/usage/stats")
async def get_usage_stats(
    days: int = 30,
    current_user: User = Depends(get_current_active_user)
):
    """Get usage statistics for the current user over the last N days"""
    try:
        stats = UsageTracker.get_user_usage_stats(current_user.id, days)
        return {"stats": stats}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch usage statistics: {str(e)}"
        )


@router.get("/usage/totals")
async def get_usage_totals(
    current_user: User = Depends(get_current_active_user)
):
    """Get total usage statistics for the current user"""
    try:
        totals = UsageTracker.get_user_total_stats(current_user.id)
        return totals
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch usage totals: {str(e)}"
        )
