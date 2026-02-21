# rag_routes.py
import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlmodel import Session
from database import get_session
from models import ChatSession, User
from dependencies import get_current_user
from rag_service import process_and_store_document

router = APIRouter(
    prefix="/rag",
    tags=["rag"],
)
UPLOAD_DIRECTORY = "./temp_uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

@router.post("/upload/{session_id}")
async def upload_document_for_session(
    session_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    Uploads a document, processes it, and associates it with a specific chat session.
    """
    # 1. Verify the chat session exists and belongs to the user
    chat_session = db.get(ChatSession, session_id)
    print(f"--- DEBUG: Chat session found: {chat_session} ---")
    if not chat_session or chat_session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Chat session not found")

    file_path = ""
    try:
        # 2. Save the uploaded file to a temporary location
        file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 3. Process the document and store embeddings with session_id metadata
        print(f"--- DEBUG: Processing file {file.filename} for session {session_id} ---")
        process_and_store_document(file_path, session_id)
        
        # 4. Mark the session as having documents
        if not chat_session.has_documents:
            chat_session.has_documents = True
            db.add(chat_session)
            db.commit()
            db.refresh(chat_session)
        print(f"--- DEBUG: Updated chat session {session_id} to have documents ---")
        return {"message": f"Successfully processed {file.filename} for session {session_id}"}
        
    except Exception as e:
        import traceback
        print(f"--- ERROR: Upload failed for session {session_id}: {e} ---")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        # 5. Clean up by deleting the temporary file
        print(f"--- DEBUG: Cleaning up temporary file {file_path} ---")
        if os.path.exists(file_path):
            os.remove(file_path)