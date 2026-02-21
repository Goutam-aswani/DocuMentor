from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from h11 import Request
from sqlmodel import Session
from database import get_session
from models import Token, User
from security import verify_password, create_access_token 
from dependencies import get_user
from datetime import timedelta
from config import settings
from limiter import limiter
from slowapi import Limiter
from dependencies import get_user

router = APIRouter(
    prefix="/auth",
    tags=["authentication"],
)

def authenticate_user(session: Session, username: str, password: str) -> User | None:
    user = get_user(session, username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

LOGIN_RATE_LIMIT = "5/minute"
@router.post("/token",response_model= Token)
async def login_for_access_token(limiter: Limiter = Depends(lambda: limiter.limit(LOGIN_RATE_LIMIT)), form_data:OAuth2PasswordRequestForm = Depends(),session: Session = Depends(get_session)):
    user = authenticate_user(session,form_data.username,form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,detail="incorrect username or password",headers={"WWW-Authenticate":"Bearer"})
    access_token_expires = timedelta(minutes = settings.access_token_expire_minutes)
    access_token = create_access_token(data={"sub":user.username},expires_delta = access_token_expires)
    return{"access_token":access_token,"token_type":"bearer"}