from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks,status
from sqlmodel import Session , select
from database import get_session
from models import User, UserCreate, UserRead,Forgot_password_request,Reset_password_request,VerifyEmailRequest,UserUpdate
from security import get_password_hash, create_access_token, create_password_reset_token, verify_password, verify_password_reset_token
from dependencies import get_current_active_user, get_user , get_current_user
from datetime import datetime, timedelta, UTC
from email_service import send_email
import secrets
router = APIRouter(
    prefix="/users",
    tags=["users"],
)

def write_log(message:str):
    with open("log.txt",mode = "a") as log_file:
        log_file.write(message)
 
def get_user_by_email(session: Session, email: str) -> User | None:
    statement = select(User).where(User.email == email)
    return session.exec(statement).first()

def get_user_by_username(session: Session, username: str) -> User | None:
    statement = select(User).where(User.username == username)
    return session.exec(statement).first()

@router.post("/register",response_model=UserRead)
async def register_new_user(user_create : UserCreate,background_tasks: BackgroundTasks,session : Session = Depends(get_session)):
    db_user = get_user(session, user_create.username)
    if db_user:
        raise HTTPException(status_code=400, detail="User already exists")

    db_user_email = get_user_by_email(session, user_create.email)
    if db_user_email:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    otp = "".join([str(secrets.randbelow(10)) for _ in range(6)])
    otp_expires_at = datetime.now(UTC) + timedelta(minutes=15)

    hashed_password = get_password_hash(user_create.password)
    user_data = user_create.model_dump(exclude={"password"})
    
    db_user = User(
        **user_data, 
        hashed_password=hashed_password,
        is_verified=True,
        verification_code=otp,
        verification_code_expires_at=otp_expires_at
    )
    background_tasks.add_task(
        write_log,f"User registered:{db_user.username}, Email : {db_user.email} \n"
    )
    session.add(db_user)
    session.commit()
    session.refresh(db_user)

    # Try to send verification email, but don't crash if SMTP is blocked (e.g. Render free tier)
    try:
        email_subject = "Welcome to DocuMentor"
        email_body = f"""
        <p>Hello {db_user.username},</p>
        <p>Your account has been created successfully. You can now log in and start using DocuMentor.</p>
        """
        background_tasks.add_task(
            send_email, [db_user.email], email_subject, email_body
        )
    except Exception:
        pass

    return db_user

@router.post("/verify-email")
def verify_email(
    request : VerifyEmailRequest,
    session : Session = Depends(get_session),

):
    user = get_user_by_email(session, request.email)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    if user.is_verified:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email is already verified.")
    if user.verification_code != request.otp:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid OTP.")
    if user.verification_code_expires_at and user.verification_code_expires_at.replace(tzinfo=UTC) < datetime.now(UTC):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OTP has expired.")
    
    user.is_verified = True
    user.verification_code = None
    user.verification_code_expires_at = None

    session.add(user)
    session.commit()

    return {"message": "Email verified successfully. You can now log in."}

@router.post("/forget_password")
def forgot_password(
    request: Forgot_password_request,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session)
):
    user = get_user_by_email(session,request.email)
    if user:
        password_reset_token  = create_password_reset_token(email= user.email)

        background_tasks.add_task(
        write_log, f"Password reset token for {user.email}: {password_reset_token}\n"
        )
        email_subject = "Password Reset for Your Notes App"
        email_body = f"""
    <p>Hello {user.username},</p>
    <p>You requested a password reset. Here is your token:</p>
    <p><b>{password_reset_token}</b></p>
    <p>This token will expire in 30 minutes.</p>
    <p>If you did not request a password reset, please ignore this email.</p>
    """
    
        background_tasks.add_task(
        send_email, [user.email], email_subject, email_body
    )    
    return {"message": "If an account with this email exists, a password reset link has been sent."}


@router.post("/reset_password")
def reset_password(
    request:Reset_password_request,
    session: Session = Depends(get_session)
):
    email = verify_password_reset_token(token=request.token)
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired token.",
        )
    user = get_user_by_email(session,email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )
    new_hashed_password = get_password_hash(request.new_password)
    user.hashed_password = new_hashed_password

    session.add(user)
    session.commit()

    return {"message": "Password updated successfully."}

@router.get("/me",response_model = UserRead)
async def read_user_me(current_user:User = Depends(get_current_active_user)):
    return current_user

@router.put("/me", response_model=UserRead)
async def update_user_me(
        *,
        user_update: UserUpdate,
        session : Session = Depends(get_session),
        current_user: User = Depends(get_current_active_user)
):
    if user_update.email:
        existing_user = get_user_by_email(session,user_update.email)
        if existing_user and existing_user.id != current_user.id:
            raise HTTPException(status_code=400, detail="Email already registered")
        current_user.email = user_update.email

    if user_update.full_name is not None:
        current_user.full_name = user_update.full_name

    session.add(current_user)
    session.commit()
    session.refresh(current_user)
    return current_user

