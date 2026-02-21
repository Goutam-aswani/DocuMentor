from typing import Optional,List 
from sqlmodel import Field,SQLModel,Relationship
from enum import Enum
from sqlalchemy import DateTime
from datetime import datetime,timedelta,UTC
from pydantic import EmailStr

class Role(str,Enum):
    USER = "user"
    ADMIN = "admin"


class ChatSession(SQLModel, table=True):
    __tablename__ = "chat_sessions"

    id: Optional[int] = Field(default=None, primary_key=True)
    title: str = Field(index=True, description="The title of the chat session, usually the first prompt.")
    
    # Foreign key to link this session to a user
    user_id: int = Field(foreign_key="users.id")
    
    # Establishes the one-to-many relationship from User -> ChatSession
    user: "User" = Relationship(back_populates="sessions")
    
    has_documents: bool = Field(default=False)
    
    # Establishes the one-to-many relationship to ChatMessage
    # messages: List["ChatMessage"] = Relationship(back_populates="session")
    messages: List["ChatMessage"] = Relationship(
        back_populates="session", 
        sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ChatMessage(SQLModel, table=True):
    __tablename__ = "chat_messages"

    id: Optional[int] = Field(default=None, primary_key=True)
    content: str
    
    # 'user' for the user's prompt, 'model' for the AI's response
    role: str  
    
    # Foreign key to link this message to a session
    session_id: int = Field(foreign_key="chat_sessions.id")
    
    # Establishes the relationship back to the ChatSession
    session: ChatSession = Relationship(back_populates="messages")
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

class User(SQLModel,table = True):
    __tablename__ = "users"

    id: Optional[int] = Field(default=None,primary_key=True)
    username: str = Field(unique= True,index= True)
    email: EmailStr = Field(unique= True)
    full_name: Optional[str] = Field(default=None)
    hashed_password: str 
    disabled: bool = False
    role: Role = Field(default=Role.USER, nullable=False)
    is_verified: bool = Field(default=False, nullable=False)
    verification_code: Optional[str] = Field(default=None, nullable=True)
    verification_code_expires_at: Optional[datetime] = Field(default=None, nullable=True)

    created_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(UTC), nullable=True
    )
    updated_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(UTC), nullable=True,
        sa_column_kwargs={"onupdate": lambda: datetime.now(UTC)}
    )
    sessions: List[ChatSession] = Relationship(back_populates="user")



class UserCreate(SQLModel):
    username: str
    password: str
    email: EmailStr
    full_name: Optional[str] = None

class UserRead(SQLModel):
    id: int
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    disabled: bool
    role: Role
    is_verified: bool
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class UserUpdate(SQLModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None

class Forgot_password_request(SQLModel):
    email:EmailStr

class Reset_password_request(SQLModel):
    token : str
    new_password: str

class VerifyEmailRequest(SQLModel):
    email: EmailStr
    otp: str
    
class Token(SQLModel):
    access_token: str
    token_type: str

class Token_data(SQLModel):
    username: Optional[str] = None


class UsageStats(SQLModel, table=True):
    __tablename__ = "usage_stats"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id")
    
    # Daily aggregates
    date: datetime = Field(index=True)
    messages_sent: int = Field(default=0)
    tokens_used: int = Field(default=0)
    sessions_created: int = Field(default=0)
    web_searches_made: int = Field(default=0)
    
    # Model usage tracking
    model_usage: Optional[str] = Field(default="{}")  # JSON string storing model usage counts
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(UTC), nullable=True,
        sa_column_kwargs={"onupdate": lambda: datetime.now(UTC)}
    )
    
    # Relationship to user
    user: "User" = Relationship()


class UsageStatsRead(SQLModel):
    date: datetime
    messages_sent: int
    tokens_used: int
    sessions_created: int
    web_searches_made: int
    model_usage: dict




    