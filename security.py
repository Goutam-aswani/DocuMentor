from datetime import datetime, timedelta, UTC
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# def verify_password(plain_password,hashed_password):
#     return pwd_context.verify(plain_password,hashed_password)
def verify_password(plain_password, hashed_password):
    # Truncate password to 72 bytes (bcrypt limit) for consistency
    truncated_password = plain_password[:72]
    return pwd_context.verify(truncated_password, hashed_password)

def get_password_hash(password):
    # Truncate password to 72 bytes (bcrypt limit) for security
    truncated_password = password[:72]
    return pwd_context.hash(truncated_password)
# def get_password_hash(password):
#     return pwd_context.hash(password)


def create_access_token(data: dict,expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) +  timedelta(minutes=15)
    
    to_encode.update({"exp" : expire})
    encoded_jwt = jwt.encode(to_encode,settings.secret_key,algorithm=settings.algorithm)
    return encoded_jwt

def create_password_reset_token(email:str) ->str:

    expire = datetime.now(UTC) + timedelta(minutes=30)
    to_encode = {"exp": expire , "sub": email}

    encoded_jwt = jwt.encode(to_encode,settings.secret_key,algorithm=settings.algorithm)
    return encoded_jwt

def verify_password_reset_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token,settings.secret_key,algorithms=settings.algorithm)

        email: Optional[str] = payload.get("sub")
        if email is None:
            return None
        return email
    except JWTError:
        return None
