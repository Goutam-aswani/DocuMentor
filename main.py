from fastapi import FastAPI
from database import create_db_and_tables
import auth
import users
import chats
import rag
from middleware import setup_cors
from limiter import limiter
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded


app = FastAPI(
    title="Basic Chatbot",
    description="This is a simple chatbot",
    version="1.0.0",

)   
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  
setup_cors(app)

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

app.include_router(auth.router)
app.include_router(users.router)
app.include_router(chats.router)
app.include_router(rag.router)


@app.get("/")
def read_root():
    return {"Hello": "World"}
