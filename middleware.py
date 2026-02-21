from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

def setup_cors(app: FastAPI):
    origins = [
        "http://localhost",
        "http://localhost:3000",
        "http://127.0.0.1:5500",
        "http://localhost:8080",
        "http://localhost:5173",
        "http://localhost:5173/chats",
    ]

    app.add_middleware( 
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Session-ID", "X-Session-Created"]
    )
