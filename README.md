# 🤖 AI Chatbot with RAG & Authentication

A full-stack AI chatbot with document-based Q&A (RAG), multi-model support, and user authentication.

## Features

- **Multi-Model Chat** — Switch between Groq (Llama, Qwen, Kimi) and OpenRouter models
- **RAG (Retrieval-Augmented Generation)** — Upload PDFs, DOCX, or TXT files and ask questions grounded in your documents
- **Web Search** — Optional real-time web search via Tavily API
- **Authentication** — JWT-based auth with email verification, password reset
- **Usage Tracking** — Per-user daily stats (messages, tokens, model usage)
- **Streaming Responses** — Real-time token-by-token response streaming

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | FastAPI, LangChain, SQLModel |
| **Frontend** | React (Vite), Tailwind CSS |
| **Database** | PostgreSQL (Supabase) |
| **Vector DB** | Qdrant Cloud |
| **LLMs** | Groq, OpenRouter |
| **Embeddings** | Google Generative AI (gemini-embedding-001) |
| **Search** | Tavily API |

## Project Structure

```
├── main.py                  # FastAPI entry point
├── config.py                # Environment settings
├── database.py              # DB engine + session
├── models.py                # SQLModel schemas
├── auth.py                  # Login / token endpoint
├── users.py                 # Registration, verification, profile
├── chats.py                 # Chat sessions, messaging, streaming
├── rag.py                   # Document upload routes
├── chatbot_service.py       # LLM chains (standard + RAG)
├── rag_service.py           # Qdrant vector store + retrieval
├── web_search_service.py    # Tavily search integration
├── usage_tracker.py         # Usage statistics
├── dependencies.py          # Auth middleware
├── security.py              # JWT + bcrypt
├── email_service.py         # Email via SMTP
├── middleware.py             # CORS config
├── limiter.py               # Rate limiting
├── Dockerfile               # Container config
├── render.yaml              # Render.com deploy blueprint
├── requirements.txt         # Python dependencies
├── .env.example             # Environment template
└── chatbot-frontend/        # React frontend
    ├── src/
    ├── vercel.json           # Vercel SPA config
    └── .env.example
```

## Setup

### Backend

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Copy and fill in environment variables
cp .env.example .env

# Run the server
uvicorn main:app --reload
```

### Frontend

```bash
cd chatbot-frontend
npm install
cp .env.example .env  # Set VITE_API_BASE_URL
npm run dev
```

## Environment Variables

See `.env.example` for all required variables. Key ones:

- `DATABASE_URL` — Supabase PostgreSQL connection string
- `QDRANT_URL` / `QDRANT_API_KEY` — Qdrant Cloud credentials
- `GOOGLE_API_KEY` — For embeddings
- `GROK_API_KEY` — Groq API key
- `OPENROUTER_API_KEY` — OpenRouter API key

## Deployment

- **Backend** → [Render.com](https://render.com) (Dockerfile)
- **Frontend** → [Vercel](https://vercel.com) (auto-detected Vite)
- **Database** → [Supabase](https://supabase.com) (free PostgreSQL)
- **Vector DB** → [Qdrant Cloud](https://cloud.qdrant.io) (free tier)
