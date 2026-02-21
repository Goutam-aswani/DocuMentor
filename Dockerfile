# Use slim Python 3.11 for a smaller image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed by psycopg2-binary and PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source (excludes .env, venv, __pycache__ via .dockerignore)
COPY . .

# Expose the port uvicorn will run on
EXPOSE 8000

# Run the server — main.py's create_db_and_tables() handles schema creation on startup
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
