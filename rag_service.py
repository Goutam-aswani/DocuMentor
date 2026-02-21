# rag_service.py
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from config import settings


# --- Helpers ---

def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Returns the embedding model (gemini-embedding-001, 3072 dims)."""
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=settings.google_api_key
    )

def get_qdrant_client() -> QdrantClient:
    """Returns a connected Qdrant client. Fails clearly if env vars are not set."""
    if not settings.qdrant_url or not settings.qdrant_api_key:
        raise RuntimeError(
            "QDRANT_URL and QDRANT_API_KEY must be set in your environment/.env file."
        )
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

COLLECTION_NAME = "chatbot_rag"


# --- 1. Document Loading and Splitting ---

def load_document(file_path: str) -> list[Document]:
    """Loads a document based on its file extension."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type.")
    print(f"--- INFO: Loading document from {file_path} ---")
    return loader.load()

def split_text(documents: list[Document]) -> list[Document]:
    """Splits documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    print(f"--- INFO: Splitting {len(documents)} documents into chunks ---")
    return text_splitter.split_documents(documents)


# --- 2. Vector Store and Retriever ---

def process_and_store_document(file_path: str, session_id: int):
    """
    The complete pipeline for processing and storing a document with session metadata.
    Signature preserved — caller in rag.py calls process_and_store_document(file_path, session_id).
    """
    documents = load_document(file_path)
    chunks = split_text(documents)

    # Add session_id metadata to each chunk so they can be filtered per session
    for chunk in chunks:
        chunk.metadata = {"session_id": str(session_id)}

    print(f"--- INFO: Generating embeddings for {len(chunks)} chunks using gemini-embedding-001 ---")
    embeddings = get_embeddings()

    try:
        # from_documents creates the collection if it doesn't exist, or adds to it if it does.
        # force_recreate=False (default) ensures existing data in other sessions is preserved.
        QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection_name=COLLECTION_NAME,
            force_recreate=False,
        )
        print(f"--- INFO: Stored {len(chunks)} chunks for session_id: {session_id} ---")

        # Ensure payload index exists for filtered retrieval.
        # Qdrant Cloud requires this index for metadata-based filtering.
        try:
            from qdrant_client.models import PayloadSchemaType
            client = get_qdrant_client()
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="metadata.session_id",
                field_schema=PayloadSchemaType.KEYWORD
            )
            print("--- INFO: Payload index on metadata.session_id ensured ---")
        except Exception:
            # Index already exists — safe to ignore
            pass

    except Exception as embed_error:
        print(f"--- ERROR: Failed to add documents to Qdrant: {embed_error} ---")
        raise


def get_session_retriever(session_id: int):
    """
    Creates a retriever that ONLY searches for documents matching the session_id.
    Signature preserved — callers use this directly via as_retriever pattern.
    """
    embeddings = get_embeddings()
    client = get_qdrant_client()

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    # Qdrant stores LangChain metadata under the "metadata" payload key
    qdrant_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.session_id",
                match=MatchValue(value=str(session_id))
            )
        ]
    )

    print(f"--- INFO: Creating retriever for session_id: {session_id} ---")
    return vector_store.as_retriever(
        search_kwargs={"filter": qdrant_filter}
    )


def get_session_retriever_with_scores(session_id: int, similarity_threshold: float = 0.9):
    """
    Creates a retriever with similarity score filtering, scoped to session_id.

    NOTE on scores: Qdrant returns similarity scores in range [0, 1] where HIGHER = more similar.
    The old ChromaDB returned distance where LOWER = more similar.
    To maintain equivalent loose filtering behaviour (threshold=0.95 was very loose with ChromaDB),
    we accept documents where score >= (1 - similarity_threshold), i.e., >= 0.05 for threshold=0.95.

    Signature + FilteredRetriever interface preserved — chatbot_service.py calls:
      retriever.get_relevant_documents(query)
    """
    embeddings = get_embeddings()
    client = get_qdrant_client()

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    qdrant_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.session_id",
                match=MatchValue(value=str(session_id))
            )
        ]
    )

    print(f"--- INFO: Creating retriever with similarity threshold {similarity_threshold} for session_id: {session_id} ---")

    # Equivalent minimum similarity score (Qdrant: higher=better, ChromaDB: lower=better)
    min_similarity = 1.0 - similarity_threshold  # e.g. threshold=0.95 → min_similarity=0.05

    def retrieve_with_filtering(query: str) -> list:
        docs_with_scores = vector_store.similarity_search_with_score(
            query,
            k=10,
            filter=qdrant_filter
        )

        print(f"--- DEBUG: Retrieved {len(docs_with_scores)} documents before filtering ---")

        filtered_docs = []
        for doc, score in docs_with_scores:
            # Qdrant score: 1.0 = identical, 0.0 = completely different
            print(f"--- DEBUG: Document score: {score:.4f} (min required: {min_similarity:.4f}) ---")
            if score >= min_similarity:
                filtered_docs.append(doc)
            else:
                print(f"--- DEBUG: Document filtered out due to low similarity ---")

        if not filtered_docs and docs_with_scores:
            print("--- INFO: No documents passed threshold, returning top-scoring document as fallback ---")
            filtered_docs.append(docs_with_scores[0][0])

        print(f"--- INFO: {len(filtered_docs)} documents passed similarity threshold ---")
        return filtered_docs

    class FilteredRetriever:
        def __init__(self, retrieve_func):
            self.retrieve_func = retrieve_func

        def get_relevant_documents(self, query: str):
            return self.retrieve_func(query)

    return FilteredRetriever(retrieve_with_filtering)