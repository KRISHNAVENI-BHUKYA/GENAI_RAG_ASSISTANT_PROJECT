import os
import logging
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import RAG pipeline & components
from backend.rag import execute_rag_pipeline
from backend.storage import storage_manager
from backend.llm import InvalidApiKeyException, RateLimitException, RequestTimeoutException

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MainAPI")

# Initialize FastAPI application
app = FastAPI(
    title="GenAI RAG Chat Assistant API",
    description="Backend API supporting similarity search, chunking, and grounded LLM answers.",
    version="1.0.0"
)

# Enable CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Absolute directory paths for robustness
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")


# Pydantic Schemas for Request/Response validation
class ChatRequest(BaseModel):
    sessionId: str = Field(..., description="Unique alphanumeric session ID for tracking conversation history.")
    message: str = Field(..., min_length=1, description="Non-empty user prompt or question.")


class ChatResponse(BaseModel):
    reply: str
    tokensUsed: int
    retrievedChunks: int


# API Exception Handlers (Graceful failure handling - Step 11)
@app.exception_handler(InvalidApiKeyException)
async def api_key_exception_handler(request: Request, exc: InvalidApiKeyException):
    logger.error(f"Invalid API key error caught: {exc}")
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={"error": "Invalid API key"}
    )


@app.exception_handler(RateLimitException)
async def rate_limit_exception_handler(request: Request, exc: RateLimitException):
    logger.error(f"Rate limit exceeded error caught: {exc}")
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"error": "Rate limit exceeded"}
    )


@app.exception_handler(RequestTimeoutException)
async def timeout_exception_handler(request: Request, exc: RequestTimeoutException):
    logger.error(f"Request timeout error caught: {exc}")
    return JSONResponse(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        content={"error": "Request timeout"}
    )


# API Endpoints
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Main endpoint for chatbot interaction.
    Validates requests, runs the RAG pipeline, and returns the grounded answer.
    """
    session_id = request.sessionId.strip()
    message = request.message.strip()

    # Extra manual validation for empty prompts after stripping whitespace
    if not message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot consist only of whitespace."
        )
        
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="SessionId is required and cannot be empty."
        )

    try:
        logger.info(f"Received chat request on session '{session_id}'")
        result = execute_rag_pipeline(session_id, message)
        return result
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}")
        # Re-raise standard RAG exceptions so they can trigger their custom handlers
        if isinstance(e, (InvalidApiKeyException, RateLimitException, RequestTimeoutException)):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred on our servers. Please try again."
        )


@app.get("/api/docs")
async def get_docs_endpoint():
    """
    Utility endpoint to return the loaded knowledge base documents.
    Used by the frontend to display the available reference materials.
    """
    try:
        chunks = storage_manager.get_all_chunks()
        # Clean chunks of the large embedding array before returning to the UI to keep it lightweight
        clean_chunks = []
        for c in chunks:
            clean_chunks.append({
                "id": c["id"],
                "title": c["title"],
                "category": c["category"],
                "content": c["content"]
            })
        return clean_chunks
    except Exception as e:
        logger.error(f"Failed to get knowledge base docs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve knowledge base documents."
        )


# Setup document knowledge base on application startup
@app.on_event("startup")
def startup_event():
    logger.info("Initializing system and knowledge base...")
    storage_manager.initialize_knowledge_base()
    logger.info("System successfully initialized.")


# Serving the static frontend SPA (Single Page Application)
# Route the root URL to frontend/index.html
@app.get("/")
async def serve_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index_path):
        return {"message": "RAG Chatbot Backend is running, but frontend/index.html is not created yet!"}
    return FileResponse(index_path)


# Mount the remaining static folder for style.css and script.js
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")
else:
    logger.warning(f"Frontend folder not found at {FRONTEND_DIR}. Static files serving is disabled.")
