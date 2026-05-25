import os
import json
import logging
from backend.embeddings import get_embedding
from backend.retrieval import chunk_text

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Storage")

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_PATH = os.path.join(BACKEND_DIR, "docs.json")
EMBEDDED_DOCS_PATH = os.path.join(BACKEND_DIR, "docs_embedded.json")


class StorageManager:
    def __init__(self):
        self.document_chunks = []
        self.conversation_histories = {} # sessionId -> list of {"role": "user"|"assistant", "content": "..."}
        
    def initialize_knowledge_base(self, force_rebuild: bool = False):
        """
        Loads document knowledge base. Chunks each document and generates embeddings.
        Saves embedded chunks to disk in docs_embedded.json to save time and API quota on subsequent startups.
        """
        # If already loaded and not forcing rebuilding
        if self.document_chunks and not force_rebuild:
            return
            
        # Check if already embedded file exists
        if os.path.exists(EMBEDDED_DOCS_PATH) and not force_rebuild:
            try:
                logger.info(f"Loading pre-embedded document chunks from {EMBEDDED_DOCS_PATH}")
                with open(EMBEDDED_DOCS_PATH, "r", encoding="utf-8") as f:
                    self.document_chunks = json.load(f)
                logger.info(f"Successfully loaded {len(self.document_chunks)} embedded chunks.")
                return
            except Exception as e:
                logger.error(f"Failed to load embedded documents from cache: {e}. Rebuilding...")

        # Rebuilding knowledge base embeddings
        logger.info(f"Rebuilding knowledge base embeddings from {DOCS_PATH}...")
        if not os.path.exists(DOCS_PATH):
            logger.error(f"Knowledge base source file not found at {DOCS_PATH}!")
            # Create a basic sample docs file if missing
            self.create_sample_docs()
            
        try:
            with open(DOCS_PATH, "r", encoding="utf-8") as f:
                documents = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read docs.json: {e}")
            documents = []
            
        chunks_to_embed = []
        for doc in documents:
            doc_id = doc.get("id", "unknown")
            title = doc.get("title", "No Title")
            content = doc.get("content", "")
            category = doc.get("category", "General")
            
            # Divide document content into chunks
            chunks = chunk_text(content, max_words=100, overlap_words=20)
            for idx, text_chunk in enumerate(chunks):
                chunks_to_embed.append({
                    "id": f"doc_{doc_id}_chunk_{idx}",
                    "doc_id": doc_id,
                    "title": title,
                    "category": category,
                    "content": text_chunk
                })
                
        # Generate embeddings for each chunk
        logger.info(f"Generating embeddings for {len(chunks_to_embed)} chunks. Please wait...")
        embedded_chunks = []
        for idx, chunk in enumerate(chunks_to_embed):
            logger.info(f"Embedding chunk {idx + 1}/{len(chunks_to_embed)}: '{chunk['title']}'")
            # Generate the embedding vector
            vector = get_embedding(chunk["content"])
            chunk["embedding"] = vector
            embedded_chunks.append(chunk)
            
        self.document_chunks = embedded_chunks
        
        # Save to cache
        try:
            with open(EMBEDDED_DOCS_PATH, "w", encoding="utf-8") as f:
                json.dump(embedded_chunks, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(embedded_chunks)} pre-embedded chunks to disk at {EMBEDDED_DOCS_PATH}.")
        except Exception as e:
            logger.error(f"Failed to save embedded chunks cache: {e}")

    def create_sample_docs(self):
        """Creates a fallback docs.json if it was deleted or missing."""
        sample_docs = [
            {
                "id": "1",
                "title": "Reset Password",
                "category": "Security",
                "content": "Users can reset their password from Settings > Security."
            },
            {
                "id": "2",
                "title": "Account Deletion",
                "category": "Account Management",
                "content": "Users can delete their account from Account Settings."
            }
        ]
        try:
            with open(DOCS_PATH, "w", encoding="utf-8") as f:
                json.dump(sample_docs, f, indent=2)
            logger.info(f"Created fallback docs.json at {DOCS_PATH}")
        except Exception as e:
            logger.error(f"Could not create fallback docs.json: {e}")

    def get_all_chunks(self) -> list:
        """Returns all embedded document chunks."""
        if not self.document_chunks:
            self.initialize_knowledge_base()
        return self.document_chunks

    def get_conversation_history(self, session_id: str, limit: int = 5) -> list:
        """
        Retrieves the conversation history for a given session.
        Limits the return value to the last 'limit' message pairs (User + Assistant).
        """
        if session_id not in self.conversation_histories:
            self.conversation_histories[session_id] = []
            
        history = self.conversation_histories[session_id]
        
        # Limit to the last N message pairs (2 * limit individual messages)
        # e.g., if limit = 5, we keep the last 10 messages
        max_messages = limit * 2
        if len(history) > max_messages:
            history = history[-max_messages:]
            
        return history

    def add_message_to_history(self, session_id: str, role: str, content: str):
        """
        Appends a message to the conversation history of a given session.
        """
        if session_id not in self.conversation_histories:
            self.conversation_histories[session_id] = []
            
        self.conversation_histories[session_id].append({
            "role": role,
            "content": content
        })
        
        # Proactively clean and keep last 10 messages (5 pairs)
        max_messages = 10
        if len(self.conversation_histories[session_id]) > max_messages:
            self.conversation_histories[session_id] = self.conversation_histories[session_id][-max_messages:]


# Global instance of storage manager
storage_manager = StorageManager()
