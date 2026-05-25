import os
import logging
from dotenv import load_dotenv
from backend.embeddings import get_embedding
from backend.retrieval import search_similar_documents
from backend.llm import generate_response
from backend.storage import storage_manager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG")

# Load environment variables
load_dotenv()


def execute_rag_pipeline(session_id: str, message: str) -> dict:
    """
    Executes the complete Retrieval-Augmented Generation (RAG) pipeline:
    1. Generates embedding for the user message.
    2. Performs similarity search against document chunks.
    3. Evaluates if the top similarity matches meet the threshold.
    4. Gathers session conversation history.
    5. Builds the grounded system prompt.
    6. Generates response via the LLM provider.
    7. Stores chat history and returns the payload.
    """
    # Load configuration
    threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.70"))
    provider = os.getenv("LLM_PROVIDER", "local").lower()
    
    logger.info(f"Executing RAG pipeline for session '{session_id}' with provider '{provider}' and threshold {threshold}")

    # Step 1: Context-Aware Query Expansion
    # Gathers previous conversation history. If the query contains referential terms (like 'that option', 'there', 'it'),
    # combines it with the previous user message for embedding generation to ensure semantic retrieval succeeds in offline mode.
    history = storage_manager.get_conversation_history(session_id, limit=4)
    search_query = message
    
    referential_words = {"that", "it", "this", "option", "setting", "there", "where", "located", "go", "find", "button", "link", "page", "tab"}
    query_words = set(message.lower().replace("?", " ").replace(".", " ").split())
    
    if history and query_words.intersection(referential_words):
        last_user_msgs = [h["content"] for h in history if h["role"] == "user"]
        if last_user_msgs:
            search_query = f"{last_user_msgs[-1]} {message}"
            logger.info(f"Conversational reference detected. Expanded query for retrieval: '{search_query}'")

    logger.info(f"Generating embedding for question: '{search_query}'")
    query_vector = get_embedding(search_query)

    # Step 2: Similarity search against document chunks
    document_chunks = storage_manager.get_all_chunks()
    top_matches = search_similar_documents(query_vector, document_chunks, threshold=threshold, top_k=3)

    # Step 3: Threshold check
    # If no document chunks pass the threshold, reject with 'Insufficient information' immediately (Step 8 & Step 15)
    if not top_matches:
        logger.warning(f"No document chunks passed similarity threshold {threshold}.")
        reply = "I do not have enough information to answer that."
        
        # Save this exchange to conversation history anyway to preserve multi-turn memory
        storage_manager.add_message_to_history(session_id, "user", message)
        storage_manager.add_message_to_history(session_id, "assistant", reply)
        
        return {
            "reply": reply,
            "tokensUsed": len(message.split()) + len(reply.split()) + 10, # Estimate tokens
            "retrievedChunks": 0,
            "chunks": []
        }

    # Step 4: Build retrieved context from top matches
    retrieved_context_blocks = []
    chunk_details = []
    for idx, match in enumerate(top_matches):
        chunk = match["chunk"]
        score = match["score"]
        
        retrieved_context_blocks.append(
            f"[Document: {chunk['title']} (Category: {chunk['category']})]\n{chunk['content']}"
        )
        
        chunk_details.append({
            "title": chunk["title"],
            "content": chunk["content"],
            "category": chunk["category"],
            "score": round(score, 3)
        })
        
    retrieved_context = "\n\n".join(retrieved_context_blocks)

    # Step 5: Gather and format conversation history
    history = storage_manager.get_conversation_history(session_id, limit=4)
    history_formatted_list = []
    for h in history:
        role_label = "User" if h["role"] == "user" else "Assistant"
        history_formatted_list.append(f"{role_label}: {h['content']}")
        
    history_formatted = "\n".join(history_formatted_list) if history_formatted_list else "No previous history."

    # Step 6: Construct the RAG Prompt (Step 9)
    # Uses clean XML tags/structural spacing for modern high-performance LLM compliance
    prompt = f"""You are a highly helpful and factual customer support AI assistant.
Your goal is to answer the user's question using ONLY the provided Context.

STRICT INSTRUCTIONS:
- You must rely ONLY on the provided Context to answer the question.
- Do not make up facts or use external training knowledge if it is not grounded in the Context.
- If the Context does not provide the answer, say exactly: "I do not have enough information to answer that."
- Keep your answer professional, concise, and focused on solving the user's request.

Context:
{retrieved_context}

History:
{history_formatted}

Question:
{message}

Answer:"""

    # Step 7: Call LLM API (Step 10)
    logger.info("Sending RAG prompt to LLM...")
    try:
        reply = generate_response(prompt, provider=provider)
    except Exception as e:
        logger.error(f"Error during LLM response generation: {e}")
        # If API failed, return the error message directly
        raise e

    # Step 8: Save current exchange to session history (Step 12)
    storage_manager.add_message_to_history(session_id, "user", message)
    storage_manager.add_message_to_history(session_id, "assistant", reply)

    # Estimate token usage (Prompt words / 0.75 + reply words / 0.75)
    prompt_tokens = int(len(prompt.split()) / 0.75)
    reply_tokens = int(len(reply.split()) / 0.75)
    total_tokens = prompt_tokens + reply_tokens

    logger.info(f"RAG execution successful. Tokens used: {total_tokens}. Chunks retrieved: {len(top_matches)}.")
    
    return {
        "reply": reply.strip(),
        "tokensUsed": total_tokens,
        "retrievedChunks": len(top_matches),
        "chunks": chunk_details
    }
