import os
import sys
import json
import logging

# Ensure parent directory is in python path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.rag import execute_rag_pipeline
from backend.storage import storage_manager

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger("RAGTest")


def run_tests():
    logger.info("=== STARTING RAG PIPELINE VERIFICATION TESTS ===")
    
    # 1. Initialize Document storage and cache
    logger.info("Initializing Storage and building/caching vectors...")
    storage_manager.initialize_knowledge_base()
    
    # Check if pre-embeddings loaded
    chunks = storage_manager.get_all_chunks()
    logger.info(f"Loaded {len(chunks)} chunks in knowledge base.")
    assert len(chunks) > 0, "No document chunks loaded in storage!"
    
    test_session = "test_verification_session_999"

    # --- Test Case 1: Valid Question ---
    logger.info("\n--- TEST CASE 1: Valid Question ---")
    query_1 = "How do I reset my password?"
    logger.info(f"Querying: '{query_1}'")
    
    result_1 = execute_rag_pipeline(test_session, query_1)
    
    logger.info(f"Answer received: {result_1['reply']}")
    logger.info(f"Tokens consumed: {result_1['tokensUsed']}")
    logger.info(f"Chunks retrieved: {result_1['retrievedChunks']}")
    
    # Assertions
    assert result_1["retrievedChunks"] > 0, "Failed to retrieve relevant chunks for a valid question!"
    assert "Settings" in result_1["reply"] or "Security" in result_1["reply"] or "Forgot Password" in result_1["reply"], "Grounded answer is missing password reset content!"
    logger.info("TEST CASE 1: [PASSED]")

    # --- Test Case 2: Unknown Question (Triggering Threshold Fallback) ---
    logger.info("\n--- TEST CASE 2: Unknown Question (Out-of-bounds) ---")
    query_2 = "What is your policy on international shipping and tracking?"
    logger.info(f"Querying: '{query_2}'")
    
    result_2 = execute_rag_pipeline(test_session, query_2)
    
    logger.info(f"Answer received: {result_2['reply']}")
    logger.info(f"Chunks retrieved: {result_2['retrievedChunks']}")
    
    # Assertions
    assert result_2["retrievedChunks"] == 0, "Retrieved irrelevant chunks for an out-of-bounds question!"
    assert result_2["reply"] == "I do not have enough information to answer that.", "Failed to trigger correct threshold fallback message!"
    logger.info("TEST CASE 2: [PASSED]")

    # --- Test Case 3: Conversation History Validation ---
    logger.info("\n--- TEST CASE 3: Multi-turn Conversation History ---")
    history = storage_manager.get_conversation_history(test_session)
    logger.info(f"Total history entries in session '{test_session}': {len(history)} messages")
    
    # We sent query_1 (User), got response (Assistant), then sent query_2 (User), got response (Assistant)
    # Total messages in history should be 4 (2 pairs)
    assert len(history) == 4, f"Expected 4 messages in session history, got {len(history)}!"
    assert history[0]["role"] == "user" and history[0]["content"] == query_1, "First user question not preserved correctly in history!"
    assert history[1]["role"] == "assistant" and history[1]["content"] == result_1["reply"], "First assistant answer not preserved in history!"
    assert history[2]["role"] == "user" and history[2]["content"] == query_2, "Second user question not preserved in history!"
    assert history[3]["role"] == "assistant" and history[3]["content"] == result_2["reply"], "Second assistant answer not preserved in history!"
    logger.info("TEST CASE 3: [PASSED]")

    logger.info("\n=== ALL RAG PIPELINE VERIFICATION TESTS COMPLETED SUCCESSFULLY! [100% PASS] ===")


if __name__ == "__main__":
    run_tests()
