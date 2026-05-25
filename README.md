# GenAI Assistant with RAG – Enterprise Customer Support Hub

A production-grade, highly optimized **Retrieval-Augmented Generation (RAG) Chat Assistant** designed to answer customer queries with zero hallucinations, strictly grounded in a document knowledge base. 

Developed with a modular **FastAPI Python backend** and a premium, **responsive glassmorphic HTML/CSS/JS frontend**, this system features a dual API provider setup (Gemini & OpenAI) along with a deterministic **offline topic-based projection fallback** for complete out-of-the-box local testing.

---

## 1. System Architecture

The overall structure of our RAG Chat Assistant represents a standard pipeline where all requests are routed, filtered by a similarity threshold, and grounded securely before LLM generation.

```
                  ┌──────────────────────────────────────────────┐
                  │                 USER BROWSER                 │
                  │   - Premium Single Page Application (SPA)     │
                  │   - Real-time Vector DB Sidebar Viewer       │
                  │   - Detailed RAG Grounding Inspector Panel   │
                  └──────────────┬──────────────────▲────────────┘
                                 │                  │
                1. User Query    │                  │ 14. JSON Grounded Response
                & Session ID     │                  │     (including chunks + metrics)
                                 ▼                  │
                  ┌─────────────────────────────────┴────────────┐
                  │                FASTAPI BACKEND               │
                  │   - Inputs Validation & Route Mapping        │
                  │   - Static File Mounting / SPA Routing       │
                  └──────────────┬──────────────────▲────────────┘
                                 │                  │
               2. Match query    │                  │ 13. Formulate RAG Output
                                 ▼                  │
                  ┌─────────────────────────────────┴────────────┐
                  │                 RAG COORDINATOR              │
                  │   - Generates query embedding                │
                  │   - Similarity Search & Ranking              │
                  │   - Applies confidence thresholds (>0.70)    │
                  └──────────────┬──────────────────▲────────────┘
                                 │                  │
            3. Call      ┌───────▼───────┐  ┌───────┴───────┐ 12. Grounded
            Embedding    │  Embeddings   │  │   LLM API     │     Response
            & API Key    │  Generator    │  │   Connector   │     (temp=0.2)
                         └───────┬───────┘  └───────▲───────┘
                                 │                  │
                                 ▼                  │ 11. Low-Temp Prompt
                        ┌─────────────────┐         │     (XML System Instructions)
                        │ LLM/Embedding   │─────────┘
                        │ API (Gemini/OAI)│
                        │   or LOCAL      │
                        │ Topic-Projector │
                        └─────────────────┘
```

---

## 2. RAG Workflow Explanation

The RAG workflow enforces rigorous truthfulness by executing the following sequential pipeline for every query:
1. **Query Ingestion & Validation:** The API validates that the `sessionId` is present and the `message` contains non-whitespace text.
2. **Query Vectorization:** The query text is transformed into a high-dimensional vector space using the selected embedding model.
3. **Semantic Similarity Search:** The query vector is compared against all document chunks in our database using **Cosine Similarity**.
4. **Rank and Filter:**
   - Chunks are ranked by score in descending order.
   - Chunks falling below the **Similarity Threshold (0.70)** are discarded.
   - If **zero chunks** pass the threshold, the system immediately returns a standard out-of-bounds rejection: *"I do not have enough information to answer that."* This saves API call cost and prevents LLM hallucination.
5. **Context Aggregation:** The contents of the top 3 matching chunks are aggregated into a single context block.
6. **Session History Retrieval:** The system retrieves the last 4 message pairs (User/Assistant) for the specific `sessionId` to maintain multi-turn conversational context.
7. **Prompt Construction:** The context, conversation history, and current question are injected into an XML-delimited prompt template instructing the LLM to rely *only* on the provided facts.
8. **Low-Temperature Response Generation:** The LLM is queried with a low temperature (`0.2`), generating a factual, grounded response.
9. **History Logging:** The user query and grounded answer are added to the session history storage for subsequent turns.

---

## 3. Embedding Strategy

Our system implements a **dual API-based neural embedding strategy** alongside a **deterministic offline topic-projection algorithm**:

### Neural API Embeddings
When provided with valid API keys:
- **Gemini Embedding Model:** Uses `models/text-embedding-004` (Google's state-of-the-art embedding model) optimized for document retrieval tasks.
- **OpenAI Embedding Model:** Uses `text-embedding-3-small` (1536 dimensions) for high-performance retrieval.

### Deterministic Offline Fallback (Topic Projection Vectorizer)
To guarantee the RAG system works flawlessly **without API keys**, we developed a custom **Deterministic Topic-Based Projection Vectorizer**.
- It projects text onto **6 key semantic topics** mapped to our database categories: Reset Password, Account Deletion, Refund Policy, Subscription Tiers, 2FA Security, and Technical Support.
- Matching terms in these topics populate specific index dimensions with indicator values.
- General vocabulary variations are mapped using feature hashing onto the remaining dimensions.
- Capping word occurrences and unit-normalizing the vectors ensures that relevant overlaps result in a cosine similarity score of **>0.80**, while out-of-bounds queries yield **<0.10** similarity.

---

## 4. Similarity Search Logic

We use **Cosine Similarity** to mathematically measure the angle between vectors, which evaluates semantic closeness independently of document length.

$$\text{Cosine Similarity}(\vec{A}, \vec{B}) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}$$

Implemented using highly optimized **NumPy** matrix functions:
```python
def cosine_similarity(vec1: list, vec2: list) -> float:
    a = np.array(vec1, dtype=float)
    b = np.array(vec2, dtype=float)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))
```

---

## 5. Prompt Design Reasoning

To keep the AI strictly grounded and prevent conversational drift, the system utilizes structured XML-tagged prompts.

### Template Structure:
```
You are a highly helpful and factual customer support AI assistant.
Your goal is to answer the user's question using ONLY the provided Context.

STRICT INSTRUCTIONS:
- You must rely ONLY on the provided Context to answer the question.
- Do not make up facts or use external training knowledge if it is not grounded in the Context.
- If the Context does not provide the answer, say exactly: "I do not have enough information to answer that."

Context:
[Document: Reset Password Instruction (Category: Security)]
To reset your password, navigate to Settings > Security & Login...

History:
User: How do I reset my password?
Assistant: Navigate to Settings > Security...

Question:
Where is that option?

Answer:
```

### Rationale:
- **Low Temperature (`temperature=0.2`):** Forces the model to choose high-probability tokens, making the response highly predictable and grounded.
- **Negative Constraints:** Explicitly defining what the model *should not do* ("Do not make up facts...", "If the Context does not provide the answer...") dramatically reduces hallucinations.
- **Structured Fields:** Labeling sections (`Context:`, `History:`, `Question:`) prevents the LLM from confusing user input with system instructions (Prompt Injection protection).

---

## 6. Setup & Installation Instructions

Follow these steps to run the application locally on Windows:

### 1. Clone or Open Project Folder
Open the project directory in VS Code or your terminal:
```powershell
cd "c:\Users\krish\OneDrive\Desktop\KRISHNA DOCS\projects\PROJECT1"
```

### 2. Configure Environment Variables
Create or open the `.env` file in the root directory. Add your Gemini or OpenAI API Key if you have one, or keep the defaults for the deterministic offline mode:
```env
# Choose "gemini", "openai" or "local" (fallback)
LLM_PROVIDER=gemini
LLM_API_KEY=your_gemini_api_key_here

EMBEDDING_PROVIDER=gemini
EMBEDDING_API_KEY=your_gemini_api_key_here

# Similarity Threshold (0.0 to 1.0)
SIMILARITY_THRESHOLD=0.70
```

### 3. Install Dependencies
Run the following command to install the required packages:
```powershell
python -m pip install -r requirements.txt
```

### 4. Run Automated Test Verification Suite
Run the test suite to verify the vectorizer, chunking, similarity thresholding, and multi-turn session history in a sandbox:
```powershell
python backend/test_rag.py
```
*Expected Output:* `=== ALL RAG PIPELINE VERIFICATION TESTS COMPLETED SUCCESSFULLY! [100% PASS] ===`

### 5. Start the FastAPI Production Server
Start the Uvicorn server to host the backend API and serve the premium frontend files:
```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### 6. Access the Application
Open your browser and navigate to:
[http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 7. Premium Frontend Features

Our Single Page Application (SPA) has been engineered with a sleek **dark glassmorphic UI** to wow users and provide a transparent inspection of the RAG pipeline:
1. **Interactive Vector DB Sidebar:** Visualizes every chunk actively loaded inside the database. Users can search and filter these chunks in real-time, or click them to expand and read details.
2. **RAG Grounding Inspector Panel:** Under every AI response, a collapsible grounding drawer allows users to inspect RAG metrics:
   - Tokens consumed.
   - Number of chunks retrieved.
   - Exact cosine similarity matching scores for each chunk (color-coded as Green for High, Yellow for Medium, and Red for Low).
   - Text contents of the grounding chunks.
3. **Session-based Memory:** Sessions are persisted in `localStorage` and display in the header. A reload button allows restarting sessions to verify clean-state memory boundaries.
4. **Interactive Suggestion Tags:** Provides quick-start questions including valid queries (which return beautiful answers) and out-of-bounds queries (which demonstrate threshold rejection).
