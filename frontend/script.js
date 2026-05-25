// State Management
let currentSessionId = "";
let knowledgeBaseChunks = [];

// DOM Elements
const chatMessages = document.getElementById("chat-messages");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const btnSend = document.getElementById("btn-send");
const chatLoading = document.getElementById("chat-loading");
const sessionDisplay = document.getElementById("session-display");
const btnNewSession = document.getElementById("btn-new-session");
const systemStatus = document.getElementById("system-status");
const docCountBadge = document.getElementById("doc-count");
const docList = document.getElementById("knowledge-base-docs");
const docSearch = document.getElementById("doc-search");
const errorNotification = document.getElementById("error-notification");
const errorTitle = document.getElementById("error-title");
const errorText = document.getElementById("error-text");
const btnClearError = document.getElementById("btn-clear-error");
const valThreshold = document.getElementById("val-threshold");

// App Startup Initializer
window.addEventListener("DOMContentLoaded", () => {
    initializeSession();
    fetchKnowledgeBase();
    setupEventListeners();
});

// Session Management (Step 12 & Step 14)
function initializeSession(forceNew = false) {
    let sessionId = localStorage.getItem("rag_session_id");
    
    if (!sessionId || forceNew) {
        // Generate robust 8-character unique alphanumeric ID
        sessionId = "sess_" + Math.random().toString(36).substring(2, 10);
        localStorage.setItem("rag_session_id", sessionId);
        
        // If forcing new, clear chat layout and re-render welcome greeting
        if (forceNew) {
            chatMessages.innerHTML = "";
            renderWelcomeCard();
        }
    }
    
    currentSessionId = sessionId;
    sessionDisplay.textContent = sessionId;
    console.log(`Active session ID set to: ${currentSessionId}`);
}

// Event Listeners
function setupEventListeners() {
    // Input keyup validator
    chatInput.addEventListener("input", () => {
        btnSend.disabled = chatInput.value.trim().length === 0;
    });

    // Chat submit handler
    chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const message = chatInput.value.trim();
        if (!message) return;
        
        sendMessage(message);
    });

    // New session button
    btnNewSession.addEventListener("click", () => {
        if (confirm("Are you sure you want to start a new session? This will clear active conversation history.")) {
            initializeSession(true);
        }
    });

    // Close error notification
    btnClearError.addEventListener("click", () => {
        errorNotification.classList.add("hidden");
    });

    // Document search filter
    docSearch.addEventListener("input", (e) => {
        const query = e.target.value.toLowerCase().trim();
        filterDocs(query);
    });

    // Global delegation for click on suggestion tags (Step 14)
    document.addEventListener("click", (e) => {
        const tag = e.target.closest(".suggest-tag");
        if (tag) {
            const query = tag.getAttribute("data-query");
            chatInput.value = query;
            btnSend.disabled = false;
            sendMessage(query);
        }
    });
}

// Fetch documents in the knowledge base (Step 2)
async function fetchKnowledgeBase() {
    try {
        const response = await fetch("/api/docs");
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        
        knowledgeBaseChunks = await response.json();
        renderKnowledgeBaseList(knowledgeBaseChunks);
        docCountBadge.textContent = `${knowledgeBaseChunks.length} Chunks`;
        
        // Update active server provider status
        updateSystemStatus();
    } catch (err) {
        console.error("Failed to fetch knowledge base documents:", err);
        docList.innerHTML = `
            <div class="loading-sidebar-spinner text-danger">
                <i class="fa-solid fa-triangle-exclamation"></i>
                <span style="color: var(--danger)">Connection failed.</span>
            </div>
        `;
    }
}

// Update Active System Provider from environment details
function updateSystemStatus() {
    // Determine provider and threshold from active connection parameters
    systemStatus.innerHTML = `<i class="fa-solid fa-cloud-bolt" style="color: var(--secondary)"></i> Connected to FastAPI Server`;
}

// Render document items in the sidebar
function renderKnowledgeBaseList(chunks) {
    if (chunks.length === 0) {
        docList.innerHTML = `<div class="loading-sidebar-spinner"><span>No documents found.</span></div>`;
        return;
    }
    
    docList.innerHTML = chunks.map(chunk => `
        <div class="doc-card" onclick="expandDocChunk('${chunk.id}')" id="chunk-card-${chunk.id}">
            <div class="doc-card-header">
                <span class="doc-card-title">${escapeHTML(chunk.title)}</span>
                <span class="doc-card-tag">${escapeHTML(chunk.category)}</span>
            </div>
            <div class="doc-card-body" id="chunk-body-${chunk.id}">
                ${escapeHTML(chunk.content)}
            </div>
        </div>
    `).join("");
}

// Filter knowledge chunks in real time
function filterDocs(query) {
    const cards = docList.getElementsByClassName("doc-card");
    
    for (let i = 0; i < cards.length; i++) {
        const title = cards[i].querySelector(".doc-card-title").textContent.toLowerCase();
        const body = cards[i].querySelector(".doc-card-body").textContent.toLowerCase();
        const tag = cards[i].querySelector(".doc-card-tag").textContent.toLowerCase();
        
        if (title.includes(query) || body.includes(query) || tag.includes(query)) {
            cards[i].style.display = "";
        } else {
            cards[i].style.display = "none";
        }
    }
}

// Expand a document chunk inside sidebar to read full context
function expandDocChunk(id) {
    const card = document.getElementById(`chunk-card-${id}`);
    const body = document.getElementById(`chunk-body-${id}`);
    
    if (body.style.display === "block" || body.style.webkitLineClamp === "unset") {
        body.style.webkitLineClamp = "3";
        body.style.display = "-webkit-box";
        card.style.borderColor = "var(--border-color)";
    } else {
        body.style.display = "block";
        body.style.webkitLineClamp = "unset";
        card.style.borderColor = "var(--secondary)";
    }
}

// Formulate message and transmit to backend (Step 13)
async function sendMessage(text) {
    // Clear input bar and lock button
    chatInput.value = "";
    btnSend.disabled = true;
    errorNotification.classList.add("hidden");

    // Remove the welcome message on first search if present
    const welcome = document.querySelector(".system-welcome");
    if (welcome) {
        welcome.remove();
    }

    // Render User bubble (Step 14)
    appendBubble("user", text);
    scrollToBottom();

    // Show loading indicators
    chatLoading.classList.add("active");
    scrollToBottom();

    const payload = {
        sessionId: currentSessionId,
        message: text
    };

    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!response.ok) {
            throw { status: response.status, body: data };
        }

        // Render response bubble
        appendBubble("assistant", data.reply, data);
        
    } catch (err) {
        console.error("API Call error:", err);
        handleAPIError(err);
    } finally {
        chatLoading.classList.remove("active");
        scrollToBottom();
    }
}

// Gracefully handle LLM server failures (Step 11)
function handleAPIError(err) {
    let errorHeadline = "Backend Server Error";
    let errorMessage = "Failed to communicate with FastAPI backend server.";
    
    if (err.status && err.body) {
        const status = err.status;
        const body = err.body;
        
        // Exact API errors required by Step 11
        if (status === 401) {
            errorHeadline = "Unauthorized API Request";
            errorMessage = "Invalid API key. Please check LLM_API_KEY / EMBEDDING_API_KEY variables in your backend .env file.";
        } else if (status === 429) {
            errorHeadline = "Rate Limit Exceeded";
            errorMessage = "Rate limit reached. Too many requests have been sent. Please wait and try again shortly.";
        } else if (status === 504) {
            errorHeadline = "Request Timeout";
            errorMessage = "LLM API provider request timed out. The server took too long to generate a grounded reply.";
        } else if (body.detail) {
            errorMessage = body.detail;
        } else if (body.error) {
            errorMessage = body.error;
        }
    } else if (err.message) {
        errorMessage = err.message;
    }
    
    // Display error banner
    errorTitle.textContent = errorHeadline;
    errorText.textContent = errorMessage;
    errorNotification.classList.remove("hidden");
    
    // Play subtle visual wiggle on the notification banner
    errorNotification.style.animation = "shake 0.4s ease";
    setTimeout(() => {
        errorNotification.style.animation = "";
    }, 400);
}

// Append Chat Message Bubble (Step 14)
function appendBubble(role, content, ragDetails = null) {
    const chatRow = document.createElement("div");
    chatRow.className = `chat-row ${role}-row`;
    
    const avatar = document.createElement("div");
    avatar.className = "bubble-avatar";
    avatar.innerHTML = role === "user" ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';
    
    const textWrapper = document.createElement("div");
    textWrapper.style.display = "flex";
    textWrapper.style.flexDirection = "column";
    textWrapper.style.width = "100%";
    
    const textBubble = document.createElement("div");
    textBubble.className = "bubble-text";
    textBubble.innerHTML = formatMarkdown(content);
    
    textWrapper.appendChild(textBubble);
    
    // If response contains retrieved chunks details, append the interactive RAG Inspection Panel!
    if (ragDetails) {
        const inspectionPanel = buildRAGInspectionPanel(ragDetails);
        textWrapper.appendChild(inspectionPanel);
    }
    
    chatRow.appendChild(avatar);
    chatRow.appendChild(textWrapper);
    
    chatMessages.appendChild(chatRow);
}

// Build beautiful RAG metadata viewer
function buildRAGInspectionPanel(details) {
    const container = document.createElement("div");
    container.className = "rag-meta-box";
    
    const toggle = document.createElement("div");
    toggle.className = "rag-header-toggle";
    
    const hasChunks = details.chunks && details.chunks.length > 0;
    const chunkBadgeColor = hasChunks ? "var(--secondary)" : "var(--text-dark)";
    
    toggle.innerHTML = `
        <span>
            <i class="fa-solid fa-circle-nodes" style="color: ${chunkBadgeColor}"></i> 
            RAG Grounding: ${details.retrievedChunks} chunks retrieved
        </span>
        <i class="fa-solid fa-chevron-down"></i>
    `;
    
    const body = document.createElement("div");
    body.className = "rag-body-details";
    body.style.display = "none";
    
    // Build internal inspector elements
    let chunkItemsHTML = "";
    if (hasChunks) {
        details.chunks.forEach(c => {
            let scoreClass = "low";
            if (c.score >= 0.85) scoreClass = "high";
            else if (c.score >= 0.70) scoreClass = "medium";
            
            chunkItemsHTML += `
                <div class="rag-chunk-item">
                    <div class="chunk-meta">
                        <span class="chunk-source-title"><i class="fa-solid fa-file-lines text-muted" style="margin-right: 4px;"></i> ${escapeHTML(c.title)}</span>
                        <span class="chunk-score ${scoreClass}">Similarity: ${c.score}</span>
                    </div>
                    <div class="chunk-text-snippet">"${escapeHTML(c.content)}"</div>
                </div>
            `;
        });
    } else {
        chunkItemsHTML = `<div class="rag-chunk-item" style="border-left-color: var(--danger)">
            <div class="chunk-text-snippet">No chunks passed the similarity threshold (${valThreshold.textContent}). Bypassed LLM connection to prevent hallucinations.</div>
        </div>`;
    }
    
    body.innerHTML = `
        <span class="rag-token-badge">
            <i class="fa-solid fa-receipt"></i> Tokens consumed: ~${details.tokensUsed}
        </span>
        ${chunkItemsHTML}
    `;
    
    // Toggle trigger
    toggle.addEventListener("click", () => {
        const isCollapsed = body.style.display === "none";
        body.style.display = isCollapsed ? "flex" : "none";
        toggle.classList.toggle("expanded", isCollapsed);
        scrollToBottom();
    });
    
    container.appendChild(toggle);
    container.appendChild(body);
    
    return container;
}

// Re-render greeting card
function renderWelcomeCard() {
    const wrapper = document.createElement("div");
    wrapper.className = "message-bubble system-welcome";
    wrapper.innerHTML = `
        <div class="welcome-card">
            <i class="fa-solid fa-wand-magic-sparkles welcome-icon"></i>
            <h3>Welcome to the GenAI RAG Hub</h3>
            <p>
                This assistant is powered by an active **Retrieval-Augmented Generation (RAG)** pipeline.
                Every query runs a real-time semantic cosine similarity search against our document database, returning grounded context to feed the LLM.
            </p>
            
            <div class="suggestion-section">
                <h4>Try a recommended search:</h4>
                <div class="suggestion-tags">
                    <button class="suggest-tag" data-query="How can I reset my password?">
                        <i class="fa-solid fa-key"></i> How do I reset my password?
                    </button>
                    <button class="suggest-tag" data-query="What is your policy on refunds?">
                        <i class="fa-solid fa-hand-holding-dollar"></i> What's the refund policy?
                    </button>
                    <button class="suggest-tag" data-query="Tell me about subscription tiers and pricing.">
                        <i class="fa-solid fa-tags"></i> Subscription tiers
                    </button>
                    <button class="suggest-tag" data-query="What is your policy on shipping items to Mars?">
                        <i class="fa-solid fa-ban"></i> Shipping to Mars (Forces Fallback)
                    </button>
                </div>
            </div>
        </div>
    `;
    chatMessages.appendChild(wrapper);
}

// Helper: Custom Micro-Markdown Formatter
function formatMarkdown(text) {
    if (!text) return "";
    
    let html = escapeHTML(text);
    
    // Bold tags (**text**)
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // In-line code fragments (`code`)
    html = html.replace(/`(.*?)`/g, '<code>$1</code>');
    
    // Multi-line spacing (converts double returns to paragraphs)
    html = html.split("\n\n").map(paragraph => `<p>${paragraph.replace(/\n/g, "<br>")}</p>`).join("");
    
    return html;
}

// Helper: Secure HTML escape to avoid XSS injections
function escapeHTML(str) {
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// Helper: Auto-scroller
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}
