"""
server.py — Flask backend for the Mailchimp Style Guide RAG chatbot.

Endpoints:
    POST /chat      { "question": "..." }  →  { "answer": "...", "sources": [...] }
    GET  /health    → { "status": "ok" }
    GET  /widget    → serves chat_widget.html

Usage:
    python server.py
    # Server runs on http://0.0.0.0:5000
"""

import os
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from pinecone import Pinecone
from fastembed import TextEmbedding
import anthropic

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
PINECONE_API_KEY  = os.getenv("PINECONE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
INDEX_NAME        = "mailchimp-style-guide"
EMBEDDING_MODEL   = "BAAI/bge-small-en-v1.5"  # 384-dim, ONNX-based (no PyTorch needed)
TOP_K             = 6     # number of chunks to retrieve
MAX_TOKENS        = 1024  # max tokens for Claude's answer
PORT              = int(os.getenv("PORT", 5000))
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about Mailchimp's Content Style Guide.

Rules you must follow:
1. Answer ONLY using the style guide excerpts provided in each user message.
2. Do NOT draw on general knowledge or invent guidance not present in the excerpts.
3. If the excerpts do not contain enough information to answer the question, say exactly:
   "I couldn't find that in the Mailchimp Content Style Guide. Try checking the full guide at https://styleguide.mailchimp.com"
4. Always cite your sources. After your answer, list the specific file(s) and section(s) you drew from, like:
   **Sources:** 01-writing-for-accessibility.md › Writing for accessibility
5. Be concise. Quote the guide directly when it adds clarity.
"""

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins="*")  # allow cross-origin requests from any domain

print("Loading embedding model…")
embedder = TextEmbedding(EMBEDDING_MODEL)

print("Connecting to Pinecone…")
pc    = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

print("Initialising Anthropic client…")
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

print(f"Ready — listening on port {PORT}\n")
# ──────────────────────────────────────────────────────────────────────────────


@app.route("/chat", methods=["POST"])
def chat():
    data     = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "No question provided."}), 400

    # 1. Embed the question
    q_vector = next(embedder.embed([question])).tolist()

    # 2. Retrieve top-k chunks from Pinecone
    results = index.query(vector=q_vector, top_k=TOP_K, include_metadata=True)

    if not results.matches:
        return jsonify({
            "answer":  "I couldn't find relevant content in the Mailchimp Content Style Guide.",
            "sources": [],
        })

    # 3. Build context string + deduplicated source list
    context_blocks = []
    seen_sources   = []

    for i, match in enumerate(results.matches, 1):
        meta    = match.metadata or {}
        text    = meta.get("text", "").strip()
        f_name  = meta.get("source_file", "unknown")
        section = meta.get("section",     "unknown")

        context_blocks.append(
            f"[Excerpt {i} | File: {f_name} | Section: {section}]\n{text}"
        )

        src = {"file": f_name, "section": section}
        if src not in seen_sources:
            seen_sources.append(src)

    context = "\n\n---\n\n".join(context_blocks)

    # 4. Call Claude with context + question
    user_message = f"""Style guide excerpts (use ONLY these to answer):

{context}

---

Question: {question}"""

    response = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    answer = response.content[0].text

    return jsonify({"answer": answer, "sources": seen_sources})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/widget", methods=["GET"])
def widget():
    widget_path = Path(__file__).parent / "chat_widget.html"
    return send_file(widget_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
