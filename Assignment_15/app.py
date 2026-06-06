import streamlit as st
import os
from utils.pdf_processor import extract_text_from_pdf, chunk_text
from utils.embeddings import get_embedding
from utils.pinecone_client import init_pinecone, upsert_chunks, query_pinecone
from groq import Groq

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mini Search Engine",
    page_icon="🔍",
    layout="wide",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --border: #1e1e2e;
    --accent: #6ee7b7;
    --accent2: #818cf8;
    --text: #e2e8f0;
    --muted: #64748b;
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp { background-color: var(--bg); }
h1, h2, h3 { font-family: 'Syne', sans-serif; }

.hero-title {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6ee7b7 0%, #818cf8 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}

.hero-sub {
    font-family: 'Space Mono', monospace;
    color: var(--muted);
    font-size: 0.85rem;
    letter-spacing: 0.1em;
    margin-bottom: 2.5rem;
}

.step-badge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    color: var(--accent);
    border: 1px solid var(--accent);
    border-radius: 2px;
    padding: 2px 8px;
    margin-bottom: 0.75rem;
    text-transform: uppercase;
}

.result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}

.result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
}

.result-filename {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: var(--accent);
    font-weight: 700;
}

.result-score {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: var(--accent2);
    background: rgba(129,140,248,0.08);
    border: 1px solid rgba(129,140,248,0.2);
    border-radius: 4px;
    padding: 2px 10px;
}

.result-text { font-size: 0.9rem; color: #94a3b8; line-height: 1.65; }

.ai-answer {
    background: linear-gradient(135deg, rgba(110,231,183,0.06), rgba(129,140,248,0.06));
    border: 1px solid rgba(110,231,183,0.2);
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.5rem;
    font-size: 0.95rem;
    line-height: 1.7;
    color: var(--text);
}

.ai-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--accent);
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}

.status-ok { color: var(--accent); font-family: 'Space Mono', monospace; font-size: 0.78rem; }
.status-err { color: #f87171; font-family: 'Space Mono', monospace; font-size: 0.78rem; }
.divider { border: none; border-top: 1px solid var(--border); margin: 2rem 0; }

.stFileUploader > div { border-color: var(--border) !important; background: var(--surface) !important; }
.stTextInput input { background: var(--surface) !important; border-color: var(--border) !important; color: var(--text) !important; }
.stButton > button {
    background: linear-gradient(135deg, #6ee7b7, #818cf8) !important;
    color: #0a0a0f !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.6rem 2rem !important;
    font-size: 0.95rem !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Mini Search Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">// SEMANTIC PDF SEARCH · GROQ AI + PINECONE</div>', unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    groq_key = st.text_input("🟢 Groq API Key", type="password", placeholder="gsk_...")
    pinecone_key = st.text_input("🔵 Pinecone API Key", type="password", placeholder="pcsk_...")
    pinecone_index = st.text_input("📌 Pinecone Index Name", placeholder="mini-search-engine")

    st.markdown("---")
    top_k = st.slider("Top-K Results", 1, 10, 5)

    st.markdown("---")
    st.caption("Groq key → console.groq.com/keys")
    st.caption("Pinecone key → app.pinecone.io → API Keys")
    st.caption("Embeddings run FREE locally (no key needed)")

# ─── Main Layout ─────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

# ── LEFT: Upload & Index ──────────────────────────────────────────────────────
with col1:
    st.markdown('<div class="step-badge">Step 01 · Upload</div>', unsafe_allow_html=True)
    st.markdown("### 📄 Upload PDF Documents")
    st.caption("Upload 5 or more PDFs to build your search index.")

    uploaded_files = st.file_uploader(
        "Drop your PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected:**")
        for f in uploaded_files:
            size_kb = len(f.getvalue()) / 1024
            st.markdown(
                f'<span class="status-ok">✓ {f.name}</span> '
                f'<span style="color:#475569;font-size:0.75rem;">({size_kb:.1f} KB)</span>',
                unsafe_allow_html=True
            )
        if len(uploaded_files) < 5:
            st.warning(f"⚠️ Need {5 - len(uploaded_files)} more PDF(s) — minimum is 5.")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    index_button = st.button("⚡ Index Documents", use_container_width=True)

    if index_button:
        if not pinecone_key or not pinecone_index:
            st.error("❌ Enter your Pinecone API key and index name in the sidebar.")
        elif not uploaded_files or len(uploaded_files) < 5:
            st.error("❌ Please upload at least 5 PDF files.")
        else:
            with st.spinner("Connecting to Pinecone..."):
                try:
                    pc_index = init_pinecone(pinecone_key, pinecone_index)
                    st.success("✅ Pinecone connected!")
                except Exception as e:
                    st.error(f"Pinecone error: {e}")
                    st.stop()

            progress = st.progress(0, text="Reading PDFs...")
            all_chunks = []

            for i, pdf_file in enumerate(uploaded_files):
                progress.progress(i / len(uploaded_files), text=f"Reading {pdf_file.name}…")
                try:
                    text = extract_text_from_pdf(pdf_file)
                    chunks = chunk_text(text, pdf_file.name)
                    all_chunks.extend(chunks)
                    st.markdown(
                        f'<span class="status-ok">✓ {pdf_file.name} → {len(chunks)} chunks</span>',
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.markdown(f'<span class="status-err">✗ {pdf_file.name}: {e}</span>', unsafe_allow_html=True)

            progress.progress(0.8, text="Generating embeddings (this may take a minute)…")
            try:
                upsert_chunks(pc_index, all_chunks)
                progress.progress(1.0, text="Done!")
                st.success(f"🎉 {len(all_chunks)} chunks indexed successfully!")
                st.balloons()
            except Exception as e:
                st.error(f"Indexing error: {e}")

# ── RIGHT: Search ─────────────────────────────────────────────────────────────
with col2:
    st.markdown('<div class="step-badge">Step 02 · Search</div>', unsafe_allow_html=True)
    st.markdown("### 🔍 Search Your Documents")
    st.caption("Ask anything in plain English — Groq AI will answer based on your PDFs.")

    query = st.text_input(
        "Your question",
        placeholder="e.g. What are the main findings about climate change?",
        label_visibility="collapsed"
    )
    search_button = st.button("🔎 Search", use_container_width=True)

    if search_button:
        if not pinecone_key or not pinecone_index:
            st.error("❌ Fill in Pinecone API key and index name in the sidebar.")
        elif not query.strip():
            st.warning("⚠️ Please enter a search query.")
        else:
            with st.spinner("Searching your documents…"):
                try:
                    pc_index = init_pinecone(pinecone_key, pinecone_index)
                    query_emb = get_embedding(query)
                    results = query_pinecone(pc_index, query_emb, top_k)

                    if not results:
                        st.info("No results found. Make sure you indexed your documents first.")
                    else:
                        # ── Groq AI Answer ────────────────────────────────
                        if groq_key:
                            context = "\n\n".join(
                                [f"[{r['metadata'].get('filename','?')}]: {r['metadata'].get('text','')}"
                                 for r in results]
                            )
                            try:
                                client = Groq(api_key=groq_key)
                                chat = client.chat.completions.create(
                                    model="llama3-8b-8192",
                                    messages=[
                                        {"role": "system", "content": (
                                            "You are a helpful assistant. Answer the user's question "
                                            "using ONLY the context provided. Be concise and clear. "
                                            "If the context doesn't contain the answer, say so."
                                        )},
                                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                                    ],
                                    max_tokens=400,
                                )
                                ai_answer = chat.choices[0].message.content
                                st.markdown('<div class="ai-label">🤖 AI ANSWER (GROQ · LLAMA 3)</div>', unsafe_allow_html=True)
                                st.markdown(f'<div class="ai-answer">{ai_answer}</div>', unsafe_allow_html=True)
                            except Exception as e:
                                st.warning(f"Groq AI answer skipped: {e}")

                        # ── Raw Results ───────────────────────────────────
                        st.markdown(f"**Top {len(results)} matching chunks:**")
                        st.markdown('<hr class="divider">', unsafe_allow_html=True)

                        for match in results:
                            meta = match.get("metadata", {})
                            score = match.get("score", 0)
                            filename = meta.get("filename", "Unknown")
                            text_chunk = meta.get("text", "")

                            st.markdown(f"""
                            <div class="result-card">
                                <div class="result-header">
                                    <span class="result-filename">📄 {filename}</span>
                                    <span class="result-score">Score: {score*100:.1f}%</span>
                                </div>
                                <div class="result-text">{text_chunk[:500]}{'…' if len(text_chunk)>500 else ''}</div>
                            </div>
                            """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Search error: {e}")
