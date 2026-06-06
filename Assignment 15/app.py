import streamlit as st
import os
from utils.pdf_processor import extract_text_from_pdf, chunk_text
from utils.embeddings import get_embedding
from utils.pinecone_client import init_pinecone, upsert_chunks, query_pinecone

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
    transition: border-color 0.2s;
}

.result-card:hover { border-left-color: var(--accent2); }

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
    letter-spacing: 0.05em;
}

.result-score {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: var(--accent2);
    background: rgba(129, 140, 248, 0.08);
    border: 1px solid rgba(129, 140, 248, 0.2);
    border-radius: 4px;
    padding: 2px 10px;
}

.result-text {
    font-size: 0.9rem;
    color: #94a3b8;
    line-height: 1.65;
}

.status-ok {
    color: var(--accent);
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
}

.status-err {
    color: #f87171;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
}

.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 2rem 0;
}

/* Streamlit widget overrides */
.stFileUploader > div { border-color: var(--border) !important; background: var(--surface) !important; }
.stTextInput input { background: var(--surface) !important; border-color: var(--border) !important; color: var(--text) !important; font-family: 'Syne', sans-serif !important; }
.stButton > button {
    background: linear-gradient(135deg, #6ee7b7, #818cf8) !important;
    color: #0a0a0f !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.6rem 2rem !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.03em;
}
.stButton > button:hover { opacity: 0.9 !important; }
.stSlider > div > div { color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Mini Search Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">// SEMANTIC PDF SEARCH · POWERED BY PINECONE + OPENAI</div>', unsafe_allow_html=True)

# ─── Sidebar – API Keys ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    openai_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    pinecone_key = st.text_input("Pinecone API Key", type="password", placeholder="pcsk_...")
    pinecone_index = st.text_input("Pinecone Index Name", placeholder="mini-search-engine")

    st.markdown("---")
    st.markdown("**Top-K Results**")
    top_k = st.slider("Number of results", 1, 10, 5)

    st.markdown("---")
    st.markdown('<span class="status-ok">How to get keys →</span>', unsafe_allow_html=True)
    st.caption("• OpenAI: platform.openai.com/api-keys")
    st.caption("• Pinecone: app.pinecone.io → API Keys")

# ─── Main Layout ─────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

# ── LEFT: Upload ──────────────────────────────────────────────────────────────
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
            st.markdown(f'<span class="status-ok">✓ {f.name}</span> <span style="color:#475569;font-size:0.75rem;">({size_kb:.1f} KB)</span>', unsafe_allow_html=True)

        if len(uploaded_files) < 5:
            st.warning(f"⚠️ Please upload at least 5 PDFs ({5 - len(uploaded_files)} more needed).")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    index_button = st.button("⚡ Index Documents", use_container_width=True)

    if index_button:
        if not openai_key or not pinecone_key or not pinecone_index:
            st.error("❌ Please fill in all API keys and index name in the sidebar.")
        elif not uploaded_files or len(uploaded_files) < 5:
            st.error("❌ Please upload at least 5 PDF files.")
        else:
            os.environ["OPENAI_API_KEY"] = openai_key

            with st.spinner("Initialising Pinecone..."):
                try:
                    pc_index = init_pinecone(pinecone_key, pinecone_index)
                    st.success("✅ Pinecone connected!")
                except Exception as e:
                    st.error(f"Pinecone error: {e}")
                    st.stop()

            progress = st.progress(0, text="Processing PDFs...")
            all_chunks = []

            for i, pdf_file in enumerate(uploaded_files):
                progress.progress((i) / len(uploaded_files), text=f"Reading {pdf_file.name}…")
                try:
                    text = extract_text_from_pdf(pdf_file)
                    chunks = chunk_text(text, pdf_file.name)
                    all_chunks.extend(chunks)
                    st.markdown(f'<span class="status-ok">✓ {pdf_file.name} → {len(chunks)} chunks</span>', unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<span class="status-err">✗ {pdf_file.name}: {e}</span>', unsafe_allow_html=True)

            progress.progress(0.8, text="Generating embeddings & uploading to Pinecone…")
            try:
                upsert_chunks(pc_index, all_chunks, openai_key)
                progress.progress(1.0, text="Done!")
                st.success(f"🎉 {len(all_chunks)} chunks indexed successfully!")
            except Exception as e:
                st.error(f"Embedding/upsert error: {e}")

# ── RIGHT: Search ─────────────────────────────────────────────────────────────
with col2:
    st.markdown('<div class="step-badge">Step 02 · Search</div>', unsafe_allow_html=True)
    st.markdown("### 🔍 Search Your Documents")
    st.caption("Ask anything — natural language questions work best.")

    query = st.text_input("Enter your query", placeholder="e.g. What are the main findings about climate change?", label_visibility="collapsed")

    search_button = st.button("🔎 Search", use_container_width=True)

    if search_button:
        if not openai_key or not pinecone_key or not pinecone_index:
            st.error("❌ Fill in all API keys in the sidebar first.")
        elif not query.strip():
            st.warning("⚠️ Please enter a search query.")
        else:
            os.environ["OPENAI_API_KEY"] = openai_key
            with st.spinner("Searching…"):
                try:
                    pc_index = init_pinecone(pinecone_key, pinecone_index)
                    query_emb = get_embedding(query, openai_key)
                    results = query_pinecone(pc_index, query_emb, top_k)

                    if not results:
                        st.info("No results found. Make sure you have indexed documents first.")
                    else:
                        st.markdown(f"**{len(results)} result(s) for:** *{query}*")
                        st.markdown('<hr class="divider">', unsafe_allow_html=True)

                        for i, match in enumerate(results):
                            meta = match.get("metadata", {})
                            score = match.get("score", 0)
                            filename = meta.get("filename", "Unknown")
                            text_chunk = meta.get("text", "")
                            score_pct = f"{score * 100:.1f}%"

                            st.markdown(f"""
                            <div class="result-card">
                                <div class="result-header">
                                    <span class="result-filename">📄 {filename}</span>
                                    <span class="result-score">Score: {score_pct}</span>
                                </div>
                                <div class="result-text">{text_chunk[:500]}{'…' if len(text_chunk) > 500 else ''}</div>
                            </div>
                            """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Search error: {e}")
