import streamlit as st
import fitz  # PyMuPDF
import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# ─── Load secrets from .env ───────────────────────────────────────────────────
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
INDEX_NAME = "mini-search-engine"

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentence Predictor",
    page_icon="🧠",
    layout="wide",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #e2e8f0;
    }

    .stApp {
        background: radial-gradient(circle at top left, rgba(96,165,250,0.16), transparent 25%),
                    radial-gradient(circle at top right, rgba(168,85,247,0.12), transparent 25%),
                    linear-gradient(145deg, #0b1120 0%, #1f2335 40%, #111827 100%);
        min-height: 100vh;
    }

    .hero {
        border-radius: 28px;
        background: rgba(15,23,42,0.82);
        border: 1px solid rgba(148,163,184,0.14);
        backdrop-filter: blur(18px);
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 3rem;
        letter-spacing: -0.04em;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p {
        color: #cbd5e1;
        font-size: 1.05rem;
        margin-top: 0.75rem;
    }

    .card {
        border-radius: 22px;
        background: rgba(15,23,42,0.8);
        border: 1px solid rgba(148,163,184,0.12);
        padding: 1.5rem;
        box-shadow: 0 18px 50px rgba(0,0,0,0.18);
        margin-bottom: 1.5rem;
    }

    .mini-card {
        border-radius: 18px;
        background: rgba(51,65,85,0.75);
        border: 1px solid rgba(148,163,184,0.1);
        padding: 1rem 1.25rem;
        color: #e2e8f0;
    }

    .mini-card h3 {
        margin: 0 0 0.4rem 0;
        font-size: 1.1rem;
    }
    .mini-card p {
        margin: 0;
        color: #94a3b8;
    }

    .section-label {
        color: #94a3b8;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.8rem;
    }

    .result-card {
        border-radius: 20px;
        background: rgba(15,23,42,0.85);
        border: 1px solid rgba(148,163,184,0.12);
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 40px rgba(15,23,42,0.35);
    }
    .result-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        flex-wrap: wrap;
        margin-bottom: 0.8rem;
    }
    .result-filename {
        font-weight: 700;
        color: #7c3aed;
        font-size: 1rem;
    }
    .result-score {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: linear-gradient(135deg, #818cf8, #a78bfa);
        color: white;
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 700;
    }
    .result-text {
        color: #cbd5e1;
        line-height: 1.8;
        font-size: 0.95rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #22d3ee);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 0.75rem 1.2rem;
        font-weight: 700;
        font-size: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 16px 32px rgba(34,211,238,0.25);
    }

    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(147,197,253,0.25);
        border-radius: 14px;
        color: #e2e8f0;
        padding: 0.9rem 1rem;
        font-size: 1rem;
        box-shadow: inset 0 0 0 1px rgba(148,163,184,0.06);
    }
    .stTextInput > div > div > input:focus {
        border-color: #60a5fa;
        box-shadow: 0 0 0 3px rgba(96,165,250,0.12);
    }

    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.06);
        border: 2px dashed rgba(96,165,250,0.45);
        border-radius: 18px;
        padding: 1.2rem;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(96,165,250,0.85);
    }

    [data-testid="collapsedControl"] { display: none; }
    #MainMenu, footer, header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Load embedding model (cached) ───────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ─── Helpers ──────────────────────────────────────────────────────────────────
def extract_text_from_pdf(file) -> str:
    try:
        file.seek(0)
        file_bytes = file.read()
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)
    except Exception:
        return ""


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    if not text:
        return []
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def get_pinecone_index():
    if not PINECONE_API_KEY:
        st.error("❌ PINECONE_API_KEY not found in .env file.")
        st.stop()
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(INDEX_NAME)


def get_index_stats(index):
    try:
        stats = index.describe_index_stats()
        return stats.total_vector_count if hasattr(stats, 'total_vector_count') else stats['total_vector_count']
    except Exception:
        return None


with st.sidebar:
    st.markdown("## 🔧 About")
    st.write(
        "A modern semantic PDF search experience built with Streamlit, Sentence Transformers, and Pinecone. "
        "Upload, index, and query your documents in seconds."
    )
    st.divider()
    st.markdown("### Status")
    try:
        index = get_pinecone_index() if PINECONE_API_KEY else None
        count = get_index_stats(index) if index else None
        st.metric("Indexed chunks", count if count is not None else "-")
    except Exception:
        st.metric("Indexed chunks", "Unavailable")
    st.markdown("---")
    st.markdown("### Tips")
    st.write("• Upload at least 5 PDFs for consistent results.")
    st.write("• Use short natural language queries.")
    st.write("• Give the index a moment to update after uploading.")


with st.container():
    st.markdown(
        """
        <div class="hero">
            <h1>Sentence Predictor</h1>
            <p>Advanced PDF semantic search with beautiful cards, clear guidance, and reliable indexing.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    overview_cols = st.columns(3, gap="large")
    overview_cols[0].markdown(
        """
        <div class='mini-card'>
            <h3>Fast embeddings</h3>
            <p>Sentence-BERT powers retrieval with rich semantic understanding.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    overview_cols[1].markdown(
        """
        <div class='mini-card'>
            <h3>Search-ready</h3>
            <p>Upload PDFs, index chunks, then ask questions in natural language.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    overview_cols[2].markdown(
        """
        <div class='mini-card'>
            <h3>Beautiful UI</h3>
            <p>Polished look and responsive layout for a premium experience.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    upload_card, search_card = st.columns([2, 3], gap="large")

    with upload_card:
        st.markdown('<p class="section-label">📂 Step 1 — Upload PDFs</p>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            label="Upload PDF files",
            type="pdf",
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            file_names = [uf.name for uf in uploaded_files]
            st.write(f"**{len(uploaded_files)} files selected**")
            st.write("\n".join(f"• {name}" for name in file_names))
            if len(uploaded_files) < 5:
                st.warning("Upload at least 5 PDFs for the best results.")
            else:
                st.success("Ready to index your documents.")
        else:
            st.info("Drag and drop PDFs or click to browse.")

        st.markdown('<p class="section-label">⚙️ Step 2 — Index Documents</p>', unsafe_allow_html=True)
        if st.button("🚀 Index Documents", disabled=not uploaded_files):
            if uploaded_files:
                index = get_pinecone_index()
                progress = st.progress(0)
                total_chunks = 0
                all_vectors = []
                total_files = len(uploaded_files)

                for idx, pdf in enumerate(uploaded_files, start=1):
                    progress.progress(int((idx - 1) / total_files * 75))
                    text = extract_text_from_pdf(pdf)
                    chunks = chunk_text(text)
                    embeddings = model.encode(chunks, show_progress_bar=False)
                    for chunk_idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                        all_vectors.append({
                            "id": f"{pdf.name}__chunk_{chunk_idx}",
                            "values": emb.tolist(),
                            "metadata": {
                                "filename": pdf.name,
                                "text": chunk[:1000],
                            },
                        })
                    total_chunks += len(chunks)
                    st.write(f"Indexed **{len(chunks)}** chunks from **{pdf.name}**")

                progress.progress(80)
                for batch_start in range(0, len(all_vectors), 100):
                    index.upsert(vectors=all_vectors[batch_start:batch_start + 100])
                progress.progress(100)
                st.success(f"Indexed {total_chunks} chunks from {total_files} PDFs.")
            else:
                st.error("Please upload PDFs before indexing.")

    with search_card:
        st.markdown('<p class="section-label">🔎 Step 3 — Search</p>', unsafe_allow_html=True)
        query = st.text_input("", placeholder="Ask a question about your PDF content…", key="search_query")
        st.write("Use clear questions like: 'What is the conclusion of the third document?' or 'List the main findings.'")

        if query:
            with st.spinner("Searching for the best matches..."):
                try:
                    index = get_pinecone_index()
                    query_embedding = model.encode(query.strip()).tolist()
                    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
                except Exception as error:
                    st.error(f"Search failed: {error}")
                    results = None

            if results and getattr(results, 'matches', None):
                st.markdown(f"### 📋 Top {len(results.matches)} Matches")
                for match in results.matches:
                    filename = match.metadata.get("filename", "Unknown")
                    score = match.score or 0.0
                    snippet = match.metadata.get("text", "No preview available.")
                    st.markdown(
                        f"""
                        <div class='result-card'>
                            <div class='result-header'>
                                <span class='result-filename'>📄 {filename}</span>
                                <span class='result-score'>Score {score:.4f}</span>
                            </div>
                            <div class='result-text'>{snippet}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            elif results is not None:
                st.info("No semantic matches found. Try a different query or add more PDFs.")


        unsafe_allow_html=True,
    )
