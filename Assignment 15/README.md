# 🔍 Mini Search Engine

Semantic PDF search app built with **Streamlit**, **OpenAI Embeddings**, and **Pinecone**.

## Quick Setup

### 1. Clone & Install
```bash
git clone <your-repo-url>
cd mini-search-engine
pip install -r requirements.txt
```

### 2. Run Locally
```bash
streamlit run app.py
```

### 3. Open in Browser
Go to `http://localhost:8501`

---

## How to Use

1. Enter your **OpenAI API Key** in the sidebar
2. Enter your **Pinecone API Key** and **Index Name** in the sidebar
3. Upload **5+ PDF files**
4. Click **⚡ Index Documents**
5. Type a natural language query and click **🔎 Search**

---

## Getting API Keys

### OpenAI
1. Go to https://platform.openai.com/api-keys
2. Click **Create new secret key**
3. Copy it — you won't see it again!

### Pinecone
1. Go to https://app.pinecone.io
2. Sign up (free)
3. Go to **API Keys** → Copy your key
4. Go to **Indexes** → You can let the app create one, just pick a name like `mini-search-engine`

---

## Deploy on Streamlit Cloud (Free)

1. Push your code to GitHub
2. Go to https://share.streamlit.io
3. Click **New app** → Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy**

---

## Project Structure

```
mini-search-engine/
├── app.py                  # Main Streamlit app
├── requirements.txt        # Dependencies
├── README.md
└── utils/
    ├── __init__.py
    ├── pdf_processor.py    # PDF text extraction & chunking
    ├── embeddings.py       # OpenAI embedding calls
    └── pinecone_client.py  # Pinecone init, upsert, query
```
