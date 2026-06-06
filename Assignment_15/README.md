# 🔍 Mini Search Engine

Semantic PDF search using **Groq AI (LLaMA 3)** + **Pinecone** + **free local embeddings**.

## Quick Setup

```bash
git clone <your-repo-url>
cd mini-search-engine
pip install -r requirements.txt
streamlit run app.py
```

## Keys You Need

| Key | Where to get it | Free? |
|-----|----------------|-------|
| Groq API Key | console.groq.com/keys | ✅ Yes |
| Pinecone API Key | app.pinecone.io → API Keys | ✅ Yes |
| OpenAI Key | ❌ Not needed | — |

## How to Use

1. Paste Groq + Pinecone keys in the sidebar
2. Type any index name e.g. `mini-search-engine`
3. Upload 5+ PDFs → click **Index Documents**
4. Type a question → click **Search**
5. Get AI answer + matching chunks!

## Deploy Free on Streamlit Cloud

1. Push to GitHub
2. Go to share.streamlit.io → New app
3. Select your repo, set main file = `app.py`
4. Deploy!
