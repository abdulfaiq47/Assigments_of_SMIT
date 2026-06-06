from pinecone import Pinecone, ServerlessSpec
from utils.embeddings import get_embeddings_batch

DIMENSION = 1536          # text-embedding-3-small output size
BATCH_SIZE = 100          # upsert N vectors at a time


def init_pinecone(api_key: str, index_name: str):
    """Connect to Pinecone and create the index if it doesn't exist yet."""
    pc = Pinecone(api_key=api_key)

    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    return pc.Index(index_name)


def upsert_chunks(index, chunks: list[dict], openai_key: str) -> None:
    """
    Generate embeddings for all chunks and upsert them into Pinecone.
    chunks: list of {id, text, filename}
    """
    texts = [c["text"] for c in chunks]

    # Embed in batches to stay within API limits
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]
        batch_embeddings = get_embeddings_batch(batch_texts, openai_key)
        all_embeddings.extend(batch_embeddings)

    # Build Pinecone vectors and upsert in batches
    vectors = []
    for chunk, embedding in zip(chunks, all_embeddings):
        vectors.append({
            "id": chunk["id"],
            "values": embedding,
            "metadata": {
                "text": chunk["text"],
                "filename": chunk["filename"],
            },
        })

    for i in range(0, len(vectors), BATCH_SIZE):
        index.upsert(vectors=vectors[i : i + BATCH_SIZE])


def query_pinecone(index, query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """Query Pinecone and return top-k matches with metadata."""
    response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )
    return [
        {
            "id": match["id"],
            "score": match["score"],
            "metadata": match.get("metadata", {}),
        }
        for match in response.get("matches", [])
    ]
