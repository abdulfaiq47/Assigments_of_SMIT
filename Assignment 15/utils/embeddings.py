from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"   # cheap & fast


def get_embedding(text: str, api_key: str) -> list[float]:
    """Return an embedding vector for a single text string."""
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text.replace("\n", " "),
    )
    return response.data[0].embedding


def get_embeddings_batch(texts: list[str], api_key: str) -> list[list[float]]:
    """Return embeddings for a list of texts in one API call (max 2048 items)."""
    client = OpenAI(api_key=api_key)
    clean = [t.replace("\n", " ") for t in texts]
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=clean,
    )
    # Preserve original order
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
