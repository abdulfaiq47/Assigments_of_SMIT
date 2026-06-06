from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"   # free, fast, 384-dim, no API key needed
_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_embedding(text: str) -> list[float]:
    """Return a single embedding vector (no API key required)."""
    model = _get_model()
    return model.encode(text, normalize_embeddings=True).tolist()


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Return embeddings for a list of texts in one pass."""
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True, batch_size=64).tolist()
