import hashlib
import math
import numpy as np

# Pure Python embeddings - no torch, no sentence-transformers
# Uses TF-IDF style bag-of-words with hashing trick
# Dimension must match PINECONE_DIMENSION in pinecone_client.py

DIMENSION = 512


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = []
    word = ""
    for ch in text:
        if ch.isalnum():
            word += ch
        else:
            if word:
                tokens.append(word)
            word = ""
    if word:
        tokens.append(word)
    return tokens


def _hash_token(token: str, dim: int) -> int:
    h = int(hashlib.md5(token.encode()).hexdigest(), 16)
    return h % dim


def _sign_token(token: str) -> float:
    h = int(hashlib.sha1(token.encode()).hexdigest(), 16)
    return 1.0 if h % 2 == 0 else -1.0


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


def get_embedding(text: str) -> list[float]:
    """Hashing trick TF-IDF embedding — no external model needed."""
    tokens = _tokenize(text)
    vec = [0.0] * DIMENSION

    # bigrams + unigrams
    ngrams = tokens + [tokens[i] + "_" + tokens[i+1] for i in range(len(tokens)-1)]

    counts: dict[str, int] = {}
    for t in ngrams:
        counts[t] = counts.get(t, 0) + 1

    total = max(len(ngrams), 1)
    for token, count in counts.items():
        tf = count / total
        idx = _hash_token(token, DIMENSION)
        sign = _sign_token(token)
        vec[idx] += sign * tf

    return _normalize(vec)


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    return [get_embedding(t) for t in texts]
