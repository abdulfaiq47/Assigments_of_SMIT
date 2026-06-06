import io
import uuid
import pypdf


def extract_text_from_pdf(pdf_file) -> str:
    """Extract all text from an uploaded PDF file object."""
    reader = pypdf.PdfReader(io.BytesIO(pdf_file.getvalue()))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()


def chunk_text(text: str, filename: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """
    Split text into overlapping chunks and attach metadata.
    Returns a list of dicts: {id, text, filename}
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        if len(chunk_text.strip()) > 20:          # skip tiny scraps
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk_text,
                "filename": filename,
            })

        start += chunk_size - overlap             # slide with overlap

    return chunks
