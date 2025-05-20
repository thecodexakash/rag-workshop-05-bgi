from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    return chunks

# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

def create_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings


if __name__ == "__main__":
    pdf_file = "lorem_ipsum.pdf"

    print("Extracting text...")
    raw_text = get_pdf_text(pdf_file)

    print("Splitting text...")
    chunks = get_text_chunks(raw_text)

    print("Creating embeddings...")
    embeddings = create_embeddings(chunks)

    print(f"\nNumber of chunks: {len(chunks)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Text: {chunk[:200]}{'...' if len(chunk) > 200 else ''}")
        print(f"Embedding (first 20 dims): {embedding[:20]}")
