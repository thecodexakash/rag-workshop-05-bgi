from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone,ServerlessSpec
import hashlib
pinecone = Pinecone(api_key="",environment="us-east-1")
index_name = "bgi-rocks"
existing_indexes = pinecone.list_indexes()
print(f"Existing indexes: {existing_indexes}")
index_names = [idx['name'] for idx in existing_indexes] 
if index_name not in index_names:
    print(f"Creating index: {index_name}")
    pinecone.create_index(index_name, dimension=384, spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ))
else:
    print(f"Index '{index_name}' already exists.")

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

def hash_text(text):
    print("Text Digest",hashlib.md5(text.encode('utf-8')).hexdigest())
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def create_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return model, embeddings

def upload_to_pinecone(index, chunks, embeddings):
    existing_ids = set()
    index_stats = index.describe_index_stats()

    if 'namespaces' in index_stats and '' in index_stats['namespaces']:
        chunk_ids_to_check = [f"chunk-{hash_text(chunk)}" for chunk in chunks]
        response = index.fetch(ids=chunk_ids_to_check)
        existing_ids.update(response.vectors.keys())

    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        chunk_hash = hash_text(chunk)
        vector_id = f"chunk-{chunk_hash}"

        if vector_id not in existing_ids:
            vector = {
                "id": vector_id,
                "values": embedding.tolist(),
                "metadata": {"text": chunk}
            }
            vectors.append(vector)

    if vectors:
        index.upsert(vectors)
        print(f"Uploaded {len(vectors)} new chunks.")
    else:
        print("No new chunks to upload.")


def query_pinecone(index, query, model, top_k=5):
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )
    return results


if __name__ == "__main__":
    import os

    pdf_file = "lorem_ipsum.pdf"

    print("Extracting text...")
    raw_text = get_pdf_text(pdf_file)

    print("Splitting text...")
    chunks = get_text_chunks(raw_text)

    print("Creating embeddings...")
    model, embeddings = create_embeddings(chunks)

    print("Initializing Pinecone...")

    index = pinecone.Index(index_name)

    print("Uploading to Pinecone...")
    upload_to_pinecone(index, chunks, embeddings)

    print("Upload complete!")

    sample_query = "What is Lorem Ipsum?"
    print(f"\nQuerying: {sample_query}")
    results = query_pinecone(index, sample_query, model)

    for match in results['matches']:
        print(f"\nScore: {match['score']:.4f}")
        print(f"Text: {match['metadata']['text'][:200]}{'...' if len(match['metadata']['text']) > 200 else ''}")