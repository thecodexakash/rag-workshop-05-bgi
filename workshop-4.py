import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import hashlib
from together import Together

PINECONE_API_KEY = "[pinecone api key]"
TOGETHER_API_KEY = "[together api key]"
INDEX_NAME = "[your index name]"


pinecone = Pinecone(api_key=PINECONE_API_KEY, environment="us-east-1")
client = Together(api_key=TOGETHER_API_KEY)

if INDEX_NAME not in [idx['name'] for idx in pinecone.list_indexes()]:
    pinecone.create_index(INDEX_NAME, dimension=384, spec=ServerlessSpec(cloud="aws", region="us-east-1"))
index = pinecone.Index(INDEX_NAME)

def extract_text_from_pdf(uploaded_file):
    text = ""
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def split_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def hash_text(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def create_embeddings(chunks, model):
    return model.encode(chunks, show_progress_bar=True)

def store_in_pinecone(chunks, embeddings):
    existing_ids = set()
    stats = index.describe_index_stats()
    if 'namespaces' in stats and '' in stats['namespaces']:
        ids = [f"chunk-{hash_text(c)}" for c in chunks]
        response = index.fetch(ids=ids)
        existing_ids.update(response.vectors.keys())
    
    vectors = []
    for chunk, emb in zip(chunks, embeddings):
        chunk_id = f"chunk-{hash_text(chunk)}"
        if chunk_id not in existing_ids:
            vectors.append({
                "id": chunk_id,
                "values": emb.tolist(),
                "metadata": {"text": chunk}
            })
    if vectors:
        index.upsert(vectors)

def search_chunks(query, model, top_k=5):
    query_emb = model.encode([query], normalize_embeddings=True)[0]
    results = index.query(vector=query_emb.tolist(), top_k=top_k, include_metadata=True)
    return [m['metadata']['text'] for m in results['matches']]

def generate_answer(context, question):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=messages,
        stream=True
    )
    for token in response:
        if hasattr(token, 'choices'):
            yield token.choices[0].delta.content or ""


st.title("Efficient RAG Applications using Vector DBs")


mode = st.radio("Choose Mode:", ["Upload PDF & Ask", "Ask from existing index"])

model = load_embedding_model()

if mode == "Upload PDF & Ask":
    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])
    if uploaded_file:
        st.success("PDF uploaded successfully!")
        raw_text = extract_text_from_pdf(uploaded_file)
        st.write(f"Extracted ~{len(raw_text)} characters.")

        chunks = split_text(raw_text)
        embeddings = create_embeddings(chunks, model)

        with st.spinner("Storing data in Pinecone..."):
            store_in_pinecone(chunks, embeddings)
            st.success("Stored in Pinecone successfully!")

        st.divider()
        st.header("Ask a Question from the PDF")

        user_query = st.text_input("Your question")
        if user_query:
            with st.spinner("Searching relevant content..."):
                top_chunks = search_chunks(user_query, model)

            if top_chunks:
                context = "\n\n".join(top_chunks)
                st.subheader("Answer:")
                response_placeholder = st.empty()
                streamed_output = ""
                for token in generate_answer(context, user_query):
                    streamed_output += token
                    response_placeholder.markdown(streamed_output)
            else:
                st.warning("No relevant content found in the PDF.")
elif mode == "Ask from existing index":
    st.header("Ask a Question from existing Pinecone index")

    user_query = st.text_input("Your question")
    if user_query:
        with st.spinner("Searching relevant content..."):
            top_chunks = search_chunks(user_query, model)

        if top_chunks:
            context = "\n\n".join(top_chunks)
            st.subheader("Answer:")
            response_placeholder = st.empty()
            streamed_output = ""
            for token in generate_answer(context, user_query):
                streamed_output += token
                response_placeholder.markdown(streamed_output)
        else:
            st.warning("No relevant content found in the index.")