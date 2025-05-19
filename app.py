from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

if __name__ == "__main__":

    pdf_file = "lorem_ipsum.pdf"

    print("Extracting text from PDF...")
    raw_text = get_pdf_text(pdf_file)

    print("Splitting text into chunks...")
    chunks = get_text_chunks(raw_text)

    print(f"Total chunks created: {len(chunks)}\n")

    for i, chunk in enumerate(chunks[:5]):
        print(f"--- Chunk {i+1} ---")
        print(chunk)
