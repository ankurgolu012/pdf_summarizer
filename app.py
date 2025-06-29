import streamlit as st
import pdfplumber
import requests
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import warnings

# --- CONFIG ---
QDRANT_URL = "https://393fd4ee-bcbb-4ed7-a045-eb8445968240.us-east-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.FrDxYmR__uxjO74Qjc3LuyJ7fOLf4ddHJXrJ1rPUkgk"
QDRANT_COLLECTION = "ncert_books"

# --- PDF DOWNLOAD ---
def download_ncert_pdf(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)

# --- PDF TEXT EXTRACTION ---
def extract_text_from_pdf(pdf_path):
    import warnings
    import pdfplumber
    try:
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception:
                        continue
        except Exception:
            st.error("PDF extraction failed. Please upload a valid PDF file.")
            return ""
        return text
    except Exception:
        st.error("PDF extraction failed. Please upload a valid PDF file.")
        return ""

# --- TEXT CHUNKING ---
def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# --- EMBEDDING (Sentence Transformers) ---
def get_embeddings(text_chunks):
    import torch
    torch_device = "cpu"
    model = SentenceTransformer('all-MiniLM-L6-v2', device=torch_device)
    embeddings = model.encode(text_chunks).tolist()
    return embeddings

# --- QDRANT UPLOAD ---
def store_in_qdrant(chunks, embeddings):
    client = QdrantClient(QDRANT_URL, api_key=QDRANT_API_KEY)
    if QDRANT_COLLECTION not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qdrant_models.VectorParams(size=len(embeddings[0]), distance="Cosine")
        )
    points = [qdrant_models.PointStruct(id=i, vector=embeddings[i], payload={"text": chunks[i]}) for i in range(len(chunks))]
    client.upsert(collection_name=QDRANT_COLLECTION, points=points)

# --- QDRANT SEARCH ---
def search_qdrant(query):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_emb = model.encode([query])[0].tolist()
    client = QdrantClient(QDRANT_URL, api_key=QDRANT_API_KEY)
    hits = client.search(collection_name=QDRANT_COLLECTION, query_vector=query_emb, limit=5)
    return [hit.payload['text'] for hit in hits]

# --- LLM SUMMARIZATION (HuggingFace Transformers) ---
def summarize_text(texts, max_length=20000, min_length=30):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    joined = " ".join(texts)
    if not joined.strip():
        return "No relevant text found for this chapter. Try a different query."
    try:
        # HuggingFace models have a much lower max_length limit (usually 1024 tokens)
        # We'll chunk the input and concatenate summaries to reach the desired word count
        import math
        import nltk
        # Download all punkt resources if missing
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(joined)
        chunk_size = 900  # tokens/words per chunk (model limit)
        chunks = [" ".join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]
        summaries = []
        for chunk in chunks:
            summary = summarizer(chunk, max_length=1024, min_length=200, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        full_summary = " ".join(summaries)
        # Ensure the summary is within the required word range
        words = full_summary.split()
        if len(words) < min_length:
            return f"Summary too short ({len(words)} words). Try a different chapter or a larger PDF."
        if len(words) > max_length:
            full_summary = " ".join(words[:max_length])
        return full_summary + f"\n\n[Word count: {len(full_summary.split())}]"
    except Exception as e:
        return f"Summarization failed: {e}"

# --- STREAMLIT UI ---
st.title("NCERT PDF Uploader & Chapter Summarizer (LLM)")

# 1. Download NCERT PDF
ncert_url = st.text_input("Enter NCERT PDF URL to download:")
if st.button("Download NCERT PDF") and ncert_url:
    download_ncert_pdf(ncert_url, "ncert.pdf")
    st.success("Downloaded and saved as ncert.pdf")

# 2. Upload PDF
uploaded_file = st.file_uploader("Or upload an NCERT PDF", type="pdf")
pdf_path = None
if uploaded_file:
    with open("uploaded_ncert.pdf", "wb") as f:
        f.write(uploaded_file.read())
    pdf_path = "uploaded_ncert.pdf"
    st.success("PDF uploaded and saved as uploaded_ncert.pdf")
elif os.path.exists("ncert.pdf"):
    pdf_path = "ncert.pdf"

# 3. Extract, chunk, embed, and store in Qdrant
if pdf_path:
    text = extract_text_from_pdf(pdf_path)
    st.subheader("Extracted Text Preview (first 1000 chars):")
    st.write(text[:1000])
    store_ready = False
    if st.button("Store in Vector DB (Qdrant)"):
        chunks = chunk_text(text)
        embeddings = get_embeddings(chunks)
        store_in_qdrant(chunks, embeddings)
        st.success("Stored in Qdrant Cloud!")
        st.session_state['qdrant_ready'] = True
    # Set flag if already stored in this session
    if 'qdrant_ready' in st.session_state and st.session_state['qdrant_ready']:
        store_ready = True
else:
    store_ready = False

# 4. LLM Chapter Description
st.header("Describe a Chapter with LLM")
if not ('qdrant_ready' in st.session_state and st.session_state['qdrant_ready']):
    st.info("Please upload a PDF and click 'Store in Vector DB (Qdrant)' before using this feature.")
else:
    chapter_query = st.text_input("Enter chapter name or number or words to summarize:")
    if st.button("Describe Chapter") and chapter_query:
        chapter_texts = search_qdrant(chapter_query)
        summary = summarize_text(chapter_texts, max_length=20000, min_length=30)
        st.subheader("Chapter Summary:")
        st.write(summary)
