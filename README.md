# NCERT PDF Uploader & Chapter Summarizer

This project is a Streamlit web application that allows you to:
- Download or upload an NCERT PDF.
- Extract and chunk the text from the PDF.
- Generate vector embeddings for the text and store them in a Qdrant vector database.
- Use a local LLM (HuggingFace Transformers) to summarize chapters from the book based on semantic search.

---

## Features & Workflow

1. **Download or Upload PDF**
   - Enter an NCERT PDF URL to download, or upload a PDF file directly.

2. **Text Extraction**
   - The app extracts text from each page of the PDF using `pdfplumber`.
   - Warnings and extraction errors are suppressed/skipped for robustness.

3. **Text Chunking**
   - The extracted text is split into chunks (default 500 characters) for embedding and storage.

4. **Embeddings**
   - Each chunk is embedded using the `sentence-transformers` library (MiniLM model, CPU enforced).

5. **Vector Storage (Qdrant)**
   - Embeddings and their corresponding text chunks are stored in a Qdrant Cloud collection.
   - If the collection does not exist, it is created automatically.

6. **Semantic Search**
   - When a user enters a chapter name/number, the app embeds the query and searches Qdrant for the most relevant chunks.

7. **Summarization**
   - The retrieved text is summarized using HuggingFace's `facebook/bart-large-cnn` model.
   - The summary is chunked and concatenated to meet the desired word count.
   - NLTK is used for sentence tokenization, with automatic resource download if needed.

---

## Main Libraries Used

- **streamlit**: For building the web UI.
- **pdfplumber**: For extracting text from PDF files.
- **requests**: For downloading PDFs from URLs.
- **qdrant-client**: For interacting with the Qdrant vector database.
- **sentence-transformers**: For generating vector embeddings from text.
- **transformers**: For local LLM summarization (HuggingFace pipeline).
- **nltk**: For sentence tokenization (used in summarization chunking).
- **warnings**: For suppressing PDF extraction warnings.

---

## Key Functions Explained

- `download_ncert_pdf(url, save_path)`: Downloads a PDF from a given URL and saves it locally.
- `extract_text_from_pdf(pdf_path)`: Extracts text from each page of the PDF, skipping pages with errors.
- `chunk_text(text, chunk_size)`: Splits the extracted text into manageable chunks for embedding.
- `get_embeddings(text_chunks)`: Generates vector embeddings for each chunk using MiniLM (CPU enforced).
- `store_in_qdrant(chunks, embeddings)`: Stores the embeddings and their text in a Qdrant collection, creating it if needed.
- `search_qdrant(query)`: Embeds the user's query and retrieves the most relevant text chunks from Qdrant.
- `summarize_text(texts, max_length, min_length)`: Summarizes the retrieved text using a local LLM, chunking as needed, and ensures the summary meets the desired word count.

---

## How to Run

1. **Install dependencies:**
   ```sh
   pip install streamlit pdfplumber requests qdrant-client sentence-transformers transformers nltk
   ```

2. **Run the app:**
   ```sh
   streamlit run app.py
   ```

3. **Use the UI:**
   - Download or upload a PDF.
   - Click "Store in Vector DB (Qdrant)".
   - Enter a chapter name/number and click "Describe Chapter".

---

## Notes
- The summarization model cannot generate extremely long summaries in a single call; the code chunks and concatenates summaries to approach the desired word count.
- NLTK resources are downloaded automatically if missing.
- All computation is done locally except for vector storage/search (Qdrant Cloud).

---

## Troubleshooting
- If you see NLTK resource errors, ensure your internet connection is active for the first run.
- If you see errors about PyTorch or transformers, ensure all dependencies are up to date.
- For Qdrant errors, check your API key and collection name.

---

## Security
- Do not share your Qdrant API key publicly.
- Uploaded PDFs are saved locally and not shared.

---

## License
This project is for educational/demo purposes. Adapt as needed for your use case.

