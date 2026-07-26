# 🏠 Real Estate RAG Search System

An intelligent search backend for property listings. It embeds long property
descriptions into a vector database (FAISS), and lets you ask natural-language
questions like *"Show me modern properties with large balconies"* — an LLM
then retrieves and summarizes the most relevant listings.

## Architecture

```
data/properties.csv          Raw property listings (title, city, price, description...)
        │
        ▼
   ingest.py                 1. Turns each listing into a Document
        │                    2. Embeds it with a free local HuggingFace model
        │                       (sentence-transformers/all-MiniLM-L6-v2)
        │                    3. Stores all vectors in a local FAISS index
        ▼
   faiss_index/               (created after you run ingest.py)
        │
        ▼
  rag_chain.py               1. Embeds the user's query the same way
        │                    2. FAISS retrieves the top-k most similar listings
        │                    3. Groq LLM (llama-3.3-70b) reads those listings
        │                       and writes a summarized recommendation
        ▼
     app.py                  Streamlit chat UI wrapping the whole pipeline
```

**Why this stack:**
- **HuggingFace embeddings** — free, runs locally, no API key or cost for the
  embedding step (only the LLM call uses an API).
- **FAISS** — fast, lightweight, in-memory-friendly vector similarity search;
  no external database server needed.
- **Groq** — free-tier API that runs open models (Llama 3.3) extremely fast,
  good for a responsive chatbot.
- **Streamlit** — quickest way to get a real chat UI without writing frontend
  code.

## Setup

1. **Install dependencies** (Python 3.10+ recommended):
   ```bash
   pip install -r requirements.txt
   ```

2. **Get a free Groq API key**: sign up at https://console.groq.com and
   create an API key (no credit card required for the free tier).

3. **Set your API key** — either:
   - Create a `.env` file in this folder with:
     ```
     GROQ_API_KEY=your_key_here
     ```
   - Or just paste it into the sidebar text box when the app is running.

4. **Build the vector index** (run once, and again whenever
   `data/properties.csv` changes):
   ```bash
   python ingest.py
   ```
   This downloads the embedding model the first time (~90MB) and creates a
   `faiss_index/` folder — that's your vector database.

5. **Run the chatbot**:
   ```bash
   streamlit run app.py
   ```
   It'll open in your browser at `http://localhost:8501`.

## Using your own data

Replace `data/properties.csv` with your own listings. Keep the same columns
(`id, title, city, price_inr, bhk, area_sqft, description`), or adjust the
column names in `ingest.py`'s `load_properties_as_documents()` function to
match your schema. Then re-run `python ingest.py` to rebuild the index.

## Example queries to try

- "Show me modern properties with large balconies"
- "2BHK apartments in Pune under 80 lakhs"
- "Something peaceful with mountain views for a weekend home"
- "Waterfront properties with a private jetty"

## Notes for extending this project

- **Metadata filtering**: FAISS retrieval here is purely semantic. For a
  stronger real estate search, add hard filters (city, price range, BHK)
  before or after the similarity search — see `vectorstore.as_retriever()`
  in `rag_chain.py`, where you can pass a `filter` dict.
- **Chunking**: if your real descriptions get much longer (multiple
  paragraphs), consider splitting them with `RecursiveCharacterTextSplitter`
  before embedding, so each chunk stays focused and retrieval stays precise.
- **Swapping providers**: to use OpenAI instead of Groq, swap `ChatGroq` for
  `ChatOpenAI` in `rag_chain.py` — the rest of the pipeline is unchanged,
  since LangChain abstracts the LLM call.
- **ChromaDB alternative**: the task mentions ChromaDB as an alternative to
  FAISS. Swapping is straightforward — replace
  `langchain_community.vectorstores.FAISS` with
  `langchain_community.vectorstores.Chroma` and adjust the `save_local` /
  `load_local` calls to Chroma's persistence API (`persist_directory`).
