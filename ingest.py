import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

DATA_PATH = "data/properties.csv"
INDEX_PATH = "faiss_index"

# Free, local embedding model — no API key needed, runs on CPU.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_properties_as_documents(csv_path: str) -> list[Document]:
    """Read the properties CSV and turn each row into a LangChain Document.

    Each Document's `page_content` is what actually gets embedded, so we
    combine the descriptive text with the structured facts (city, price,
    bhk, area) in natural language. That way semantic search can match
    both on vibe ("modern", "large balcony") and on hard facts.
    Metadata keeps the structured fields for filtering/display later.
    """
    df = pd.read_csv(csv_path)
    documents = []

    for _, row in df.iterrows():
        content = (
            f"Title: {row['title']}\n"
            f"City: {row['city']}\n"
            f"Price: INR {row['price_inr']:,}\n"
            f"Configuration: {row['bhk']} BHK, {row['area_sqft']} sqft\n"
            f"Description: {row['description']}"
        )
        metadata = {
            "id": int(row["id"]),
            "title": row["title"],
            "city": row["city"],
            "price_inr": int(row["price_inr"]),
            "bhk": int(row["bhk"]),
            "area_sqft": int(row["area_sqft"]),
        }
        documents.append(Document(page_content=content, metadata=metadata))

    return documents


def build_and_save_index():
    print(f"Loading properties from {DATA_PATH} ...")
    documents = load_properties_as_documents(DATA_PATH)
    print(f"Loaded {len(documents)} property listings.")

    print(f"Loading embedding model '{EMBEDDING_MODEL}' (first run downloads it) ...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Embedding listings and building FAISS index ...")
    vectorstore = FAISS.from_documents(documents, embeddings)

    vectorstore.save_local(INDEX_PATH)
    print(f"FAISS index saved to ./{INDEX_PATH}/")


if __name__ == "__main__":
    build_and_save_index()
