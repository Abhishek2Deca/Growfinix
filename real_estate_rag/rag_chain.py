import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Groq model: fast + free tier. Swap for another Groq-hosted model if you like.
GROQ_MODEL = "llama-3.3-70b-versatile"

PROMPT_TEMPLATE = """You are a helpful real estate assistant. A user is searching for a property.
Using ONLY the listings provided below, recommend the most relevant ones and explain briefly why
each matches the user's request. If nothing matches well, say so honestly instead of making things up.
Do not invent listings, prices, or features that are not in the context below.

User query: {question}

Relevant listings:
{context}

Respond with:
1. A short 1-2 sentence summary of what you found.
2. A bulleted list of the best-matching listings (title, city, price, BHK, and a one-line reason it fits).
"""


def _load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(
        INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )


def _format_docs(docs) -> str:
    """Turn retrieved Documents into a plain-text block for the prompt."""
    blocks = []
    for doc in docs:
        blocks.append(doc.page_content)
    return "\n\n---\n\n".join(blocks)


def build_chain(groq_api_key: str, top_k: int = 5):
    """Builds the retriever + LLM chain. Call once per session and reuse."""
    vectorstore = _load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=groq_api_key,
        temperature=0.2,
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = prompt | llm | StrOutputParser()

    return retriever, chain


def answer_query(retriever, chain, question: str):
    """Runs retrieval + generation for a single user question.

    Returns a tuple: (answer_text, retrieved_documents) so the UI can
    show both the LLM's summary and the raw matched listings.
    """
    docs = retriever.invoke(question)
    context = _format_docs(docs)
    answer = chain.invoke({"question": question, "context": context})
    return answer, docs
