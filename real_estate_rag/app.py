import os
import streamlit as st
from dotenv import load_dotenv
from rag_chain import build_chain, answer_query

load_dotenv()

st.set_page_config(page_title="Real Estate RAG Search", page_icon="🏠", layout="wide")

st.title("🏠 Real Estate RAG Search")
st.caption(
    "Ask in plain English — e.g. *\"Show me modern properties with large balconies "
    "in Pune under 1.5 crore\"* — and the assistant will search and summarize matching listings."
)

# --- Sidebar: API key + settings ---
with st.sidebar:
    st.header("Settings")
    default_key = os.getenv("GROQ_API_KEY", "")
    groq_api_key = st.text_input(
        "Groq API Key",
        value=default_key,
        type="password",
        help="Get a free key at https://console.groq.com",
    )
    top_k = st.slider("Number of listings to retrieve", min_value=2, max_value=10, value=5)
    st.divider()
    st.markdown(
        "**Pipeline:** query → HuggingFace embedding → FAISS similarity search "
        "→ Groq LLM summarizes top matches."
    )

if not groq_api_key:
    st.info("👈 Enter your free Groq API key in the sidebar to get started.")
    st.stop()

if not os.path.exists("faiss_index"):
    st.error(
        "No FAISS index found. Run `python ingest.py` first to build the vector "
        "store from data/properties.csv."
    )
    st.stop()

# --- Build (and cache) the retriever + chain ---
@st.cache_resource(show_spinner="Loading vector store and model...")
def get_chain(api_key: str, k: int):
    return build_chain(api_key, top_k=k)

retriever, chain = get_chain(groq_api_key, top_k)

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
user_query = st.chat_input("Describe the property you're looking for...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching listings..."):
            answer, docs = answer_query(retriever, chain, user_query)
        st.markdown(answer)

        with st.expander(f"📄 View {len(docs)} raw retrieved listings"):
            for doc in docs:
                meta = doc.metadata
                st.markdown(
                    f"**{meta['title']}** — {meta['city']} — "
                    f"₹{meta['price_inr']:,} — {meta['bhk']} BHK, {meta['area_sqft']} sqft"
                )
                st.caption(doc.page_content.split("Description: ")[-1])
                st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})
