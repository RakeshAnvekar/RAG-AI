import streamlit as st
from colorama import Fore, Style
from dotenv import load_dotenv
from src.pdf_processor import process_all_pdfs
from src.text_splitter import split_documents
from src.embedding_manager import EmbeddingManager
from src.vector_store import VectorStore
from src.rag_retriever import RAGRetriever

# Load environment variables
load_dotenv()

# Page setup
st.set_page_config(page_title="SR Solutions - PDF Q&A", page_icon="📄", layout="wide")
st.title("📄 SR Solutions - PDF Question Answering System")

# ---------------- Step 1: Load PDFs ----------------
with st.spinner("Loading PDFs..."):
    all_pdf_documents = process_all_pdfs("Data/")
    chunks = split_documents(all_pdf_documents)

# ---------------- Step 2: Setup Embedding & Vector Store ----------------
embedding_manager = EmbeddingManager()
texts = [chunk.page_content for chunk in chunks]

if not texts:
    st.warning("⚠️ No text found in PDFs. Please add PDF files in the 'Data/' folder.")
else:
    with st.spinner("Generating embeddings..."):
        embeddings = embedding_manager.generate_embeddings(texts)

    vector_store = VectorStore()
    vector_store.add_documents(chunks, embeddings)

    rag_retriever = RAGRetriever(vector_store, embedding_manager)

    # ---------------- Step 3: User Input ----------------
    query = st.text_input("🔍 Ask your question about the PDFs:", "")

    if st.button("Search"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving information..."):
                results = rag_retriever.retrieve(query)

            if results:
                st.subheader("📚 Results:")
                for doc in results:
                    st.markdown(f"**Answer Preview:** {doc['content'][:300]}...")
                    st.markdown(f"**Source:** `{doc['metadata'].get('source_file', 'unknown')}`")
                    st.markdown(f"**Similarity Score:** {doc['similarity_score']:.4f}")
                    st.markdown("---")
            else:
                st.info("No relevant results found.")
