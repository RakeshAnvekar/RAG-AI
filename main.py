from src.pdf_processor import process_all_pdfs
from src.text_splitter import split_documents
from src.embedding_manager import EmbeddingManager
from src.vector_store import VectorStore
from src.rag_retriever import RAGRetriever

from colorama import Fore, Style
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------------- Step 1: Load PDFs ----------------
all_pdf_documents = process_all_pdfs("Data/")
#all_pdf_documents contain all the pdf documents

# ---------------- Step 2: Split PDFs ----------------
chunks = split_documents(all_pdf_documents)

# ---------------- Step 3: Generate Embeddings ----------------
embedding_manager = EmbeddingManager()
texts = [chunk.page_content for chunk in chunks]

#chunks variable holds the list of text chunks extracted from your PDFs.
# Each chunk is usually a LangChain Document object that looks like this

#   Document(
    #page_content="This is some paragraph text from the PDF.",
    #metadata={"page": 2, "source": "C:/files/abc.pdf"}
#    )


embeddings = embedding_manager.generate_embeddings(texts)

# ---------------- Step 4: Add to Vector Store ----------------
vector_store = VectorStore()
vector_store.add_documents(chunks, embeddings)

# ---------------- Step 5: RAG Retrieval ----------------
rag_retriever = RAGRetriever(vector_store, embedding_manager)
query = "What is the vision and mission of Anvekers.com?"
results = rag_retriever.retrieve(query) #this is the one that finds relevant chunks

# ---------------- Step 6: Print in colors ----------------
for doc in results:
    print(Fore.GREEN + "Answer Preview: " + Style.RESET_ALL + doc['content'][:200] + "...")
    print(Fore.BLUE + "Source: " + Style.RESET_ALL + doc['metadata'].get('source_file','unknown'))
    print(Fore.YELLOW + "Similarity Score: " + Style.RESET_ALL + str(doc['similarity_score']))
    print("="*80)
