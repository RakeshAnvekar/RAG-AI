from langchain_text_splitters import RecursiveCharacterTextSplitter #splits long documents into smaller overlapping chunks for RAG

def split_documents(documents, chunk_size=1000, chunk_overlap=200):#Each chunk is around 1000 characters, overlapping by 200.
    """
    Split documents into smaller chunks for RAG.
    Each chunk has overlap to preserve context between chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""] #break text at paragraph, line, or space
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

    if split_docs:
        print(f"\nSample chunk content: {split_docs[0].page_content[:200]}...")
        print(f"Sample metadata: {split_docs[0].metadata}")

    return split_docs

# if PDF has 1 page, and that page has 5000 characters

#Chunk  Start index  End index  Characters Covered   Overlap with prev

# 1	    0	        1000	     1000 chars	—
# 2	    800	        1800	     1000 chars	        200 chars overlap
# 3	    1600	    2600	     1000 chars	        200 chars overlap

# why overlap
# Example without overlap:
# Chunk 1: "The company's revenue increased by 20% last year. The main reason"
# Chunk 2: "for this growth was the introduction of a new AI-powered product..."
# When the LLM or vector search looks at one chunk alone, the context is lost — it doesn’t understand the full sentence or meaning.

#With 200-character overlap:
# Chunk 1: "...The company's revenue increased by 20% last year. The main reason"
# Chunk 2: "The main reason for this growth was the introduction of a new AI-powered product..."
# Now both chunks contain the complete context
