import os
import uuid
import numpy as np
import chromadb
from typing import List, Any

class VectorStore:
    """Saves document embeddings in a ChromaDB vector store."""

    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "Data/vector_store"):
               # pdf_documents is like table name of the database,
               # persist_directory → where the database files are stored on disk
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):#to actually connect to the ChromaDB instance
        """Initialize ChromaDB client and collection."""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)# Creates the folder (if not already there)
            #Initializes a persistent client that stores data on disk (not just in memory).So your embeddings remain available even after restarting your app.
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            #Either gets an existing collection or creates a new.
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings for RAG"}
            )
            print(f"Vector store initialized | Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise
    #This is the main function that inserts new data into ChromaDB.
    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
         #documents: the list of text chunks (each Document from LangChain).
         #embeddings: corresponding numeric vectors for each chunk.
        """
        Add documents and their embeddings to the vector store.

        Args:
            documents: List of text chunks (each doc must have .page_content and .metadata)
            embeddings: Corresponding numeric vectors (numpy array)
        """
        if not documents or len(documents) == 0:
            print("No documents to add, skipping.")
            return

        if embeddings is None or len(embeddings) == 0:
            print("No embeddings provided, skipping.")
            return

        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        print(f"Adding {len(documents)} documents to vector store...")

        ids, metadatas, docs_text, embeddings_list = [], [], [], []

        # ids: unique document identifiers
        # metadatas: metadata for each document (page, chunk index, etc.)
        # documents_text: actual text content
        # embeddings_list: corresponding embedding vectors


        ###################################
        #[
            # Document(page_content="AI is transforming industries.", metadata={"page": 1}),
            # Document(page_content="Machine learning improves automation.", metadata={"page": 2}),
        #]

        #[
            # [0.123, -0.542, 0.768, ...],
            # [0.234, -0.678, 0.321, ...],
        #]

        # goal in this loop is to package each document and its embedding together in a structured way so 
        # they can be stored in your Chroma vector database.

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            metadata = dict(getattr(doc, "metadata", {}))
            metadata["doc_index"] = i
            metadata["content_length"] = len(getattr(doc, "page_content", ""))
            metadatas.append(metadata)
            docs_text.append(getattr(doc, "page_content", ""))
            embeddings_list.append(embedding.tolist())

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=docs_text
            )
            print(f"Successfully added {len(documents)} documents")
            print(f"Total documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise
