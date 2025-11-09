from typing import List, Dict, Any

class RAGRetriever:
    """Retrieve relevant documents from vector store."""

    def __init__(self, vector_store, embedding_manager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        #query → the user’s question (e.g. "What is the mission of Anvekers.com?")
        #top_k → how many top similar results to fetch (default 5)
        #score_threshold → ignore results below this similarity score

       # Output → list of dictionaries with the most relevant chunks.

        print(f"Retrieving documents for query: {query}")
        
        #You convert the text query into a vector embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        results = self.vector_store.collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)
        retrieved_docs = []

        if results['documents'] and results['documents'][0]:
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            ids = results['ids'][0]

            for i, (doc_id, doc, meta, dist) in enumerate(zip(ids, documents, metadatas, distances)):
                similarity_score = 1 - dist
                if similarity_score > score_threshold:
                    retrieved_docs.append({
                        'id': doc_id,
                        'content': doc,
                        'metadata': meta,
                        'similarity_score': similarity_score,
                        'distance': dist,
                        'rank': i + 1
                    })
        else:
            print("No documents found")
        print(f"Retrieved {len(retrieved_docs)} documents after filtering")
        return retrieved_docs
