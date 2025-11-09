import numpy as np
from sentence_transformers import SentenceTransformer #loads a transformer model (like all-MiniLM-L6-v2) for generating embeddings.
from typing import List

class EmbeddingManager:
    """
    Handles document embedding generation using SentenceTransformer.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the SentenceTransformer model."""
        try:
            print(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded! Dimension: {self.model.get_sentence_embedding_dimension()}")

        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not self.model:
            raise ValueError("Model not loaded")
        if not texts:
            raise ValueError("No texts provided for embeddings!")

        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)

        #"AI is amazing" → ["AI", "is", "am", "##az", "##ing"]
        #[0.123, -0.542, 0.768, ... , 0.005]

        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
