"""Text Representation and Dense Embedding Module.

Supports Sentence-BERT, ClinicalBERT, and resilient TF-IDF / SVD dense encoders
for encoding individual posts and longitudinal timeline sequence tensors.
"""

from typing import List, Optional, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class LongitudinalTextEncoder:
    """Encodes patient posts into dense semantic feature representations."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        fallback_dim: int = 64,
        max_tfidf_features: int = 500,
        use_transformer: bool = False
    ):
        self.model_name = model_name
        self.fallback_dim = fallback_dim
        self.max_tfidf_features = max_tfidf_features
        self.use_transformer = use_transformer
        self.dense_model = None
        self.tfidf_vectorizer = None
        self.svd_reducer = None
        self.embedding_dim = fallback_dim

        self._init_encoder()

    def _init_encoder(self):
        """Attempts to load SentenceTransformer if requested, otherwise uses lightweight TF-IDF + SVD."""
        if self.use_transformer and self.model_name:
            try:
                from sentence_transformers import SentenceTransformer
                self.dense_model = SentenceTransformer(self.model_name)
                self.embedding_dim = self.dense_model.get_sentence_embedding_dimension()
                print(f"[TextEncoder] Successfully loaded Transformer model: {self.model_name} (dim={self.embedding_dim})")
                return
            except Exception as e:
                print(f"[TextEncoder] Transformer loading failed ({e}). Falling back to fast TF-IDF.")

        # Lightweight fast TF-IDF + SVD encoder (memory footprint < 10MB)
        self.dense_model = None
        self.embedding_dim = self.fallback_dim
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_tfidf_features,
            stop_words="english",
            ngram_range=(1, 2)
        )
        # Pre-seed with clinical domain terms so vectorizer is immediately usable
        seed_corpus = [
            "fatigue joint pain doctor dismissed anxiety malar rash neuropathy brain fog inflammation stiffness",
            "positive ana bloodwork rheumatologist referral diagnostic delay flare autoimmune symptoms fever",
            "normal labs crying after appointment specialist connective tissue disease ssa antibodies clinical validation"
        ]
        tfidf_mat = self.tfidf_vectorizer.fit_transform(seed_corpus)
        n_comp = min(self.fallback_dim, tfidf_mat.shape[1] - 1 if tfidf_mat.shape[1] > 1 else 1)
        self.svd_reducer = TruncatedSVD(n_components=n_comp, random_state=42)
        self.svd_reducer.fit(tfidf_mat)

    def fit(self, all_corpus_texts: List[str]):
        """Fits TF-IDF and SVD if transformer is not present."""
        if self.dense_model is None and self.tfidf_vectorizer is not None:
            valid_texts = [t for t in all_corpus_texts if t and isinstance(t, str)]
            if not valid_texts:
                valid_texts = ["sample text baseline placeholder"]
            tfidf_mat = self.tfidf_vectorizer.fit_transform(valid_texts)
            n_comp = min(self.fallback_dim, tfidf_mat.shape[1] - 1 if tfidf_mat.shape[1] > 1 else 1)
            self.svd_reducer = TruncatedSVD(n_components=n_comp, random_state=42)
            self.svd_reducer.fit(tfidf_mat)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encodes a list of text strings into an array of shape (N, fallback_dim)."""
        clean_texts = [t if (t and isinstance(t, str)) else " " for t in texts]
        
        if self.dense_model is not None:
            embeddings = self.dense_model.encode(clean_texts, show_progress_bar=False, convert_to_numpy=True)
            return embeddings.astype(np.float32)
        else:
            tfidf_mat = self.tfidf_vectorizer.transform(clean_texts)
            reduced = self.svd_reducer.transform(tfidf_mat)
            
            # Ensure output is always of shape (N, fallback_dim)
            if reduced.shape[1] < self.fallback_dim:
                pad_width = ((0, 0), (0, self.fallback_dim - reduced.shape[1]))
                dense_mat = np.pad(reduced, pad_width, mode='constant')
            else:
                dense_mat = reduced[:, :self.fallback_dim]
            return dense_mat.astype(np.float32)

    def encode_timeline_sequence(
        self,
        timeline_texts: List[str],
        max_seq_len: int = 15
    ) -> np.ndarray:
        """Encodes a sequence of posts into a tensor of shape (max_seq_len, embedding_dim).
        Pads with zeros if length < max_seq_len.
        """
        if not timeline_texts:
            return np.zeros((max_seq_len, self.embedding_dim), dtype=np.float32)

        # Truncate if exceeds max_seq_len
        truncated = timeline_texts[-max_seq_len:]
        post_embeddings = self.encode_texts(truncated)

        seq_tensor = np.zeros((max_seq_len, self.embedding_dim), dtype=np.float32)
        actual_len = len(post_embeddings)
        seq_tensor[-actual_len:] = post_embeddings  # Left pad or right align

        return seq_tensor
