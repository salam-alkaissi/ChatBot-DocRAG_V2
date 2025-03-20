# src/retrieval.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class RetrievalSystem:
    def __init__(self, method="tfidf"):
        """
        Initialize retrieval system
        :param method: 'tfidf' or 'semantic'
        """
        self.method = method
        self.vectorizer = None
        self.embeddings = None
        self.corpus = []
        
        if method == "semantic":
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.model = None
            self.vectorizer = TfidfVectorizer(
                stop_words='english',
                token_pattern=r'(?u)\b[a-zA-Z-]{2,}\b'
            )

    def index_documents(self, documents):
        """Index documents for retrieval"""
        self.corpus = documents
        
        if self.method == "tfidf":
            self.vectorizer.fit(documents)
        elif self.method == "semantic":
            self.embeddings = self.model.encode(documents)
            
    def retrieve(self, query, top_k=5):
        """Retrieve relevant document chunks"""
        if not self.corpus:
            raise ValueError("No documents indexed for retrieval")
            
        if self.method == "tfidf":
            return self._tfidf_retrieval(query, top_k)
        elif self.method == "semantic":
            return self._semantic_retrieval(query, top_k)
            
    def _tfidf_retrieval(self, query, top_k):
        """TF-IDF based retrieval"""
        query_vec = self.vectorizer.transform([query])
        doc_vectors = self.vectorizer.transform(self.corpus)
        sim_scores = cosine_similarity(query_vec, doc_vectors).flatten()
        top_indices = np.argsort(sim_scores)[-top_k:][::-1]
        return [self.corpus[i] for i in top_indices]
        
    def _semantic_retrieval(self, query, top_k):
        """Semantic similarity based retrieval"""
        query_embedding = self.model.encode([query])
        sim_scores = cosine_similarity(query_embedding, self.embeddings).flatten()
        top_indices = np.argsort(sim_scores)[-top_k:][::-1]
        return [self.corpus[i] for i in top_indices]