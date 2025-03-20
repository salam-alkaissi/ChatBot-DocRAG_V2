from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class NLPPipeline:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            token_pattern=r'(?u)\b[a-zA-Z-]{2,}\b'  # Improved token pattern
        )
        self.tfidf_matrix = None
        self.corpus = []

    def fit(self, corpus):
        self.corpus = corpus
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def retrieve_chunks(self, query, top_k=5):
        query_vec = self.vectorizer.transform([query])
        sim_scores = np.dot(query_vec, self.tfidf_matrix.T).toarray()[0]
        top_indices = sim_scores.argsort()[-top_k:][::-1]
        return [self.corpus[i] for i in top_indices]

    def extract_keywords(self, text, top_n=10):
        """Return keywords with actual counts from text"""
        # Tokenize using the same pattern as vectorizer
        tokenizer = self.vectorizer.build_tokenizer()
        tokens = tokenizer(text.lower())  # Match vectorizer's lowercase
        
        # Count valid tokens that exist in vocabulary
        vocab = self.vectorizer.vocabulary_
        valid_tokens = [t for t in tokens if t in vocab]
        
        # Get frequency counts
        unique, counts = np.unique(valid_tokens, return_counts=True)
        sorted_indices = np.argsort(-counts)
        
        # Return top N with counts
        keywords = unique[sorted_indices][:top_n].tolist()
        counts = counts[sorted_indices][:top_n].tolist()
        
        return keywords, counts