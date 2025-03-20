# generation.py
import re
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import networkx as nx
from nltk.tokenize import sent_tokenize
from textwrap import dedent

STRUCTURED_TEMPLATE = """
### Key Themes
{themes}

### Detailed Analysis
- **Impact**: {impact}
- **Technical Challenges**: {challenges}
- **Solution Approaches**: {solutions}
"""

class DocumentAnalyzer:
    def __init__(self):
        self.lang_detector = LanguageDetector()
        self.domain_classifier = DomainClassifier()
        self.keyterm_extractor = KeytermExtractor()
        self.summarizer = TextRankSummarizer()
        self._initialize_nlp_resources()

    def _initialize_nlp_resources(self):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)

    def full_analysis(self, text):
        """Traditional NLP analysis pipeline"""
        try:
            clean_text = self._clean_text(text)
            return {
                "language": self.lang_detector.detect(clean_text),
                "domain": self.domain_classifier.predict(clean_text),
                "key_terms": self.keyterm_extractor.extract(clean_text),
                "structured_summary": self._generate_structured_analysis(clean_text),
                "text_summary": self.summarizer.summarize(clean_text)
            }
        except Exception as e:
            return {"error": str(e)}

    def _clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        return re.sub(r'[^\w\s.,;:!?]', '', text)

    def _generate_structured_analysis(self, text):
        """Generate structured output using traditional NLP"""
        key_terms = self.keyterm_extractor.extract(text)
        summary = self.summarizer.summarize(text)
        
        # Create template-based analysis
        return STRUCTURED_TEMPLATE.format(
            themes='\n'.join(f'- {term}' for term in key_terms[:3]),
            impact=self._generate_impact_statement(summary),
            challenges=self._find_challenges(text),
            solutions=self._suggest_solutions(text)
        )

    def _generate_impact_statement(self, summary):
        """Heuristic-based impact statement"""
        verbs = ["enable", "facilitate", "improve", "enhance"]
        for verb in verbs:
            if verb in summary:
                return f"This research {verb}s {summary.split(verb)[1][:100]}..."
        return "Significant impact on the field."

    def _find_challenges(self, text):
        """Rule-based challenge detection"""
        challenges = []
        if 'complex' in text: challenges.append("System complexity")
        if 'ambiguous' in text: challenges.append("Ambiguity resolution")
        if 'data' in text: challenges.append("Data scarcity")
        return '\n'.join(f'- {c}' for c in challenges[:3]) or "No specific challenges identified"

    def _suggest_solutions(self, text):
        """Pattern-based solution suggestions"""
        solutions = []
        if 'algorithm' in text: solutions.append("Novel algorithmic approaches")
        if 'model' in text: solutions.append("Improved modeling techniques")
        if 'data' in text: solutions.append("Enhanced data collection methods")
        return '\n'.join(f'- {s}' for s in solutions[:3]) or "Standard methodologies apply"
        
class LanguageDetector:
    # Your existing implementation
    def __init__(self):
        self.stopwords = {
            'en': set(nltk.corpus.stopwords.words('english')),
            'fr': set(nltk.corpus.stopwords.words('french'))
        }
    
    def detect(self, text):
        words = [word.lower() for word in re.findall(r'\w+', text)]
        lang_scores = {lang: sum(1 for word in words if word in stops) 
                      for lang, stops in self.stopwords.items()}
        return max(lang_scores, key=lang_scores.get)

class DomainClassifier:
    # Your existing implementation
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2))
        self.classifier = LinearSVC()
        self.classes = ['technology', 'health', 'finance', 'education']
        
    def train(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
    
    def predict(self, text):
        X = self.vectorizer.transform([text])
        return self.classifier.predict(X)[0]

class KeytermExtractor:
    # Enhanced implementation
    def __init__(self, top_n=10):
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=500)
        self.top_n = top_n
        
    def extract(self, text):
        try:
            tfidf_matrix = self.tfidf.fit_transform([text])
            return [self.tfidf.get_feature_names_out()[i] 
                   for i in np.argsort(tfidf_matrix.toarray())[0][-self.top_n:]]
        except:
            return []

class TextRankSummarizer:
    # Optimized implementation
    def __init__(self, ratio=0.2):
        self.ratio = ratio
        
    def summarize(self, text):
        sentences = sent_tokenize(text)
        if len(sentences) < 3:
            return text
            
        sim_matrix = self._build_similarity_matrix(sentences)
        scores = nx.pagerank(nx.from_numpy_array(sim_matrix))
        ranked = sorted(((scores[i], s) for i, s in enumerate(sentences)), 
                      key=lambda x: x[0], reverse=True)
        return ' '.join(s for _, s in ranked[:int(len(ranked)*self.ratio)])
    
    def _sentence_similarity(self, sent1, sent2):
        words1 = set(sent1.lower().split())
        words2 = set(sent2.lower().split())
        return len(words1 & words2) / (len(words1 | words2) + 1e-8)
    
    def _build_similarity_matrix(self, sentences):
        size = len(sentences)
        matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if i != j:
                    matrix[i][j] = self._sentence_similarity(
                        sentences[i], sentences[j])
        return matrix