# src/ml_pipeline.py
import torch
from transformers import (
    pipeline,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig
)
from sentence_transformers import SentenceTransformer
from langdetect import detect, DetectorFactory

class DocumentProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models_loaded = False
        self.quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        
    def _init_models(self):
        """Load models with memory optimizations"""
        if not self.models_loaded:
            # Free memory before loading new models
            torch.cuda.empty_cache()

            # 4-bit quantized domain classifier
            # 4-bit quantization config
            self.quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
            self.domain_classifier = AutoModelForSequenceClassification.from_pretrained(
                "philschmid/tiny-bert-sst2-distilled",
                quantization_config=self.quant_config
                # device_map="auto"
            ).to(self.device) 
            
            self.domain_tokenizer = AutoTokenizer.from_pretrained(
                "philschmid/tiny-bert-sst2-distilled"
            )

            # Lightweight sentence model on CPU
            self.keyphrase_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device="cpu"
            )

            # CPU-only PDF processor
            self.pdf_extractor = pipeline(
                "document-question-answering",
                model="impira/layoutlm-document-qa",
                device=-1,
                torch_dtype=torch.float16
            )

            self.models_loaded = True

    def process_pdf(self, file_path):
        try:
            self._init_models()
            
            # Processing pipeline
            raw_text = self.pdf_extractor(image=file_path)["answers"][0]["answer"]
            clean_text = self._clean_text(raw_text)
            
            with torch.no_grad():
                domain = self._classify_domain(clean_text)
                language = self._detect_language(clean_text)
                
            keywords = self._extract_keyphrases(clean_text)
            
            return {
                "text": clean_text,
                "language": language,
                "keywords": keywords,
                "summary": self._generate_summary(clean_text),
                "domain": domain
            }
        # except Exception as e:
        #     return {
        #         "error": str(e),
        #         "text": "",
        #         "domain": -1,
        #         "keywords": [],
        #         "summary": ""
        #     }
        finally:
            self._release_memory()
    def _detect_language(self, text):
        """Language detection with fallback"""
        try:
            return detect(text[:1000])
        except:
            return "unknown"
        
    def _classify_domain(self, text):
        inputs = self.domain_tokenizer(
            text[:256],  # Smaller input size
            return_tensors="pt",
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.domain_classifier(**inputs)
        return torch.argmax(outputs.logits).item()
        # outputs = self.domain_classifier(**inputs)
        # return torch.argmax(outputs.logits).item()

    def _extract_keyphrases(self, text):
        return self.keyphrase_model.encode(
            [text[:512]],  # Truncate long texts
            convert_to_tensor=True,
            show_progress_bar=False
        )

    def _release_memory(self):
        """Aggressive memory cleanup"""
        if hasattr(self, 'domain_classifier'):
            del self.domain_classifier
        if hasattr(self, 'pdf_extractor'):
            del self.pdf_extractor
            
        torch.cuda.empty_cache()
        self.models_loaded = False
        
    def _generate_summary(self, text):
        # Add summarization logic
        return ""