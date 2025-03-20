# src/generation.py
import torch
# from transformers import T5Tokenizer, T5ForConditionalGeneration
from src.document_processing import chunk_text
from transformers import BartTokenizer, BartForConditionalGeneration
import logging
import cohere
import os
import re
##COHERE_API_KEY="ZGzGvuaIHCoMUihq1DoHSpl741wUTwZuXdSkgcKQ"
##COHERE_API_KEY="P28PBGM9qfKcd0TJZChtwrhJPSVLK102JcQ2aN0v"

STRUCTURED_PROMPT_TEMPLATE = """Generate a comprehensive summary with this exact structure:

### Key Themes
{bullet_points} 
- **Impact**: {impact_sentences}

### Detailed Analysis
{examples}
- **Technical Challenges**: {challenges}
- **Solution Approaches**: {solutions}

Use professional academic language. Focus on conceptual relationships and technical implementations.
"""

class SummaryGenerator:
    def __init__(self):
        # Set memory management for CUDA
        # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        self.device = torch.device("cpu") #"cuda" if torch.cuda.is_available() else 
        self.model_name = "facebook/bart-large-cnn" #"google/flan-t5-small"  # Better model for summarization google/flan-t5-base
        # self.tokenizer = None
        # self.model = None
        logging.info(f"Initializing with model: {self.model_name}")
        self.initialize_model()
        self.co = cohere.Client(os.getenv("COHERE_API_KEY"))
        self.chunk_size = 2000
        # Configure logging
        # logging.basicConfig(level=logging.INFO)
        
    def initialize_model(self):
        """Initialize the BART model and tokenizer."""
        # try:
        #     self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, legacy=False)
        #     self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        #     self.model = self.model.to(self.device)
        #     logging.info(f"Loaded model {self.model_name} on {self.device}")
        # except Exception as e:
        #     logging.error(f"Model loading failed: {str(e)}")
        #     raise
        try:
            logging.info(f"Attempting to load tokenizer for {self.model_name}")
            self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
            logging.info(f"Attempting to load model for {self.model_name}")
            self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
            self.model = self.model.to(self.device)
            logging.info(f"Loaded model {self.model_name} on {self.device}")
            logging.info(f"Loaded model {self.model_name} on {self.device}")
        except Exception as e:
            logging.error(f"Model loading failed: {str(e)}")
            raise
        
    def _chunk_text(self, text):
        """Split text into chunks based on character count."""
        return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

    def generate(self, text, max_input_length=1024, max_new_tokens=200):  
    #     try:
    #         if not text or len(text.strip()) < 50:
    #             return "Insufficient text content for meaningful summary"

    # # Better prompt engineering
    #         inputs = self.tokenizer(
    #             f"Generate a comprehensive, detailed summary of the following document: {text}",
    #             max_length=max_input_length,
    #             truncation=True,
    #             padding="max_length",
    #             return_tensors="pt"
    #         ).to(self.device)

    #         # Enhanced generation parameters
    #         outputs = self.model.generate(
    #             inputs.input_ids,
    #             max_new_tokens=max_new_tokens,     # Increased from 150
    #             min_length=50,          # Increased from 50
    #             # length_penalty=2,      # Adjusted for longer summaries
    #             # no_repeat_ngram_size=4,
    #             early_stopping=True,
    #             num_beams=4,
    #             # temperature=0.8,         # Slightly higher for diversity
    #             do_sample=False,         # Better coherence with beam search
    #             repetition_penalty=2.0
    #         )

    #         return self.postprocess_summary(
    #             self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    #         )
    
    #     except Exception as e:
    #         logging.error(f"Generation failed: {str(e)}")
    #         return "Summary generation error"
        """Generate a summary using the local BART model."""
        try:
            if not text or len(text.strip()) < 50:
                return "Insufficient text content for meaningful summary"

            # Truncate to max_input_length
            inputs = self.tokenizer(
                f"Generate a comprehensive summary of the following document: {text[:max_input_length]}",
                max_length=max_input_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                min_length=50,
                early_stopping=True,
                num_beams=4,
                do_sample=False,
                repetition_penalty=2.0
            )

            return self.postprocess_summary(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
        except Exception as e:
            logging.error(f"Generation failed: {str(e)}")
            return "Summary generation error"

    def postprocess_summary(self, summary):
        """Clean up generated summary"""
        # Remove any bullet points or markdown artifacts
        summary = summary.replace("•", "").replace("##", "").strip()
        
        # Capitalize first letter if needed
        if summary and summary[0].islower():
            summary = summary[0].upper() + summary[1:]
            
        # Ensure proper sentence endings
        if summary and summary[-1] not in {'.', '!', '?'}:
            summary += '.'
            
        return summary
    
    def generate_structured_summary(self, text):
        # """Generate structured summary using Cohere"""
        # try:
        #     response = self.co.chat(
        #         message=self._build_cohere_prompt(text),
        #         model="command-r-plus",
        #         temperature=0.3,
        #         preamble="You are a technical documentation analyst",
        #         connectors=[{"id": "web-search"}]
        #     )
        #     return self._format_cohere_response(response.text)
            
        # except Exception as e:
        #     return f"Cohere Error: {str(e)}"
        # """Generate a structured summary using Cohere for the full text."""
        # try:
        #     if not self.co:
        #         raise ValueError("Cohere API key not available")
            
        #     chunks = self._chunk_text(text)
        #     all_results = []

        #     for chunk in chunks:
        #         prompt = self._build_cohere_prompt(chunk)
        #         response = self.co.generate(
        #             model='command-xlarge',
        #             prompt=prompt,
        #             max_tokens=500,  # Increased for detailed response
        #             temperature=0.3,
        #             stop_sequences=["\n\n"]
        #         )
        #         all_results.append(response.generations[0].text)

        #     return self._synthesize_cohere_results(all_results)
        # except Exception as e:
        #     logging.error(f"Cohere structured summary failed: {str(e)}")
        #     return f"Cohere Error: {str(e)}"
        try:
            if not self.co:
                raise ValueError("Cohere API key not available")

            chunks = self._chunk_text(text)
            all_results = []

            for i, chunk in enumerate(chunks):
                prompt = self._build_cohere_prompt(chunk)
                response = self.co.generate(
                    model='command-xlarge',
                    prompt=prompt,
                    max_tokens=500,
                    temperature=0.3,
                    stop_sequences=["\n\n"]
                )
                all_results.append(response.generations[0].text)

            return self._synthesize_cohere_results(all_results)
        except Exception as e:
            logging.error(f"Cohere structured summary failed: {str(e)}")
            return self.generate(text) if text else f"Cohere Error: {str(e)}"
        
    def _build_cohere_prompt(self, text):
        example = """1. Key Themes
        - Theme 1: Example theme
        - Theme 2: Another theme
        2. Detailed Analysis
        - Technical Challenges
          - Challenge 1: Example challenge
          - Challenge 2: Another challenge
        - Solution Approaches
          - Approach 1: Example solution
          - Approach 2: Another solution
        3. Impact Assessment
        - Impact 1: Example impact sentence.
        - Impact 2: Another impact sentence."""
        return f"""Analyze this document section and structure your response EXACTLY as shown in the example below, with:
        1. Key Themes (3-5 bullet points)
        2. Detailed Analysis with:
           - Technical Challenges (5-10 challenges)
           - Solution Approaches (5-10 methods)
        3. Impact Assessment (5-10 sentences)

        Include quantitative estimates where possible. Use professional academic language and focus on technical implementations.

        Example response:
        {example}

        Document section: "{chunk_text}"
        """

    def _format_cohere_response(self, response):
        """Convert Cohere response to markdown format"""
        sections = {
            "1. Key Themes": "### Key Themes",
            "2. Detailed Analysis": "### Detailed Analysis",
            "3. Impact Assessment": "### Impact"
        }
        
        for k, v in sections.items():
            response = response.replace(k, v)
            
        return response.replace("- ", "• ")

    def _synthesize_cohere_results(self, results):
        """Combine chunked Cohere results into a single structured response."""
        themes = []
        challenges = []
        solutions = []
        impacts = []

        for result in results:
            # Use regex to extract sections more robustly
            theme_match = re.findall(r'1\. Key Themes\s*[\n-]*(.*?)(?=\n2\.|\Z)', result, re.DOTALL)
            if theme_match:
                themes.extend([t.strip() for t in re.findall(r'- (.*)', theme_match[0]) if t.strip()])

            analysis_match = re.findall(r'2\. Detailed Analysis\s*[\n-]*(.*?)(?=\n3\.|\Z)', result, re.DOTALL)
            if analysis_match:
                challenges_match = re.findall(r'- Technical Challenges\s*[\n-]*(.*?)(?=\n- Solution Approaches|\Z)', analysis_match[0], re.DOTALL)
                if challenges_match:
                    challenges.extend([c.strip() for c in re.findall(r'- (.*)', challenges_match[0]) if c.strip()])
                solutions_match = re.findall(r'- Solution Approaches\s*[\n-]*(.*?)(?=\n3\.|\Z)', analysis_match[0], re.DOTALL)
                if solutions_match:
                    solutions.extend([s.strip() for s in re.findall(r'- (.*)', solutions_match[0]) if s.strip()])

            impact_match = re.findall(r'3\. Impact Assessment\s*[\n-]*(.*?)(?=\n1\.|\Z)', result, re.DOTALL)
            if impact_match:
                impacts.extend([i.strip() for i in re.findall(r'- (.*)', impact_match[0]) if i.strip()])

        # Limit and deduplicate
        themes = list(dict.fromkeys(themes))[:5]
        challenges = list(dict.fromkeys(challenges))[:10]
        solutions = list(dict.fromkeys(solutions))[:10]
        impacts = list(dict.fromkeys(impacts))[:10]

        # Fill with placeholders if sections are empty
        if not themes:
            themes = ["- No key themes identified"]
        if not challenges:
            challenges = ["- No technical challenges identified"] * 5
        if not solutions:
            solutions = ["- No solution approaches identified"] * 5
        if not impacts:
            impacts = ["- No impact assessment available."] * 5

        bullet_points = "\n".join(themes)
        examples = "\n".join(challenges) + "\n" + "\n".join(solutions)
        impact_sentences = "\n".join(impacts)

        return STRUCTURED_PROMPT_TEMPLATE.format(
            bullet_points=bullet_points,
            examples=examples,
            challenges="\n".join(challenges),
            solutions="\n".join(solutions),
            impact_sentences=impact_sentences
        )