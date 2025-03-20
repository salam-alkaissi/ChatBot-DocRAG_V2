# docurag/app.py
import gradio as gr
import numpy as np
import os
import time
from dotenv import load_dotenv
from src.document_processing import extract_text, clean_text, chunk_text, detect_language, generate_chunk_summaries
from src.hybrid_retrieval import HybridRetrieval
from src.generation import SummaryGenerator
from src.graph_summary import generate_keyword_table, generate_bar_chart
from src.rag_pipeline import RAGPipeline
from transformers import pipeline
import torch
from datetime import datetime
import tempfile
import topicwizard
from src.topic_visualization import generate_topic_visualization
import requests
from transformers import pipeline, BertConfig, BertForTokenClassification, AutoTokenizer
from src.wikidata_utils import get_wikidata_entities, get_simple_description  
# import set
# from nltk.metrics.distance import jaccard_distance 
import matplotlib
matplotlib.use('Agg')  # Needed for Gradio compatibility

# Load environment variables
load_dotenv("config/api_keys.env")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
# NER Setup
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=True)
model = BertForTokenClassification.from_pretrained("./models/NER_finetuned_bert_base", from_tf=True)
nlp = pipeline('ner', model=model, tokenizer=tokenizer)

# Initialize components
hybrid_retriever = HybridRetrieval()
summarizer = SummaryGenerator()
rag_pipeline = RAGPipeline(hybrid_retriever)
notes_output = gr.Textbox(
    label="Saved Notes",
    interactive=False,  # Make it read-only
    lines=10,
    elem_classes="notes-box"
)


processing_outputs = [
    gr.Textbox(label="Processing Status"),        # Index 0
    gr.Textbox(label="Detected Language"),       # Index 1
    gr.Markdown(label="Document Summary"),       # Index 2 (LLM summary)
    gr.DataFrame(label="Key Terms",              # Index 3
                headers=["Term", "Frequency", "Wikidata Description"],
                datatype=["str", "number", "str"]),
    gr.Image(label="Visualization"),              # Index 4
    gr.Textbox(label="Document Domain"),           # Index 5
    gr.Image(label="Topic Network"), #index 6
    gr.Image(label="Topic Projection"), #index 7
    gr.DataFrame(label="Topic Terms",                   #index 8
                    headers=["Topic", "Terms"],
                    datatype=["str", "str"]),
    # gr.Textbox(label="Wikidata Entities", lines=10)  # Index 9 - New output 
]

# Add to event handlers
def save_to_notes(chat_history, current_notes):
    """Format notes as text"""
    new_notes = current_notes or ""
    if chat_history:
        last_entry = chat_history[-1]
        note_content = last_entry[1] if isinstance(last_entry, tuple) else last_entry.get("content", "")
        new_notes += f"Note saved at {datetime.now().strftime('%H:%M')}:\n{note_content}\n\n"
    return new_notes

def save_notes_to_file(notes):
    # Check if notes is empty or just whitespace
    if not notes or not notes.strip():
        return None
    
    # Create directory if it doesn’t exist
    os.makedirs("user_notes", exist_ok=True)
    
    # Generate a unique filename with timestamp
    filename = f"user_notes/notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    try:
        # Write notes to file with UTF-8 encoding
        with open(filename, "w", encoding="utf-8") as f:
            f.write(notes)
        return filename
    except Exception as e:
        print(f"Error saving notes to file: {e}")
        return None
    
def classify_domain(text):
    """Classify document domain using zero-shot classification"""
    classifier = pipeline("zero-shot-classification",
                        # model="facebook/bart-large-mnli"
                        model="MoritzLaurer/deberta-v3-base-zeroshot-v1",  # Smaller model
                        device=-1 )
    
    candidate_labels = ["education", "health", "technology", "legal", "weather", "sports", "finance", "entertainment", "news","Curriculum vitae",
                        "politics", "economy", "entertainment", "environment","other"]
    result = classifier(text[:5000], candidate_labels)  
    return result['labels'][0]  

def chunk_summarization(text, summarizer, chunk_size=500):
    """Memory-safe chunk-based summarization"""
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = []
    
    for chunk in chunks:
        try:
            summary = summarizer.generate_structured_summary(chunk) #generate_structured_summary #generate
            summaries.append(summary)
            torch.cuda.empty_cache()
        except Exception as e:
            summaries.append(f"[Chunk error: {str(e)}]")
    
    return "\n".join(summaries)

def process_basic(file):
    """Handle basic processing only"""
    try:
        text = extract_text(file.name)
        cleaned_text = clean_text(text)
        detected_language = detect_language(cleaned_text[:1000])
        
        return [
            "Basic processing completed!",  # Index 0
            f"Language: {detect_language(cleaned_text[:1000])}",  # Index 1
            f"Domain: {classify_domain(cleaned_text)}"  # Index 5
        ]
    
    except Exception as e:
        return [
            f"Error: {str(e)}",  # Index 0
            "Language detection failed",  # Index 1
            "Domain classification failed"  # Index 5
        ]


def process_analysis(file):
    """Handle analysis-only processing"""
    try:
        text = extract_text(file.name)
        cleaned_text = clean_text(text)
        chunks = chunk_text(cleaned_text)
        hybrid_retriever.index_documents(chunks)
        
        keywords = hybrid_retriever.extract_keywords(cleaned_text)
        counts = [cleaned_text.lower().count(kw.lower()) for kw in keywords]
        # Wikipedia descriptions
        descriptions = [get_simple_description(kw) for kw in keywords]
        # Process text for NER and get detailed Wikidata info
        word_entity_pairs = process_text(cleaned_text)
        wiki_entities = get_wikidata_entities(word_entity_pairs)
        # Enhance key terms with detailed Wikidata info where available
        key_terms_data = []
        for kw, cnt, desc in zip(keywords[:10], counts[:10], descriptions[:10]):
            wiki_desc = desc  # Default to simple description
            # Check if this keyword has detailed Wikidata entries
            for (word, label), entities in wiki_entities.items():
                if word.lower() == kw.lower():
                    wiki_desc = "; ".join([f"{e['text']}: {e['description']}" for e in entities])
                    break
            key_terms_data.append([kw, cnt, wiki_desc])
        # chart_path = generate_bar_chart(keywords[:10], counts[:10])
        chart_path = generate_bar_chart(keywords[:10], counts[:10]) if keywords else None
        
        summary = summarizer.generate_structured_summary(cleaned_text) #generate_structured_summary , generate
         # Generate Topic Visualization
        network_path, projection_path, topic_terms = generate_topic_visualization(cleaned_text)
            
        return [
            summary,  # Index 2
            key_terms_data,
            # [[kw, cnt] for kw, cnt in zip(keywords, counts)][:10],  # Index 3
            chart_path,  # Index 4
            network_path,    # Use actual figures from generate_topic_visualization
            projection_path, 
            [[topic, terms] for topic, terms in topic_terms]
            # wiki_output
        ]
    
    except Exception as e:
        return [
            "Summary unavailable",  # Index 2
            [["Error", 0]],  # Index 3
            None,  # Index 4
            None,  # topic_network
            None,  # topic_projection
            []     # topic_summary
            # f"Error fetching Wikidata: {str(e)}"
        ]

def process_text(text: str):
    """Process text and get word-entity pairs"""
    result = nlp(text)
    words = []
    entities = []
    
    for entry in result:
        token = entry["word"]
        entity = entry["entity"]
        if token.startswith("##"):
            words[-1] = words[-1] + token[2:]
        else:
            if entity.startswith("I"):
                words[-1] = words[-1] + " " + token
            else:
                words.append(token)
                entities.append(entity[2:])
    
    return set(zip(words, entities))

def handle_user_query(query, chat_history):
    """Process queries with strict message formatting"""
    try:
        # Generate response from RAG system
        response = rag_pipeline.generate_response(query)
        
        # Append messages in correct format
        updated_history = chat_history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ]
        
        return updated_history
    
    except Exception as e:
        error_msg = f"System Error: {str(e)}"
        return chat_history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": error_msg}
        ]

def create_interface():
    with gr.Blocks(title="DocRAG", css=".markdown {max-width: 1200px} footer {display: none !important;}") as app:
        gr.Markdown("# 📄 DocRAG - Intelligent Document Analysis")
        
        with gr.Row():
            # Document Upload Column
            with gr.Column(scale=1):
                gr.Markdown("## Step 1: Document Processing")
                file_input = gr.File(label="Upload PDF/TXT", file_types=[".pdf", ".txt"])
                upload_btn = gr.Button("Process Document", variant="primary")
                processing_outputs[0].render()
                processing_outputs[1].render()
                processing_outputs[5].render()
                analysis_btn = gr.Button("Analyze Document", variant="primary")
                
            # Chat Interface Column
            with gr.Column(scale=3):
                gr.Markdown("## Step 2: Interactive Analysis")
                chatbot = gr.Chatbot(
                    value=[{"role": "assistant", "content": "Ready to analyze your document!"}],
                    label="Analysis Dialogue",
                    height=500,
                    render_markdown=True,
                    type="messages",  
                    avatar_images=(None, None)
                )
                with gr.Row():
                    suggestion1 = gr.Button("What is the main topic?", size="sm")
                    suggestion2 = gr.Button("Summarize key points", size="sm")
                    suggestion3 = gr.Button("Important findings", size="sm")
                with gr.Row():
                    suggestion4 = gr.Button("Specific recommendations", size="sm")
                    suggestion5 = gr.Button("Document conclusion", size="sm")
                query_input = gr.Textbox(
                    label="Query Input",
                    placeholder="Type your question here...",
                    lines=1
                )
                submit_btn = gr.Button("Generate Analysis", variant="primary")
            
            # Results Column
            with gr.Column(scale=2):
                gr.Markdown("## Analysis Results")
                with gr.Tab("Summary"):
                    processing_outputs[2].render()
                with gr.Tab("Key Terms"):
                    processing_outputs[3].render()
                with gr.Tab("Visualization"):
                    processing_outputs[4].render()
                with gr.Tab("Topics", elem_id="topic-tab"):
                    with gr.Column():
                        with gr.Row():
                            processing_outputs[6].render()
                            processing_outputs[7].render()
                    processing_outputs[8].render()
                # with gr.Tab("Wikidata"):  # New tab for Wikidata results
                #     processing_outputs[9].render()
                with gr.Tab("Notes"):
                    notes_output.render()
                    note_btn = gr.Button("Add to Notes", variant="secondary") 
                    # copy_btn = gr.Button("Copy Notes", variant="secondary", elem_classes="copy-btn")
                    clear_notes_btn = gr.Button("Clear Notes", variant="secondary")
                    save_btn = gr.Button("Save to File", variant="secondary")
                saved_file = gr.File(label="Download Notes", visible=False)
        # Add state for notes
        notes_state = gr.State(value="")
        
        # Event chain for note saving
        note_btn.click(
            save_to_notes,
            inputs=[chatbot, notes_state],
            outputs=[notes_state]
        ).then(
            lambda x: x,
            inputs=[notes_state],
            outputs=[notes_output]
        )
        
        # Clear notes handler
        clear_notes_btn.click(
            lambda: ("", ""),  # Clear both state and textbox
            outputs=[notes_state, notes_output]
        )
            
        # Event handlers
        upload_btn.click(
            process_basic,
            inputs=file_input,
            outputs= [
                processing_outputs[0],  # Status
                processing_outputs[1],  # Language
                processing_outputs[5]   # Domain
            ]
        )
        analysis_btn.click(
            process_analysis,
            inputs=file_input,
            outputs=[
                processing_outputs[2],  # Summary
                processing_outputs[3],  # Key Terms
                processing_outputs[4],   # Visualization
                processing_outputs[6],         
                processing_outputs[7],      
                processing_outputs[8]
                # processing_outputs[9]
            ]
        )
        
        submit_btn.click(
            handle_user_query,
            inputs=[query_input, chatbot],
            outputs=[chatbot]  
        ).then(
            lambda: "",  
            inputs=None,
            outputs=query_input
        )
        save_btn.click(
            fn=save_notes_to_file,
            inputs=notes_state,
            outputs=notes_output
        )
        # Add suggestion button handlers
        suggestion1.click(lambda: "What is the main topic of this document?", outputs=query_input)
        suggestion2.click(lambda: "Can you summarize the key points?", outputs=query_input)
        suggestion3.click(lambda: "What are the most important findings?", outputs=query_input)
        suggestion4.click(lambda: "Are there any specific recommendations?", outputs=query_input)
        suggestion5.click(lambda: "What is the document's conclusion?", outputs=query_input)

    return app

if __name__ == "__main__":
    # Set memory optimization flags
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)
    
    app = create_interface()
    app.launch(
        server_name="localhost",
        server_port=7861,
        share=False,
        # auth=("admin", os.getenv("APP_PASSWORD", "docurag")),
        # auth_message="Enter admin credentials:"
    )