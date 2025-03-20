import gradio as gr
import numpy as np
from src.document_processing import extract_text, clean_text, chunk_text
from src.hybrid_retrieval import HybridRetrieval  # New import
from src.generation import SummaryGenerator
from src.graph_summary import generate_keyword_table, generate_bar_chart

# Initialize components
hybrid_retriever = HybridRetrieval()
summarizer = SummaryGenerator()
doc_analyzer = DocumentAnalyzer()

def process_document(file):
    try:
        # Step 1: Extract and analyze document
        text = doc_analyzer.extract_text(file.name)
        analysis = doc_analyzer.analyze_document(text)
        
        # Step 2: Store analysis results
        stored_data = {
            'raw_text': text,
            'analysis': analysis
        }
        
        return "Document analyzed successfully!", stored_data
    
    except Exception as e:
        return f"Error processing document: {str(e)}", None

def handle_query(query, chat_history, stored_data):
    try:
        # Step 3: Semantic search within analysis
        query_embedding = doc_analyzer.semantic_model.encode(query)
        similarity = np.dot(stored_data['analysis']['embedding'], query_embedding)
        
        # Step 4: Generate context-aware summary
        summary = summarizer.generate(
            stored_data['raw_text'], 
            stored_data['analysis']
        )
        
        # Step 5: Format results
        return {
            "summary": summary,
            "concepts": stored_data['analysis']['concepts'],
            "similarity": similarity
        }
    
    except Exception as e:
        return {"error": str(e)}
    
def create_interface(rag_pipeline):
    """Build Gradio interface with proper output mapping"""
    with gr.Blocks() as app:
    # Document upload and processing
        with gr.Row():
            file_input = gr.File()
            process_btn = gr.Button("Analyze Document")
        
        # Analysis visualization
        with gr.Row():
            gr.Markdown("### Document Structure")
            section_view = gr.JSON()
            concept_view = gr.HighlightedText()
        
        # Query interface
        with gr.Row():
            query_input = gr.Textbox(label="Ask about the document")
            submit_btn = gr.Button("Get Summary")
            
        # New GenAI Chat Section
        with gr.Row():
            chatbot = gr.Chatbot(height=400)
            msg = gr.Textbox(label="Ask about documents")
            
        def respond(message, chat_history):
            response = rag_pipeline.generate_response(message)
            chat_history.append((message, response))
            return "", chat_history
        
        msg.submit(respond, [msg, chatbot], [msg, chatbot])    
        # Results display
        with gr.Column():
            summary_output = gr.Textbox(label="AI Summary")
            confidence = gr.Label(label="Relevance Score")
    
        # Connect components
        process_btn.click(
            process_document,
            inputs=file_input,
            outputs=[status, stored_data]
        )
        
    submit_btn.click(
        handle_query,
        inputs=[query_input, stored_data],
        outputs=[summary_output, confidence]
    )
    return app

# Launch application
if __name__ == "__main__":
    app = create_interface()
    app.launch(
        server_name="localhost",
        server_port=7861,
        share=False
    )

