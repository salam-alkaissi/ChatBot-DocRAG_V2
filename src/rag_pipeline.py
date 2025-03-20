# src/rag_pipeline.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import google.generativeai as genai

load_dotenv("config/api_keys.env")
genai.configure(api_key="AIzaSyDbkx5JmWKdCZ7FHKOVZJZRZo940VAH0f8")

class RAGPipeline:
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = genai.GenerativeModel(model_name='gemini-2.0-flash') #gemini-2.0-flash   #gemini-2.0-flash-lite #Gemini 1.5 Flash
        # self.llm = ChatOpenAI(
        #     openai_api_key=os.getenv("OPENAI_API_KEY"),
        #     openai_api_base="https://openrouter.ai/api/v1",
        #     model_name="deepseek/deepseek-r1:free", #gpt-3.5-turbo #deepseek/deepseek-r1:free
        #     temperature=0.7,
        #     max_tokens=1000,
        #     # model_kwargs={
        #     #     "headers": {    
        #     #         "HTTP-Referer": os.getenv("HTTP_REFERER", "http://localhost:7861"),
        #     #         "X-Title": "DocuRAG AI"
        #     #     }
        #     # }
        # )
        
        self.prompt = ChatPromptTemplate.from_template(
            """Answer using this context:
            {context}
            
            Question: {question}
            """
        )

    # def generate_response(self, question):
    #     context = self.retriever.get_relevant_documents(question)
    #     chain = self.prompt | self.llm
    #     return chain.invoke({
    #         "context": context,
    #         "question": question
    #     }).content
    
    # Gemini 2.0 Flash
    def generate_response(self, question):
        # Retrieve context from the retriever
    
        context = self.retriever.get_relevant_documents(question)
    
        # Format the prompt using your template
        prompt_text = self.prompt.format(context=context, question=question)
        
        # Call the Gemini model with the correct method
        response = self.llm.generate_content(prompt_text)  
        
        # Return the text from the response
        return response.text
        
