"""
Gemini evaluator implementation
"""

import os
import time
import google.genai as genai
from .base_evaluator import BaseEvaluator


class GeminiEvaluator(BaseEvaluator):
    """
    Gemini Pro evaluator using google.genai
    """
    
    def __init__(self, model_name: str = "gemini-2.5-flash", **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = kwargs.get('api_key') or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not found")
    
    def _initialize_client(self):
        """Initialize Gemini client"""
        try:
            self.client = genai.Client(api_key=self.api_key)
            print(f"✅ Gemini evaluator initialized: {self.model_name}")
        except Exception as e:
            raise Exception(f"Failed to initialize Gemini: {str(e)}")
    
    def _call_llm(self, prompt: str, **kwargs) -> str:
        """Call Gemini API"""
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                chat = self.client.chats.create(model=self.model_name)
                response = chat.send_message(prompt)
                return response.text.strip()
                
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if attempt < max_retries - 1:
                        print(f"Rate limit exceeded. Waiting 60 seconds...")
                        time.sleep(60)
                        continue
                elif "503" in error_str or "UNAVAILABLE" in error_str:
                    if attempt < max_retries - 1:
                        print(f"Service unavailable. Waiting 30 seconds...")
                        time.sleep(30)
                        continue
                
                raise Exception(f"Gemini API error: {str(e)}")
        
        return ""