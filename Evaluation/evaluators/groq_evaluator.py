"""
Groq evaluator implementation
"""

import os
import time
from functools import wraps
from groq import Groq
from .base_evaluator import BaseEvaluator


def retry_with_exponential_backoff(max_retries=3, initial_delay=2.5, exponential_base=2):
    """Decorator to retry with exponential backoff on rate limits"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e)
                    if "rate_limit_exceeded" in error_str or "429" in error_str:
                        if attempt < max_retries - 1:
                            print(f"⏸️ Rate limit hit. Waiting {delay:.1f}s before retry {attempt + 1}/{max_retries}...")
                            time.sleep(delay)
                            delay *= exponential_base
                        else:
                            print(f"❌ Max retries reached. Rate limit still active.")
                            raise
                    else:
                        raise
            return None
        return wrapper
    return decorator


class GroqEvaluator(BaseEvaluator):
    """
    Groq evaluator using Groq API
    """
    
    def __init__(self, model_name: str = "llama-3.1-70b-versatile", **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = kwargs.get('api_key') or os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable not found")
    
    def _initialize_client(self):
        """Initialize Groq client"""
        try:
            if self.api_key:
                self.client = Groq(api_key=self.api_key)
            else:
                self.client = Groq()  # Will look for GROQ_API_KEY environment variable
            
            # Test the model with a simple prompt
            test_response = self._call_llm("Hello", test_mode=True)
            if not test_response:
                raise Exception(f"Model {self.model_name} is not responding properly")
            
            print(f"✅ Groq evaluator initialized: {self.model_name}")
            
        except Exception as e:
            raise Exception(f"Failed to initialize Groq: {str(e)}")
    
    @retry_with_exponential_backoff(max_retries=5, initial_delay=3)
    def _call_llm(self, prompt: str, test_mode: bool = False, **kwargs) -> str:
        """Call Groq API"""
        try:
            # Prepare messages for Groq API
            messages = [{"role": "user", "content": prompt}]
            
            # Make request to Groq API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                top_p=kwargs.get("top_p", 1),
                stream=False
            )
            
            # Extract content from response
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                return content.strip() if content else ""
            else:
                raise Exception("No response from Groq API")
                
        except Exception as e:
            raise Exception(f"Groq API call failed: {str(e)}")