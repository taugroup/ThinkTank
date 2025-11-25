"""
Qwen evaluator implementation (requires Ollama)
"""

import requests
from .base_evaluator import BaseEvaluator


class QwenEvaluator(BaseEvaluator):
    """
    Qwen evaluator using Ollama
    """
    
    def __init__(self, model_name: str = "qwen3:8b", base_url: str = "http://localhost:11434", **kwargs):
        super().__init__(model_name, **kwargs)
        self.base_url = base_url
        self.thinking_mode = kwargs.get('thinking_mode', False)
        self.session = None
    
    def _initialize_client(self):
        """Initialize Ollama connection"""
        try:
            self.session = requests.Session()
            
            # Check if Ollama server is running
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                raise Exception(f"Ollama server not accessible at {self.base_url}")
            
            # Check if the specified model is available
            available_models = response.json().get("models", [])
            model_names = [model["name"] for model in available_models]
            
            if self.model_name not in model_names:
                print(f"Model {self.model_name} not found. Attempting to pull...")
                
                # Try to pull the model
                pull_response = self.session.post(
                    f"{self.base_url}/api/pull",
                    json={"name": self.model_name}
                )
                
                if pull_response.status_code != 200:
                    raise Exception(f"Failed to pull model {self.model_name}")
                
                print(f"Successfully pulled {self.model_name}")
            
            # Test the model
            test_response = self._call_llm("Hello", test_mode=True)
            if not test_response:
                raise Exception(f"Model {self.model_name} is not responding properly")
            
            print(f"✅ Qwen evaluator initialized: {self.model_name}")
            
        except requests.exceptions.ConnectionError:
            raise Exception(
                "Cannot connect to Ollama server. Please ensure:\n"
                "1. Ollama is installed (download from https://ollama.com)\n"
                "2. Run 'ollama serve' in a terminal\n"
                f"3. The server is accessible at {self.base_url}"
            )
        except Exception as e:
            raise Exception(f"Failed to initialize Qwen: {str(e)}")
    
    def _call_llm(self, prompt: str, test_mode: bool = False, **kwargs) -> str:
        """Call Qwen via Ollama API"""
        # Add thinking mode instruction if enabled
        if self.thinking_mode and not test_mode:
            prompt = f"{prompt} /think"
        elif not self.thinking_mode and not test_mode:
            prompt = f"{prompt} /no_think"
        
        # Prepare the payload for Ollama API
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
                "num_ctx": kwargs.get("context_length", 4096)
            }
        }
        
        try:
            # Make request to Ollama chat API
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=360
            )
            
            if response.status_code == 200:
                data = response.json()
                choices = data.get("choices", [])
                if choices and isinstance(choices, list) and len(choices) > 0:
                    message = choices[0].get("message", {})
                    content = message.get("content", "")
                    return content.strip()
                else:
                    raise Exception("No choices returned from Ollama API")
            else:
                # Fallback to legacy API format
                legacy_payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": payload["options"]
                }
                
                legacy_response = self.session.post(
                    f"{self.base_url}/api/generate",
                    json=legacy_payload,
                    timeout=360
                )
                
                if legacy_response.status_code == 200:
                    return legacy_response.json().get("response", "").strip()
                else:
                    raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
                    
        except requests.exceptions.Timeout:
            return "Error: Request timed out. Qwen may be processing a complex reasoning task."
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {str(e)}")
    
    def set_thinking_mode(self, enabled: bool):
        """Enable or disable Qwen's thinking mode"""
        self.thinking_mode = enabled
        print(f"Qwen thinking mode: {'enabled' if enabled else 'disabled'}")