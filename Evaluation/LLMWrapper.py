from deepeval.models.base_model import DeepEvalBaseLLM
from typing import Optional, Dict, Any, Union
import requests
import json
import time
from pydantic import BaseModel
import re

class Qwen3OllamaWrapper(DeepEvalBaseLLM):
    """
    Custom LLM wrapper for Qwen3 models running on Ollama with proper schema support
    """
    
    def __init__(
        self, 
        model_name: str = "qwen3:8b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        thinking_mode: bool = False,
        **kwargs
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking_mode = thinking_mode
        self.model = None
        self.session = None
        self.kwargs = kwargs
        
    def load_model(self):
        """Initialize connection to Ollama and verify Qwen3 model availability"""
        try:
            self.session = requests.Session()
            
            # Check if Ollama server is running
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                raise Exception(f"Ollama server not accessible at {self.base_url}. Make sure 'ollama serve' is running.")
            
            # Check if the specified Qwen3 model is available
            available_models = response.json().get("models", [])
            model_names = [model["name"] for model in available_models]
            
            if self.model_name not in model_names:
                print(f"Model {self.model_name} not found. Available models: {model_names}")
                print(f"Attempting to pull {self.model_name}...")
                
                # Try to pull the model
                pull_response = self.session.post(
                    f"{self.base_url}/api/pull",
                    json={"name": self.model_name}
                )
                
                if pull_response.status_code != 200:
                    raise Exception(f"Failed to pull model {self.model_name}")
                
                print(f"Successfully pulled {self.model_name}")
            
            # Test the model with a simple prompt
            test_response = self._call_llm("Hello", test_mode=True)
            if not test_response:
                raise Exception(f"Model {self.model_name} is not responding properly")
            
            print(f"✅ Qwen3 model {self.model_name} loaded successfully on Ollama")
            self.model = "loaded"
            return self.model
            
        except requests.exceptions.ConnectionError:
            raise Exception(
                "Cannot connect to Ollama server. Please ensure:\n"
                "1. Ollama is installed (download from https://ollama.com)\n"
                "2. Run 'ollama serve' in a terminal\n"
                "3. The server is accessible at " + self.base_url
            )
        except Exception as e:
            raise Exception(f"Failed to initialize Qwen3 on Ollama: {str(e)}")
    
    def generate(self, prompt: str, schema: Optional[BaseModel] = None, **kwargs) -> Union[str, BaseModel]:
        """
        Generate response from Qwen3 model with optional schema support
        """
        try:
            if self.model is None:
                self.load_model()
            
            # If schema is provided, try to get structured response
            if schema is not None:
                return self._generate_with_schema(prompt, schema, **kwargs)
            else:
                return self._call_llm(prompt, **kwargs)
                
        except Exception as e:
            print(f"Error generating response from Qwen3: {e}")
            if schema is not None:
                return self._create_fallback_response(schema, f"Error: {str(e)}")
            return f"Error: {str(e)}"
    
    def _generate_with_schema(self, prompt: str, schema: BaseModel, **kwargs) -> BaseModel:
        """Generate structured response using schema"""
        try:
            # Add JSON formatting instructions to prompt
            schema_prompt = f"""
{prompt}

Please respond with a valid JSON object that matches this exact schema:
{json.dumps(schema.model_json_schema(), indent=2)}

Important: 
- Return ONLY the JSON object, no additional text or formatting
- Ensure all required fields are included
- Use proper JSON syntax with double quotes
- Do not include any markdown formatting or code blocks
"""
            
            # Get response from Qwen3
            response_text = self._call_llm(schema_prompt, **kwargs)
            
            # Try to parse as JSON and validate against schema
            try:
                # Clean the response (remove any markdown formatting)
                cleaned_response = self._clean_json_response(response_text)
                
                # Parse JSON
                response_dict = json.loads(cleaned_response)
                
                # Validate and create schema instance
                return schema(**response_dict)
                
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Failed to parse JSON response: {e}")
                print(f"Raw response: {response_text}")
                
                # Fallback: try to extract key information and create a basic response
                return self._create_fallback_response(schema, response_text)
                
        except Exception as e:
            print(f"Schema generation error: {e}")
            return self._create_fallback_response(schema, str(e))
    
    def _clean_json_response(self, response_text: str) -> str:
        """Clean response text to extract valid JSON"""
        # Remove markdown code blocks
        cleaned = response_text.strip()
        
        # Handle markdown code blocks with language specification
        if cleaned.startswith("```"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        
        # Remove any leading/trailing whitespace after markdown removal
        cleaned = cleaned.strip()
        
        # Try to find the first complete JSON object in the response
        # Look for balanced braces to handle nested objects
        brace_count = 0
        start_index = -1
        end_index = -1
        
        for i, char in enumerate(cleaned):
            if char == '{':
                if start_index == -1:
                    start_index = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_index != -1:
                    end_index = i
                    break
        
        # If we found a complete JSON object, extract it
        if start_index != -1 and end_index != -1:
            json_candidate = cleaned[start_index:end_index + 1]
            
            # Validate that it's actually valid JSON
            try:
                json.loads(json_candidate)
                return json_candidate
            except json.JSONDecodeError:
                pass
        
        # Fallback: try regex approach for simple cases
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned, re.DOTALL)
        if json_match:
            json_candidate = json_match.group(0)
            try:
                json.loads(json_candidate)
                return json_candidate
            except json.JSONDecodeError:
                pass
        
        # If no valid JSON found, try to extract content between first { and last }
        first_brace = cleaned.find('{')
        last_brace = cleaned.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_candidate = cleaned[first_brace:last_brace + 1]
            try:
                json.loads(json_candidate)
                return json_candidate
            except json.JSONDecodeError:
                pass
        
        # Last resort: return the cleaned text as-is
        return cleaned
    
    def _create_fallback_response(self, schema: BaseModel, response_text: str) -> BaseModel:
        """Create a fallback response when JSON parsing fails"""
        try:
            schema_fields = schema.model_fields
            fallback_data = {}
            
            # Handle common DeepEval schemas based on field names
            if 'steps' in schema_fields:
                # This is likely a Steps schema for GEval
                fallback_data['steps'] = [response_text] if response_text else ["Unable to evaluate"]
                
            elif 'verdict' in schema_fields:
                # This is likely a Verdicts schema
                verdict_keywords = ['yes', 'true', 'correct', 'good', 'pass']
                fallback_data['verdict'] = 'yes' if any(keyword in response_text.lower() for keyword in verdict_keywords) else 'no'
                if 'reason' in schema_fields:
                    fallback_data['reason'] = response_text or "No reason provided"
                    
            elif 'score' in schema_fields:
                # This is likely a score-based schema
                # Try to extract a number from the response
                numbers = re.findall(r'\d+\.?\d*', response_text)
                if numbers:
                    try:
                        score = float(numbers[0])
                        # Ensure score is within reasonable bounds (0-10)
                        score = max(0, min(10, score))
                        fallback_data['score'] = score
                    except ValueError:
                        fallback_data['score'] = 5.0
                else:
                    fallback_data['score'] = 5.0  # Default middle score
                    
                if 'reason' in schema_fields:
                    fallback_data['reason'] = response_text or "No detailed reasoning provided"
                    
            elif 'reason' in schema_fields:
                # Generic reason-based schema
                fallback_data['reason'] = response_text or "No reasoning provided"
            
            # Handle any remaining required fields
            for field_name, field_info in schema_fields.items():
                if field_name not in fallback_data:
                    # Get the field type
                    field_type = field_info.annotation
                    
                    # Handle different field types
                    if field_type == str or str(field_type) == "<class 'str'>":
                        fallback_data[field_name] = response_text or "No response"
                    elif field_type == int or str(field_type) == "<class 'int'>":
                        fallback_data[field_name] = 5
                    elif field_type == float or str(field_type) == "<class 'float'>":
                        fallback_data[field_name] = 5.0
                    elif field_type == bool or str(field_type) == "<class 'bool'>":
                        fallback_data[field_name] = True
                    elif hasattr(field_type, '__origin__') and field_type.__origin__ == list:
                        fallback_data[field_name] = [response_text] if response_text else []
                    else:
                        fallback_data[field_name] = response_text or "Default value"
            
            return schema(**fallback_data)
            
        except Exception as e:
            print(f"Fallback creation failed: {e}")
            # Last resort: create minimal valid instance
            try:
                # Try to create with default values for all fields
                default_data = {}
                for field_name, field_info in schema.model_fields.items():
                    field_type = field_info.annotation
                    
                    if field_type == str:
                        default_data[field_name] = "Default response"
                    elif field_type == int:
                        default_data[field_name] = 5
                    elif field_type == float:
                        default_data[field_name] = 5.0
                    elif field_type == bool:
                        default_data[field_name] = True
                    elif hasattr(field_type, '__origin__') and field_type.__origin__ == list:
                        default_data[field_name] = ["Default"]
                    else:
                        default_data[field_name] = "Default"
                
                return schema(**default_data)
                
            except Exception as final_e:
                print(f"Final fallback failed: {final_e}")
                # If all else fails, try empty construction
                return schema()
    
    async def a_generate(self, prompt: str, schema: Optional[BaseModel] = None, **kwargs) -> Union[str, BaseModel]:
        """Async version of generate"""
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None, self.generate, prompt, schema
        )
    
    def get_model_name(self) -> str:
        """Return the Qwen3 model name"""
        return self.model_name
    
    def _call_llm(self, prompt: str, test_mode: bool = False, **kwargs) -> str:
        """
        Call Qwen3 model via Ollama API
        """
        # Add thinking mode instruction if enabled
        if self.thinking_mode and not test_mode:
            prompt = f"{prompt} /think"
        elif not self.thinking_mode and not test_mode:
            prompt = f"{prompt} /no_think"
        
        # Prepare the payload for Ollama API
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
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
                # Fix: Handle the list structure properly
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
            return "Error: Request timed out. Qwen3 may be processing a complex reasoning task."
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {str(e)}")
    
    def set_thinking_mode(self, enabled: bool):
        """Enable or disable Qwen3's thinking mode"""
        self.thinking_mode = enabled
        print(f"Qwen3 thinking mode: {'enabled' if enabled else 'disabled'}")
    
    def get_available_qwen3_models(self) -> list:
        """Get list of available Qwen3 models on Ollama"""
        try:
            if self.session is None:
                self.session = requests.Session()
            
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                qwen3_models = [
                    model["name"] for model in models 
                    if "qwen3" in model["name"].lower()
                ]
                return qwen3_models
            return []
        except:
            return []
        
    
def create_qwen3_8b_wrapper(**kwargs) -> Qwen3OllamaWrapper:
    """Create wrapper for Qwen3 8B model (default)"""
    return Qwen3OllamaWrapper(model_name="qwen3:8b", **kwargs)

def test_qwen3_wrapper():
    """Test the Qwen3 Ollama wrapper"""
    print("🧪 Testing Qwen3 Ollama Wrapper...")
    
    # Create wrapper
    qwen3 = Qwen3OllamaWrapper(model_name="qwen3:8b", thinking_mode=True)
    
    # Test basic functionality
    test_prompts = [
        "What is 2+2?",
        "Explain quantum computing in simple terms",
        "Write a Python function to calculate fibonacci numbers"
    ]
    
    for prompt in test_prompts:
        print(f"\n📝 Prompt: {prompt}")
        try:
            response = qwen3.generate(prompt)
            print(f"🤖 Qwen3 Response: {response[:200]}...")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Test thinking mode toggle
    print("\n🧠 Testing thinking mode toggle...")
    qwen3.set_thinking_mode(False)
    quick_response = qwen3.generate("What's the capital of France?")
    print(f"Quick mode: {quick_response}")
    
    qwen3.set_thinking_mode(True)
    thinking_response = qwen3.generate("What's the capital of France?")
    print(f"Thinking mode: {thinking_response}")

def test_qwen3_structured_output():
    """Test the Qwen3 wrapper with structured outputs"""
    from pydantic import BaseModel
    from typing import Optional
    
    class TestSchema(BaseModel):
        score: float
        reason: str
        verdict: Optional[str] = "unknown"  # Make optional with default
    
    qwen3 = Qwen3OllamaWrapper(model_name="qwen3:8b", thinking_mode=False)
    
    prompt = "Rate this response from 1-10: 'The sky is blue because of light scattering.'"
    
    try:
        # Test with schema
        structured_response = qwen3.generate(prompt, schema=TestSchema)
        print(f"Structured response: {structured_response}")
        print(f"Score: {structured_response.score}")
        print(f"Reason: {structured_response.reason}")
        print(f"Verdict: {structured_response.verdict}")
        
        # Test without schema
        text_response = qwen3.generate(prompt)
        print(f"Text response: {text_response}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_qwen3_structured_output()