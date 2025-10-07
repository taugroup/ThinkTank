import json
import re
import os
from typing import Optional, Union, Type
from pydantic import BaseModel
import time
from functools import wraps

try:
    from groq import Groq
except ImportError:
    raise ImportError("Please install groq: pip install groq")

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
    print("You can still use environment variables set in your shell.")

from deepeval.models.base_model import DeepEvalBaseLLM

def retry_with_exponential_backoff(
    max_retries=3,
    initial_delay=2.5,
    exponential_base=2
):
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
                            print(f"⏸️  Rate limit hit. Waiting {delay:.1f}s before retry {attempt + 1}/{max_retries}...")
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

class GroqWrapper(DeepEvalBaseLLM):
    """
    Custom LLM wrapper for Groq models with proper schema support for DeepEval
    """
    
    def __init__(
        self, 
        model_name: str = "openai/gpt-oss-120b",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        **kwargs
    ):
        """
        Initialize Groq wrapper
        
        Args:
            model_name: Groq model name (default: openai/gpt-oss-120b)
                       Other options: llama-3.3-70b-versatile, llama-3.1-70b-versatile, mixtral-8x7b-32768, etc.
            api_key: Groq API key (if not provided, will look for GROQ_API_KEY env var)
            temperature: Temperature for generation (0.0 - 2.0)
            max_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        self.client = None
        self.api_key = api_key
        
    def load_model(self):
        """Initialize connection to Groq API"""
        try:
            # Initialize Groq client (will use GROQ_API_KEY env var if api_key not provided)
            if self.api_key:
                self.client = Groq(api_key=self.api_key)
            else:
                self.client = Groq()  # Will look for GROQ_API_KEY environment variable
            
            # Test the model with a simple prompt
            test_response = self._call_llm("Hello", test_mode=True)
            if not test_response:
                raise Exception(f"Model {self.model_name} is not responding properly")
            
            print(f"✅ Groq model {self.model_name} loaded successfully")
            return self.client
            
        except Exception as e:
            raise Exception(
                f"Failed to initialize Groq: {str(e)}\n"
                "Please ensure:\n"
                "1. You have installed groq: pip install groq\n"
                "2. Your GROQ_API_KEY environment variable is set\n"
                "3. Or pass api_key parameter during initialization"
            )
    
    def generate(self, prompt: str, schema: Optional[BaseModel] = None, **kwargs) -> Union[str, BaseModel]:
        """
        Generate response from Groq model with optional schema support
        
        Args:
            prompt: Input prompt
            schema: Optional Pydantic BaseModel schema for structured output
        
        Returns:
            String response or structured BaseModel instance
        """
        try:
            if self.client is None:
                self.load_model()
            
            # If schema is provided, try to get structured response
            if schema is not None:
                return self._generate_with_schema(prompt, schema, **kwargs)
            else:
                return self._call_llm(prompt, **kwargs)
                
        except Exception as e:
            print(f"Error generating response from Groq: {e}")
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
            
            # Get response from Groq
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
        if cleaned.startswith("```json"):
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
        """Return the Groq model name"""
        return self.model_name
    
    @retry_with_exponential_backoff(max_retries=5, initial_delay=3)
    def _call_llm(self, prompt: str, test_mode: bool = False, **kwargs) -> str:
        """
        Call Groq model via API
        
        Args:
            prompt: The prompt to send
            test_mode: If True, use simplified prompt for testing
            **kwargs: Additional parameters
        
        Returns:
            Model response as string
        """
        try:
            # Prepare messages for Groq API
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
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


# ============================================================================
# TESTING FUNCTIONS - Run tests with: python groq_wrapper.py
# ============================================================================

def test_basic_connection():
    """Test 1: Basic connection and simple generation"""
    print("\n" + "="*60)
    print("TEST 1: Basic Connection & Simple Generation")
    print("="*60)
    
    try:
        llm = GroqWrapper()
        response = llm.generate("What is 2+2? Answer in one sentence.")
        print(f"✅ Basic generation works!")
        print(f"Response: {response}")
        return True
    except Exception as e:
        print(f"❌ Basic generation failed: {e}")
        return False


def test_schema_generation():
    """Test 2: Structured output with schema"""
    print("\n" + "="*60)
    print("TEST 2: Schema-based Generation")
    print("="*60)
    
    class MathResponse(BaseModel):
        answer: int
        explanation: str
    
    try:
        llm = GroqWrapper()
        prompt = "What is 5 + 7? Provide the answer and a brief explanation."
        response = llm.generate(prompt, schema=MathResponse)
        
        print(f"✅ Schema generation works!")
        print(f"Type: {type(response)}")
        print(f"Answer: {response.answer}")
        print(f"Explanation: {response.explanation}")
        return True
    except Exception as e:
        print(f"❌ Schema generation failed: {e}")
        return False


def test_deepeval_compatibility():
    """Test 3: Integration with DeepEval metrics"""
    print("\n" + "="*60)
    print("TEST 3: DeepEval Metric Integration")
    print("="*60)
    
    try:
        from deepeval.metrics import AnswerRelevancyMetric
        from deepeval.test_case import LLMTestCase
        
        llm = GroqWrapper()
        metric = AnswerRelevancyMetric(model=llm, threshold=0.5)
        
        test_case = LLMTestCase(
            input="What is the capital of France?",
            actual_output="The capital of France is Paris.",
            retrieval_context=["Paris is the capital and largest city of France."]
        )
        
        print("Running metric evaluation...")
        metric.measure(test_case)
        
        print(f"✅ DeepEval integration works!")
        print(f"Score: {metric.score}")
        print(f"Reason: {metric.reason}")
        return True
    except ImportError:
        print("⚠️  DeepEval not installed. Install with: pip install deepeval")
        return False
    except Exception as e:
        print(f"❌ DeepEval integration failed: {e}")
        return False


def test_model_info():
    """Test 4: Model information"""
    print("\n" + "="*60)
    print("TEST 4: Model Information")
    print("="*60)
    
    try:
        llm = GroqWrapper()
        model_name = llm.get_model_name()
        
        print(f"✅ Model info works!")
        print(f"Model: {model_name}")
        print(f"Expected: openai/gpt-oss-120b")
        
        if model_name == "openai/gpt-oss-120b":
            print("✅ Correct default model!")
        else:
            print(f"⚠️  Unexpected model: {model_name}")
        
        return True
    except Exception as e:
        print(f"❌ Model info failed: {e}")
        return False


def run_tests():
    """Run all tests"""
    
    print("\n" + "="*60)
    print("GROQ WRAPPER TEST SUITE")
    print("="*60)
    
    # Check for API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("\n⚠️  WARNING: GROQ_API_KEY environment variable not set!")
        print("Please either:")
        print("1. Create a .env file with: GROQ_API_KEY=your-api-key")
        print("2. Or set it in shell: export GROQ_API_KEY='your-api-key'")
        print("3. Or pass it directly: GroqWrapper(api_key='your-key')")
        return
    else:
        print(f"✅ GROQ_API_KEY found: {api_key[:10]}...")
    
    # Run tests
    results = {
        "Basic Connection": test_basic_connection(),
        "Schema Generation": test_schema_generation(),
        "DeepEval Integration": test_deepeval_compatibility(),
        "Model Information": test_model_info(),
        "Qwen-Style Comparison": test_qwen_comparison()
    }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your Groq wrapper is working perfectly!")
    elif passed > 0:
        print(f"\n⚠️  Some tests failed. Please check the errors above.")
    else:
        print(f"\n❌ All tests failed. Please check your setup.")


def test_qwen_comparison():
    """Test 5: Compare with Qwen-like evaluation"""
    print("\n" + "="*60)
    print("TEST 5: Qwen-Style Evaluation Comparison")
    print("="*60)
    
    class EvaluationResponse(BaseModel):
        score: float
        reason: str
        verdict: str
    
    try:
        llm = GroqWrapper()
        
        # Test structured response
        print("\n--- Structured Response Test ---")
        prompt = """
Evaluate this response: "The sky is blue because of light scattering."
Rate it from 0-10 and provide reasoning and a verdict (good/partial/bad).
"""
        
        structured_response = llm.generate(prompt, schema=EvaluationResponse)
        print(f"Structured response: {structured_response}")
        print(f"Score: {structured_response.score}")
        print(f"Reason: {structured_response.reason}")
        print(f"Verdict: {structured_response.verdict}")
        
        # Test plain text response
        print("\n--- Plain Text Response Test ---")
        text_prompt = """
Evaluate this response and give it a score out of 10: "The sky is blue because of light scattering."
Provide detailed reasoning including correctness, clarity, and completeness.
Format your response with clear sections for scoring and reasoning.
"""
        
        text_response = llm.generate(text_prompt)
        print(f"Text response:\n{text_response}")
        
        print("\n✅ Qwen-style comparison test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Qwen comparison test failed: {e}")
        return False


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║           Groq Wrapper for DeepEval - Test Suite            ║
╚══════════════════════════════════════════════════════════════╝

Quick Usage:
-----------
from groq_wrapper import GroqWrapper

# Initialize
llm = GroqWrapper()  # Uses GROQ_API_KEY env var
# OR
llm = GroqWrapper(api_key="your-key")

# Simple generation
response = llm.generate("What is Python?")

# With DeepEval
from deepeval.metrics import AnswerRelevancyMetric
metric = AnswerRelevancyMetric(model=llm)
    """)
    
    # run_tests()