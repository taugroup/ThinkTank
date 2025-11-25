"""
Base evaluator class for all LLM evaluators
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, Dict, Any
from pydantic import BaseModel
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics.g_eval.schema import Steps
import time


class BaseEvaluator(DeepEvalBaseLLM, ABC):
    """
    Abstract base class for all evaluators
    """
    
    def __init__(self, model_name: str, temperature: float = 0.1, max_tokens: int = 2000, **kwargs):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        self.client = None
        self._initialized = False
    
    @abstractmethod
    def _initialize_client(self):
        """Initialize the LLM client"""
        pass
    
    @abstractmethod
    def _call_llm(self, prompt: str, **kwargs) -> str:
        """Make a call to the LLM"""
        pass
    
    def load_model(self):
        """Load and initialize the model"""
        if not self._initialized:
            self._initialize_client()
            self._initialized = True
        return self.model_name
    
    def get_model_name(self) -> str:
        """Return the model name"""
        return self.model_name
    
    def generate(self, prompt: str, schema: Optional[BaseModel] = None, **kwargs) -> Union[str, BaseModel]:
        """
        Generate response with optional schema support
        """
        try:
            if not self._initialized:
                self.load_model()
            
            # Get raw response
            response_text = self._call_llm(prompt, **kwargs)
            
            # Handle schema-based responses
            if schema is not None:
                if schema == Steps:
                    return Steps(steps=[response_text])
                return self._generate_with_schema(response_text, schema)
            
            # Check if this looks like an evaluation call
            if self._is_evaluation_prompt(prompt):
                return Steps(steps=[response_text])
            
            return response_text
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            if schema is not None:
                if schema == Steps:
                    return Steps(steps=[error_msg])
                return self._create_fallback_response(schema, error_msg)
            
            if self._is_evaluation_prompt(prompt):
                return Steps(steps=[error_msg])
            
            return error_msg
    
    async def a_generate(self, prompt: str, schema: Optional[BaseModel] = None, **kwargs) -> Union[str, BaseModel]:
        """Async version of generate"""
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None, self.generate, prompt, schema, **kwargs
        )
    
    def _is_evaluation_prompt(self, prompt: str) -> bool:
        """Check if prompt looks like an evaluation request"""
        evaluation_keywords = ['evaluate', 'score', 'rate', 'assess', 'criteria', 'judge']
        return any(keyword in prompt.lower() for keyword in evaluation_keywords)
    
    def _generate_with_schema(self, response_text: str, schema: BaseModel) -> BaseModel:
        """Generate structured response from text using schema"""
        try:
            import json
            # Try to parse as JSON first
            cleaned_response = self._clean_json_response(response_text)
            response_dict = json.loads(cleaned_response)
            return schema(**response_dict)
        except (json.JSONDecodeError, ValueError):
            return self._create_fallback_response(schema, response_text)
    
    def _clean_json_response(self, response_text: str) -> str:
        """Clean response text to extract valid JSON"""
        import re
        
        cleaned = response_text.strip()
        
        # Remove markdown code blocks
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        
        cleaned = cleaned.strip()
        
        # Try to find JSON object
        first_brace = cleaned.find('{')
        last_brace = cleaned.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return cleaned[first_brace:last_brace + 1]
        
        return cleaned
    
    def _create_fallback_response(self, schema: BaseModel, response_text: str) -> BaseModel:
        """Create a fallback response when JSON parsing fails"""
        import re
        
        try:
            schema_fields = schema.model_fields
            fallback_data = {}
            
            # Handle common DeepEval schemas
            if 'steps' in schema_fields:
                fallback_data['steps'] = [response_text] if response_text else ["Unable to evaluate"]
            elif 'verdict' in schema_fields:
                verdict_keywords = ['yes', 'true', 'correct', 'good', 'pass']
                fallback_data['verdict'] = 'yes' if any(keyword in response_text.lower() for keyword in verdict_keywords) else 'no'
                if 'reason' in schema_fields:
                    fallback_data['reason'] = response_text or "No reason provided"
            elif 'score' in schema_fields:
                numbers = re.findall(r'\d+\.?\d*', response_text)
                if numbers:
                    try:
                        score = float(numbers[0])
                        score = max(0, min(10, score))
                        fallback_data['score'] = score
                    except ValueError:
                        fallback_data['score'] = 5.0
                else:
                    fallback_data['score'] = 5.0
                
                if 'reason' in schema_fields:
                    fallback_data['reason'] = response_text or "No detailed reasoning provided"
            elif 'reason' in schema_fields:
                fallback_data['reason'] = response_text or "No reasoning provided"
            
            # Handle remaining required fields
            for field_name, field_info in schema_fields.items():
                if field_name not in fallback_data:
                    field_type = field_info.annotation
                    
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
            
        except Exception:
            # Last resort: create minimal valid instance
            try:
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
            except Exception:
                return schema()