"""
Configuration management for the evaluation system
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv


class Config:
    """
    Configuration manager for evaluation system
    """
    
    def __init__(self):
        load_dotenv()
        self._config = self._load_default_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            "evaluators": {
                "gemini": {
                    "model_name": "gemini-2.5-flash",
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "api_key_env": ["GEMINI_API_KEY", "GOOGLE_API_KEY"]
                },
                "groq": {
                    "model_name": "llama-3.1-70b-versatile",
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "api_key_env": ["GROQ_API_KEY"]
                },
                "qwen": {
                    "model_name": "qwen3:8b",
                    "base_url": "http://localhost:11434",
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "thinking_mode": True
                }
            },
            "evaluation": {
                "max_workers": 3,
                "timeout_per_evaluation": 300,  # 5 minutes
                "retry_attempts": 2,
                "metric_weights": {
                    "Agent Collaboration Quality": 0.25,
                    "Agent Role Adherence": 0.20,
                    "Agent Response Quality": 0.25,
                    "Meeting Progress Assessment": 0.30
                }
            },
            "output": {
                "save_detailed_report": True,
                "report_filename": "meeting_evaluation_report.json",
                "include_conversation_analysis": True,
                "include_statistical_summary": True
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_api_key(self, evaluator_name: str) -> str:
        """Get API key for specific evaluator"""
        api_key_envs = self.get(f"evaluators.{evaluator_name}.api_key_env", [])
        
        for env_var in api_key_envs:
            api_key = os.getenv(env_var)
            if api_key:
                return api_key
        
        return None
    
    def get_evaluator_config(self, evaluator_name: str) -> Dict[str, Any]:
        """Get configuration for specific evaluator"""
        return self.get(f"evaluators.{evaluator_name}", {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration"""
        return self.get("evaluation", {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration"""
        return self.get("output", {})