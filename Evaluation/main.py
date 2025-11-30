"""
Main entry point for the meeting evaluation system
"""

import json
import time
from typing import Dict, Any, Optional

from .core import EvaluationManager
from .utils import Config, validate_api_keys, load_transcript_file


class EvaluationSystem:
    """
    Main evaluation system interface
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()    
    def evaluate_transcript_file(self, file_path: str) -> Dict[str, Any]:
        """Evaluate a meeting transcript from file"""
        transcript_data = load_transcript_file(file_path)
        return self.evaluate_transcript_data(transcript_data)    
    def evaluate_transcript_data(self, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate meeting transcript data"""
        # Validate API keys
        available_evaluators = validate_api_keys()
        if not any(available_evaluators.values()):
            raise ValueError("No API keys found. Please set GEMINI_API_KEY, GOOGLE_API_KEY, or GROQ_API_KEY")
        
        print(f"🔑 Available evaluators: {', '.join([k.title() for k, v in available_evaluators.items() if v])}")
        
        # Initialize evaluation manager
        eval_config = self.config.get_evaluation_config()
        
        eval_manager = EvaluationManager(
            meeting_topic=transcript_data["meeting_topic"],
            experts=transcript_data["experts"],
            project_name=transcript_data.get("project_name", "Unknown Project"),
            max_workers=eval_config.get("max_workers", 3)
        )
        
        try:
            # Evaluate the transcript
            evaluation_report = eval_manager.evaluate_transcript(transcript_data)
            
            # Save report if configured
            output_config = self.config.get_output_config()
            if output_config.get("save_detailed_report", True):
                report_filename = output_config.get("report_filename", "meeting_evaluation_report.json")
                with open(report_filename, "w") as f:
                    json.dump(evaluation_report, f, indent=2, default=str)
                print(f"✅ Detailed report saved to '{report_filename}'")
            
            return evaluation_report
            
        finally:
            # Cleanup
            eval_manager.shutdown()    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and configuration"""        
        api_keys = validate_api_keys()        
        return {
            "version": "1.0.0",
            "available_evaluators": [k for k, v in api_keys.items() if v],
            "configuration": {
                "max_workers": self.config.get("evaluation.max_workers"),
                "timeout_per_evaluation": self.config.get("evaluation.timeout_per_evaluation"),
                "retry_attempts": self.config.get("evaluation.retry_attempts")
            },
            "api_keys_status": api_keys
        }    

def evaluate_meeting_transcript(
    transcript_file_path: str = None, 
    transcript_data: Dict = None,
    config: Optional[Config] = None
) -> Dict[str, Any]:
    """
    Main function to evaluate a meeting transcript
    
    Args:
        transcript_file_path: Path to JSON file containing transcript
        transcript_data: Direct transcript data dictionary
        config: Optional configuration object
    
    Returns:
        Comprehensive evaluation report
    """
    system = EvaluationSystem(config)    
    if transcript_file_path:
        return system.evaluate_transcript_file(transcript_file_path)
    elif transcript_data:
        return system.evaluate_transcript_data(transcript_data)
    else:
        raise ValueError("Either transcript_file_path or transcript_data must be provided"