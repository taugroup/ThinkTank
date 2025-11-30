"""
Helper utilities for the evaluation system
"""

import os
import json
from typing import Dict, List, Any
from dotenv import load_dotenv


def validate_api_keys() -> Dict[str, bool]:
    """Validate available API keys"""
    load_dotenv()
    
    api_keys = {
        "gemini": bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
        "groq": bool(os.getenv("GROQ_API_KEY")),
        "qwen": True  # Qwen doesn't require API key (local)
    }
    
    return api_keys


def load_transcript_file(file_path: str) -> Dict[str, Any]:
    """Load and validate transcript file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        
        # Basic validation
        required_fields = ["meeting_topic", "experts", "transcript"]
        for field in required_fields:
            if field not in transcript_data:
                raise ValueError(f"Missing required field: {field}")
        
        return transcript_data
        
    except FileNotFoundError:
        raise ValueError(f"Transcript file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in transcript file: {e}")


def get_available_evaluators() -> List[str]:
    """Get list of available evaluators based on API keys"""
    api_keys = validate_api_keys()
    available = []
    
    for evaluator, available_key in api_keys.items():
        if available_key:
            available.append(evaluator)
    
    return available


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def calculate_score_distribution(scores: List[float]) -> Dict[str, Any]:
    """Calculate score distribution statistics"""
    if not scores:
        return {"error": "No scores provided"}
    
    sorted_scores = sorted(scores)
    n = len(scores)
    
    return {
        "count": n,
        "mean": sum(scores) / n,
        "median": sorted_scores[n // 2],
        "min": min(scores),
        "max": max(scores),
        "range": max(scores) - min(scores),
        "quartiles": {
            "q1": sorted_scores[n // 4],
            "q2": sorted_scores[n // 2],
            "q3": sorted_scores[3 * n // 4]
        }
    }


def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """Extract keywords from text"""
    import re
    
    # Simple keyword extraction
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    # Filter by length and common stop words
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
    
    keywords = [word for word in words if len(word) >= min_length and word not in stop_words]
    
    # Return unique keywords
    return list(set(keywords))


def create_summary_stats(data: List[Dict[str, Any]], score_field: str = "composite_score") -> Dict[str, Any]:
    """Create summary statistics for evaluation data"""
    if not data:
        return {"error": "No data provided"}
    
    scores = [item.get(score_field, 0) for item in data if score_field in item]
    
    if not scores:
        return {"error": f"No valid scores found for field: {score_field}"}
    
    return calculate_score_distribution(scores)