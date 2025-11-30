"""
Transcript parsing and validation utilities
"""

from typing import Dict, List, Any
import time


class TranscriptParser:
    """
    Handles parsing and validation of meeting transcripts
    """
    
    @staticmethod
    def validate_transcript(transcript_data: Dict) -> None:
        """Validate transcript data structure"""
        required_fields = ["meeting_topic", "experts", "transcript"]
        
        for field in required_fields:
            if field not in transcript_data:
                raise ValueError(f"Missing required field: {field}")
        
        if not transcript_data["transcript"]:
            raise ValueError("Transcript is empty")
        
        if len(transcript_data["experts"]) == 0:
            raise ValueError("No experts defined")
    
    @staticmethod
    def parse_transcript(transcript: List[Dict]) -> List[Dict]:
        """Parse transcript into structured response format"""
        structured_responses = []
        
        for entry in transcript:
            # Determine response type based on agent name patterns
            response_type = TranscriptParser._classify_response_type(entry["name"])
            
            # Extract round number (default to 1 if not specified)
            round_num = entry.get("round", 1)
            
            structured_response = {
                "agent": entry["name"],
                "content": entry["content"],
                "round": round_num,
                "type": response_type,
                "timestamp": time.time(),
                "word_count": len(entry["content"].split()),
                "character_count": len(entry["content"])
            }
            
            structured_responses.append(structured_response)
        
        return structured_responses
    
    @staticmethod
    def _classify_response_type(agent_name: str) -> str:
        """Classify response type based on agent name"""
        agent_name_lower = agent_name.lower()
        
        if any(keyword in agent_name_lower for keyword in ["meeting", "round", "summary", "final"]):
            return "system"
        elif "critic" in agent_name_lower:
            return "critic"
        elif "feedback" in agent_name_lower or "pi" in agent_name_lower:
            return "facilitator"
        elif agent_name.startswith("#") or agent_name.startswith("**"):
            return "header"
        else:
            return "expert"
    
    @staticmethod
    def get_agent_info(agent_name: str, experts: List[Dict]) -> Dict:
        """Get agent information from experts list"""
        return next(
            (exp for exp in experts if exp.get('title') == agent_name), 
            {"title": agent_name, "expertise": "Unknown", "role": "Unknown", "goal": "Unknown"}
        )