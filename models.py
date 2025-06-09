from typing import List, Dict, Any
from pydantic import BaseModel

class Expert(BaseModel):
    title: str
    expertise: str
    goal: str
    role: str

class Meeting(BaseModel):
    project_name: str
    experts: List[Expert]
    vector_store: List[List[str]]  # List of base64-encoded file bytes
    meeting_topic: str
    rounds: int
    timestamp: int
    transcript: List[Dict[str, str]]  # List of strings representing the transcript
    summary: str

    def serialize(self) -> Dict[str, Any]:
        return {
            "project_name": self.project_name,
            "experts": [expert.dict() for expert in self.experts],
            "vector_store": self.vector_store,
            "meeting_topic": self.meeting_topic,
            "rounds": self.rounds,
            "timestamp": self.timestamp,
            "transcript": self.transcript,
            "summary": self.summary
        }



class Project(BaseModel):
    title: str
    description: str
    meetings: List[Meeting]

    def serialize(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "meetings": [meeting.serialize for meeting in self.meetings]
        }
    