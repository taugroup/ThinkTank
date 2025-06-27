from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class FileData(BaseModel):
    filename: str
    content: str  # base64-encoded file content

class FileReference(BaseModel):
    original_name: str
    size: int

class Expert(BaseModel):
    title: str
    expertise: str
    goal: str
    role: str

class Meeting(BaseModel):
    project_name: str
    experts: List[Expert]
    vector_store: Optional[List[List[FileData]]] = []  # Legacy support for base64 files
    file_references: Optional[Dict[str, List[FileReference]]] = {}  # New: expert_name -> file_references
    session_id: Optional[str] = None  # For file upload sessions
    meeting_topic: str
    rounds: int
    timestamp: Optional[int] = None
    transcript: Optional[List[Dict[str, str]]] = []
    summary: Optional[str] = ""

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
    