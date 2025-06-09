from typing import List, Dict, Any
from pydantic import BaseModel

class Expert(BaseModel):
    title: str
    expertise: str
    goal: str
    role: str

class Meeting(BaseModel):
    id: str
    projectTitle: str;
    topic: str;
    timestamp: int
    rounds: int;
    transcript: str;
    summary: str;
    experts: List[Expert];

    def serialize(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "projectTitle": self.projectTitle,
            "topic": self.topic,
            "timestamp": self.timestamp,
            "rounds": self.rounds,
            "transcript": self.transcript,
            "summary": self.summary,
            "experts": [expert.dict() for expert in self.experts]
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
    