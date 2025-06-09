from typing import List, Dict, Any
from pydantic import BaseModel

class Scientist(BaseModel):
    title: str
    expertise: str
    goal: str
    role: str

class MeetingRequest(BaseModel):
    project_name: str
    project_desc: str
    scientists: List[Scientist]
    meeting_topic: str
    rounds: int

class ProjectInfo(BaseModel):
    description: str
    scientists: List[Scientist]