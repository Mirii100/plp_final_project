from pydantic import BaseModel
from typing import List

class SubjectGradePoints(BaseModel):
    subject: str
    grade: str
    points: int

class Recommendation(BaseModel):
    name: str
    type: str
    similarity_score: float
    description: str
    reasoning: str
    job_applicability: str
    future_trends: str
    automation_risk: str

class Recommendations(BaseModel):
    average_points: float
    profile_rating: str
    model_accuracy: float
    subject_grades_points: List[SubjectGradePoints]
    courses: List[Recommendation]
    careers: List[Recommendation]
