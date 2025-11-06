from pydantic import BaseModel
from typing import List

from pydantic import BaseModel
from typing import List, Dict

class Student(BaseModel):
    grades: Dict[str, str]
    interests: List[str]
    skills: List[str]
