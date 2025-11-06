from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .models import Student
from .schemas import Recommendations
from .recommender import Recommender

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender = Recommender(
    courses_path="./data/courses.csv",
    careers_path="./data/careers.csv",
    model_path="./app/course_recommender_model.joblib",
    metrics_path="./app/model_metrics.json"
)

@app.post("/recommend", response_model=Recommendations)
def get_recommendations(student: Student):
    recommendations = recommender.recommend(student)
    return recommendations

@app.get("/")
def read_root():
    return {"message": "Welcome to the Career Guidance API"}
