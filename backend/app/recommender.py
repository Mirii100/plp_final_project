import pandas as pd
import joblib
import numpy as np
import json
from app.custom_transformers import MLBWrapper

class Recommender:
    def __init__(self, courses_path, careers_path, model_path, metrics_path):
        print("Initializing Recommender with Random Forest model...")
        self.courses_df = pd.read_csv(courses_path).fillna('N/A')
        self.careers_df = pd.read_csv(careers_path).fillna('N/A')
        self.model_pipeline = joblib.load(model_path)
        
        with open(metrics_path, 'r') as f:
            self.metrics = json.load(f)
        
        print("Model and metrics loaded successfully.")

        self.grade_points = {
            'A': 12, 'A-': 11, 'B+': 10, 'B': 9, 'B-': 8,
            'C+': 7, 'C': 6, 'C-': 5, 'D+': 4, 'D': 3, 'D-': 2, 'E': 1
        }
        self.all_subjects = ['Mathematics', 'Kiswahili', 'English', 'Arabic', 'German', 'French', 'Chemistry', 'Physics', 'Biology', 'Home Science', 'Agriculture', 'Computer Studies', 'History', 'Geography', 'Religious Education', 'Life Skills', 'Business Studies', 'Music', 'Art and Design', 'Drawing and Design', 'Building Construction', 'Power and Mechanics', 'Metalwork', 'Aviation', 'Woodwork', 'Electronics']

    def _get_profile_rating(self, avg_points):
        if avg_points >= 10:
            return "Excellent Profile"
        elif avg_points >= 8:
            return "Strong Profile"
        elif avg_points >= 6:
            return "Good Profile"
        else:
            return "Developing Profile"

    def _generate_reasoning(self, student_df, item_features):
        return "Recommended based on a predictive model trained on thousands of student profiles. It has identified this as a strong match for your academic and personal profile."

    def recommend(self, student_input):
        total_points = 0
        num_subjects = 0
        subject_grades_points = []
        student_data = {}

        for subject in self.all_subjects:
            grade = student_input.grades.get(subject, 'E')
            points = self.grade_points.get(grade.upper(), 1)
            student_data[subject] = points
            if subject in student_input.grades:
                total_points += points
                num_subjects += 1
                subject_grades_points.append({"subject": subject, "grade": grade.upper(), "points": points})

        average_points = total_points / num_subjects if num_subjects > 0 else 0
        profile_rating = self._get_profile_rating(average_points)
        
        student_data['interests'] = [student_input.interests]
        student_data['skills'] = [student_input.skills]

        student_df = pd.DataFrame(student_data)

        probabilities = self.model_pipeline.predict_proba(student_df)[0]
        model_classes = self.model_pipeline.classes_

        top_5_indices = np.argsort(probabilities)[::-1][:5]
        
        course_recommendations = []
        for i in top_5_indices:
            course_id = model_classes[i]
            score = probabilities[i]
            
            course_info = self.courses_df[self.courses_df['course_id'] == course_id].iloc[0]
            
            course_recommendations.append({
                "name": course_info['course_name'],
                "type": "course",
                "similarity_score": score,
                "description": course_info['description'],
                "reasoning": self._generate_reasoning(student_df, course_info['required_subjects']),
                "job_applicability": course_info['job_applicability'],
                "future_trends": course_info['future_trends'],
                "automation_risk": course_info['automation_risk']
            })

        career_recommendations = self.get_similar_careers(student_input)

        return {
            "average_points": average_points,
            "profile_rating": profile_rating,
            "model_accuracy": self.metrics.get("course_model_accuracy", 0.0),
            "subject_grades_points": subject_grades_points,
            "courses": course_recommendations,
            "careers": career_recommendations
        }

    def get_similar_careers(self, student_input):
        print("Warning: Career recommendation is using a placeholder method.")
        return self.careers_df.head(5).apply(lambda row: {
            "name": row['career_name'],
            "type": "career",
            "similarity_score": 0.0,
            "description": row['description'],
            "reasoning": "This is a general suggestion. A specific career prediction model has not been implemented yet.",
            "job_applicability": row['job_applicability'],
            "future_trends": row['future_trends'],
            "automation_risk": row['automation_risk']
        }, axis=1).tolist()
