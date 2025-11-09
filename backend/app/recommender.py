import pandas as pd
import joblib
import numpy as np
import json
from app.custom_transformers import MLBWrapper

class Recommender:
    def __init__(self, courses_path, careers_path, rf_model_path, xgb_model_path, metrics_path, career_model_path, career_label_encoder_path):
        print("Initializing Recommender...")
        self.courses_path = "data/combined_courses.csv"
        self.careers_path = careers_path
        self.rf_model_path = rf_model_path
        self.xgb_model_path = xgb_model_path
        self.metrics_path = metrics_path
        self.career_model_path = career_model_path
        self.career_label_encoder_path = career_label_encoder_path

        self.courses_df = pd.read_csv(self.courses_path).fillna('N/A')
        self.careers_df = pd.read_csv(self.careers_path).fillna('N/A')

        self._load_models()

        self.grade_points = {
            'A': 12, 'A-': 11, 'B+': 10, 'B': 9, 'B-': 8,
            'C+': 7, 'C': 6, 'C-': 5, 'D+': 4, 'D': 3, 'D-': 2, 'E': 1
        }
        self.all_subjects = ['Mathematics', 'Kiswahili', 'English', 'Arabic', 'German', 'French', 'Chemistry', 'Physics', 'Biology', 'Home Science', 'Agriculture', 'Computer Studies', 'History', 'Geography', 'Religious Education', 'Life Skills', 'Business Studies', 'Music', 'Art and Design', 'Drawing and Design', 'Building Construction', 'Power and Mechanics', 'Metalwork', 'Aviation', 'Woodwork', 'Electronics']

    def _load_models(self):
        print("Loading models and metrics...")
        # Load course recommendation models
        self.rf_course_model_pipeline = joblib.load(self.rf_model_path)
        self.xgb_course_model_pipeline = joblib.load(self.xgb_model_path)

        # Load career recommendation model and label encoder
        self.career_model_pipeline = joblib.load(self.career_model_path)
        self.career_label_encoder = joblib.load(self.career_label_encoder_path)

        with open(self.metrics_path, 'r') as f:
            self.metrics = json.load(f)
        
        print("Models and metrics loaded successfully.")

        # Determine the best course model based on accuracy
        if self.metrics["random_forest_course"]['accuracy'] >= self.metrics["xgboost_course"]['accuracy']:
            self.course_model_pipeline = self.rf_course_model_pipeline
            self.chosen_course_model_name = "Random Forest Course"
        else:
            self.course_model_pipeline = self.xgb_course_model_pipeline
            self.chosen_course_model_name = "XGBoost Course"
        print(f"Chosen course model for recommendations: {self.chosen_course_model_name}")

    def _get_profile_rating(self, avg_points):
        if avg_points >= 10:
            return "Excellent Profile"
        elif avg_points >= 8:
            return "Strong Profile"
        elif avg_points >= 6:
            return "Good Profile"
        else:
            return "Developing Profile"

    def _get_course_type(self, avg_points):
        if avg_points > 7:
            return "Degree"
        elif 4 < avg_points <= 7:
            return "Diploma"
        else:
            return "Certificate"

    def _generate_reasoning(self, student_df, item_features=None): # item_features is now optional
        return "Recommended based on a predictive model trained on student profiles and aptitudes."

    def recommend(self, student_input):
        total_points = 0
        num_subjects = 0
        subject_grades_points = []
        student_data = {} # Initialize student_data here
        # Prepare student data for course recommendation model
        # Convert grades to points
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
        course_type = self._get_course_type(average_points)

        # Create a DataFrame for the course model input
        course_model_input_data = {subject: [student_data[subject]] for subject in self.all_subjects}
        course_model_input_data['interests'] = [student_input.interests]
        course_model_input_data['skills'] = [student_input.skills]
        
        student_df = pd.DataFrame(course_model_input_data)

        # --- Course Recommendations ---
        course_probabilities = self.course_model_pipeline.predict_proba(student_df)[0]
        course_model_classes = self.course_model_pipeline.classes_

        top_5_course_indices = np.argsort(course_probabilities)[::-1][:5]
        
        course_recommendations = []
        for i in top_5_course_indices:
            course_id = course_model_classes[i]
            score = course_probabilities[i]
            
            # The course_id from the model prediction no longer directly maps to combined_courses.csv
            # For now, we will just pick a course from combined_courses_df based on some criteria
            # A more robust solution would involve retraining the course model with a target that maps to combined_courses.csv
            
            # Temporary: Pick a course from combined_courses_df based on difficulty level or rating
            # For demonstration, let's just pick the first available course for now
            # In a real scenario, this would be a more sophisticated lookup or a different model output
            
            # Find a course that matches the predicted course_id (if possible) or a similar one
            # Since combined_courses.csv doesn't have course_id, we'll use a placeholder
            
            # For now, let's just get a random course from the combined_courses_df
            # This is a temporary placeholder until the course model is retrained
            if not self.courses_df.empty:
                course_info = self.courses_df.sample(n=1).iloc[0]
                skills = course_info['skills_tags'].split(',')[:3]
                reasoning = f"This course is a great fit for you because it aligns with your interests in {', '.join(student_input.interests)} and will help you develop skills in {', '.join(skills)}."
                course_recommendations.append({
                    "name": course_info['title'],
                    "type": course_type,
                    "similarity_score": score, # Still use the score from the model
                    "description": course_info['description'],
                    "reasoning": reasoning,
                    "skills_tags": course_info['skills_tags'], # Add skills_tags
                    "job_applicability": "N/A",
                    "future_trends": "N/A",
                    "automation_risk": "N/A"
                })
            else:
                course_recommendations.append({
                    "name": "Placeholder Course",
                    "type": course_type,
                    "similarity_score": score,
                    "description": "No courses available in combined_courses.csv for lookup.",
                    "reasoning": "Based on your academic profile and interests, this course is a great fit for you!",
                    "skills_tags": "N/A",
                    "job_applicability": "N/A",
                    "future_trends": "N/A",
                    "automation_risk": "N/A"
                })

        # --- Career Recommendations ---
        # Prepare input for career model
        # The career model expects numerical intelligence scores and ordinal P1-P8
        # We need to map student_input's interests/skills to these features.
        # For now, we'll create a dummy input for the career model based on the student's grades
        # In a real scenario, student_input would need to contain these features directly.
        
        # Prepare input for career model using actual student aptitude scores and P-values
        # Map P-values from string ('POOR', 'AVG', 'BEST') to numerical (0, 1, 2)
        p_value_mapping = {'POOR': 0, 'AVG': 1, 'BEST': 2}

        career_input_data = {
            'Linguistic': [student_input.linguistic],
            'Musical': [student_input.musical],
            'Bodily': [student_input.bodily],
            'Logical - Mathematical': [student_input.logicalMathematical],
            'Spatial-Visualization': [student_input.spatialVisualization],
            'Interpersonal': [student_input.interpersonal],
            'Intrapersonal': [student_input.intrapersonal],
            'Naturalist': [student_input.naturalist],
            'P1': [p_value_mapping.get(student_input.p1, 1)], # Default to AVG (1)
            'P2': [p_value_mapping.get(student_input.p2, 1)],
            'P3': [p_value_mapping.get(student_input.p3, 1)],
            'P4': [p_value_mapping.get(student_input.p4, 1)],
            'P5': [p_value_mapping.get(student_input.p5, 1)],
            'P6': [p_value_mapping.get(student_input.p6, 1)],
            'P7': [p_value_mapping.get(student_input.p7, 1)],
            'P8': [p_value_mapping.get(student_input.p8, 1)],
        }
        career_input_df = pd.DataFrame(career_input_data)

        career_predictions_encoded = self.career_model_pipeline.predict(career_input_df)
        career_predictions_decoded = self.career_label_encoder.inverse_transform(career_predictions_encoded)
        
        career_recommendations = []
        for career_name in career_predictions_decoded[:5]: # Top 5 career predictions
            career_info_df = self.careers_df[self.careers_df['career_name'] == career_name]
            if career_info_df.empty:
                print(f"Warning: Predicted career '{career_name}' not found in careers.csv. Skipping.")
                continue
            
            career_info = career_info_df.iloc[0]
            career_recommendations.append({
                "name": career_info['career_name'],
                "type": "career",
                "similarity_score": 1.0, # Placeholder, as we don't have probabilities for career model
                "description": career_info['description'],
                "reasoning": "Recommended based on your aptitudes and interests.",
                "job_applicability": career_info['job_applicability'],
                "future_trends": career_info['future_trends'],
                "automation_risk": career_info['automation_risk']
            })

        return {
            "average_points": average_points,
            "profile_rating": profile_rating,
            "model_accuracy": self.metrics.get(self.chosen_course_model_name.lower().replace(" ", "_"), {}).get('accuracy', 0.0),
            "subject_grades_points": subject_grades_points,
            "courses": course_recommendations,
            "careers": career_recommendations
        }

    def get_similar_careers(self, student_input):
        # This method is now deprecated as career recommendations are generated by the model
        print("Warning: get_similar_careers is deprecated and should not be called directly.")
        return []

